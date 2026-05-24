from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

WIDTH = 7
HEIGHT = 6
STRIDE = HEIGHT + 1
RECORD_SIZE = 16 + 1 + 4 + 4

# Binary layout per record: 16-byte state key | 1-byte action | f32 q_value | u32 visits
_RECORD_FMT = struct.Struct("<16sBfI")

# Pre-computed bit masks for every (col, row_from_bottom) position — shape (WIDTH, HEIGHT)
# Avoids per-sample Python loops in encode_board_current_player.
_BIT_INDICES = (
    np.arange(WIDTH, dtype=np.int64)[:, None] * STRIDE
    + np.arange(HEIGHT, dtype=np.int64)[None, :]
)                              # (7, 6)
_BIT_MASKS = np.int64(1) << _BIT_INDICES   # (7, 6)


def decode_state_key(state_key: int) -> tuple[int, int, int]:
    player_a_bits = state_key & ((1 << 64) - 1)
    current_player = int((state_key >> 127) & 1)
    player_b_bits = (state_key >> 64) & ((1 << 63) - 1)
    return player_a_bits, player_b_bits, current_player


def encode_board_current_player(state_key: int) -> torch.Tensor:
    player_a_bits, player_b_bits, current_player = decode_state_key(state_key)

    current_bits  = player_a_bits if current_player == 0 else player_b_bits
    opponent_bits = player_b_bits if current_player == 0 else player_a_bits

    # Vectorised: check all 42 bits at once with pre-computed masks.
    # _BIT_MASKS shape: (WIDTH, HEIGHT); result shape after flip+T: (HEIGHT, WIDTH)
    cur = (np.int64(current_bits)  & _BIT_MASKS) != 0   # (WIDTH, HEIGHT) bool
    opp = (np.int64(opponent_bits) & _BIT_MASKS) != 0

    # row_from_top = HEIGHT - 1 - row_from_bottom → flip the height axis, then transpose
    board_np = np.empty((2, HEIGHT, WIDTH), dtype=np.float32)
    board_np[0] = np.flip(cur, axis=1).T
    board_np[1] = np.flip(opp, axis=1).T

    return torch.from_numpy(board_np.copy())


@dataclass(frozen=True)
class QValueRecord:
    state_key: int
    action: int
    q_value: float
    visits: int


class VValuesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Groups Q-value records by state and targets V(s) = max_a Q(s, a)."""

    def __init__(
        self,
        path: str | Path,
        *,
        min_visits: int = 1,
        max_samples: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.states: list[tuple[int, float, int]] = []  # (state_key, v_value, total_visits)
        self._load(min_visits, max_samples)

    def _load(self, min_visits: int, max_samples: int | None) -> None:
        size = self.path.stat().st_size
        if size % RECORD_SIZE != 0:
            raise ValueError(f"Invalid q_values size: {size} is not divisible by {RECORD_SIZE}")

        # Aggregate per state: track best Q-value and total visits
        best_q: dict[int, float] = {}
        total_v: dict[int, int] = {}

        with self.path.open("rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as data:
                for state_bytes, action, q_value, visits in _RECORD_FMT.iter_unpack(data):
                    if visits < min_visits or action >= WIDTH:
                        continue
                    state_key = int.from_bytes(state_bytes, "little")
                    if state_key not in best_q or q_value > best_q[state_key]:
                        best_q[state_key] = q_value
                    total_v[state_key] = total_v.get(state_key, 0) + visits

        for state_key, v in best_q.items():
            self.states.append((state_key, v, total_v[state_key]))
            if max_samples is not None and len(self.states) >= max_samples:
                break

        if not self.states:
            raise ValueError(f"No usable records in {self.path} with min_visits={min_visits}")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        state_key, v_value, visits = self.states[index]
        x = encode_board_current_player(state_key)
        y = torch.tensor([v_value], dtype=torch.float32)
        w = torch.tensor([float(visits)], dtype=torch.float32)
        return x, y, w


class QValuesDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        *,
        min_visits: int = 1,
        max_samples: int | None = None,
        validate_actions: bool = True,
    ) -> None:
        self.path = Path(path)
        self.min_visits = min_visits
        self.max_samples = max_samples
        self.validate_actions = validate_actions
        self.records = self._load_records()

    def _load_records(self) -> list[QValueRecord]:
        size = self.path.stat().st_size
        if size % RECORD_SIZE != 0:
            raise ValueError(
                f"Invalid q_values size: {size} is not divisible by {RECORD_SIZE}"
            )

        records: list[QValueRecord] = []
        with self.path.open("rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as data:
                for state_bytes, action, q_value, visits in _RECORD_FMT.iter_unpack(data):
                    if visits < self.min_visits:
                        continue
                    if self.validate_actions and action >= WIDTH:
                        continue

                    state_key = int.from_bytes(state_bytes, "little")
                    records.append(QValueRecord(state_key, action, q_value, visits))

                    if self.max_samples is not None and len(records) >= self.max_samples:
                        break

        if not records:
            raise ValueError(
                f"No usable records found in {self.path} with min_visits={self.min_visits}"
            )

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]

        x = encode_board_current_player(record.state_key)

        y = torch.zeros(WIDTH, dtype=torch.float32)
        mask = torch.zeros(WIDTH, dtype=torch.float32)
        y[record.action] = record.q_value
        mask[record.action] = 1.0

        visits = torch.tensor(float(record.visits), dtype=torch.float32)
        return x, y, mask, visits
