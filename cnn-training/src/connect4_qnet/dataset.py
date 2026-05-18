from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

WIDTH = 7
HEIGHT = 6
STRIDE = HEIGHT + 1
RECORD_SIZE = 16 + 1 + 4 + 4


def decode_state_key(state_key: int) -> tuple[int, int, int]:
    player_a_bits = state_key & ((1 << 64) - 1)
    current_player = int((state_key >> 127) & 1)
    player_b_bits = (state_key >> 64) & ((1 << 63) - 1)
    return player_a_bits, player_b_bits, current_player


def encode_board_current_player(state_key: int) -> torch.Tensor:
    player_a_bits, player_b_bits, current_player = decode_state_key(state_key)

    if current_player == 0:
        current_bits = player_a_bits
        opponent_bits = player_b_bits
    else:
        current_bits = player_b_bits
        opponent_bits = player_a_bits

    board = torch.zeros((2, HEIGHT, WIDTH), dtype=torch.float32)

    for col in range(WIDTH):
        for row_from_bottom in range(HEIGHT):
            bit_index = col * STRIDE + row_from_bottom
            bit = 1 << bit_index
            row = HEIGHT - 1 - row_from_bottom

            if current_bits & bit:
                board[0, row, col] = 1.0
            if opponent_bits & bit:
                board[1, row, col] = 1.0

    return board


@dataclass(frozen=True)
class QValueRecord:
    state_key: int
    action: int
    q_value: float
    visits: int


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
                total = size // RECORD_SIZE

                for index in range(total):
                    offset = index * RECORD_SIZE
                    state_key = int.from_bytes(
                        data[offset : offset + 16],
                        byteorder="little",
                        signed=False,
                    )
                    offset += 16

                    action = data[offset]
                    offset += 1

                    q_value = struct.unpack_from("<f", data, offset)[0]
                    offset += 4

                    visits = struct.unpack_from("<I", data, offset)[0]

                    if visits < self.min_visits:
                        continue
                    if self.validate_actions and action >= WIDTH:
                        continue

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
