from __future__ import annotations

import argparse
import mmap
import struct
from collections import Counter
from pathlib import Path

from .dataset import RECORD_SIZE, WIDTH, decode_state_key


def inspect(path: Path, limit: int | None) -> None:
    size = path.stat().st_size
    if size % RECORD_SIZE != 0:
        raise ValueError(f"{path} size {size} is not divisible by {RECORD_SIZE}")

    total = size // RECORD_SIZE
    action_counts: Counter[int] = Counter()
    current_player_counts: Counter[int] = Counter()
    invalid_actions = 0
    min_q = float("inf")
    max_q = float("-inf")
    min_visits = 2**32 - 1
    max_visits = 0
    visited = 0

    with path.open("rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as data:
            count = total if limit is None else min(total, limit)
            for index in range(count):
                offset = index * RECORD_SIZE
                state_key = int.from_bytes(data[offset : offset + 16], "little")
                offset += 16
                action = data[offset]
                offset += 1
                q_value = struct.unpack_from("<f", data, offset)[0]
                offset += 4
                visits = struct.unpack_from("<I", data, offset)[0]

                _, _, current_player = decode_state_key(state_key)
                action_counts[action] += 1
                current_player_counts[current_player] += 1
                invalid_actions += int(action >= WIDTH)
                min_q = min(min_q, q_value)
                max_q = max(max_q, q_value)
                min_visits = min(min_visits, visits)
                max_visits = max(max_visits, visits)
                visited += int(visits > 0)

    inspected = total if limit is None else min(total, limit)
    print(f"path={path}")
    print(f"record_size={RECORD_SIZE}")
    print(f"total_records={total}")
    print(f"inspected_records={inspected}")
    print(f"invalid_actions={invalid_actions}")
    print(f"visited_records={visited}")
    print(f"q_range=[{min_q:.6f}, {max_q:.6f}]")
    print(f"visits_range=[{min_visits}, {max_visits}]")
    print(f"action_counts={dict(sorted(action_counts.items()))}")
    print(f"current_player_counts={dict(sorted(current_player_counts.items()))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="q_values")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inspect(Path(args.path), args.limit)


if __name__ == "__main__":
    main()
