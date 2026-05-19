from __future__ import annotations

import argparse
import mmap
import struct
from collections import Counter
from pathlib import Path

from .dataset import RECORD_SIZE, WIDTH, decode_state_key


def print_numeric_stats(name: str, values: list[float] | list[int]) -> None:
    if not values:
        print(f"{name}=empty")
        return

    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mean = sum(ordered) / count

    def percentile(p: float) -> float:
        if count == 1:
            return ordered[0]

        position = p * (count - 1)
        low = int(position)
        high = min(low + 1, count - 1)
        fraction = position - low
        return ordered[low] * (1.0 - fraction) + ordered[high] * fraction

    print(f"{name}_count={count}")
    print(f"{name}_min={ordered[0]:.6f}")
    print(f"{name}_mean={mean:.6f}")
    print(f"{name}_max={ordered[-1]:.6f}")
    print(f"{name}_p01={percentile(0.01):.6f}")
    print(f"{name}_p05={percentile(0.05):.6f}")
    print(f"{name}_p25={percentile(0.25):.6f}")
    print(f"{name}_p50={percentile(0.50):.6f}")
    print(f"{name}_p75={percentile(0.75):.6f}")
    print(f"{name}_p95={percentile(0.95):.6f}")
    print(f"{name}_p99={percentile(0.99):.6f}")


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
    all_q_values: list[float] = []
    visited_q_values: list[float] = []
    visited_visits: list[int] = []

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
                all_q_values.append(q_value)
                if visits > 0:
                    visited_q_values.append(q_value)
                    visited_visits.append(visits)

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
    print()
    print_numeric_stats("q_all_records", all_q_values)
    print()
    print_numeric_stats("q_visited_records", visited_q_values)
    print()
    print_numeric_stats("visits_visited_records", visited_visits)


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
