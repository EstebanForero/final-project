"""
Side-by-side timing: Python minimax vs Mojo plain negamax vs Mojo + TT.
Run from project root:
    uv run python groups/minimax-mojo/benchmark.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "groups" / "minimax-mojo"))
sys.path.insert(0, str(ROOT / "groups" / "Group B"))

from connect4.connect_state import ConnectState

import mojo.importer  # noqa: F401
import negamax_mojo  # isort: skip
import policy as group_b  # Python minimax


def _make_boards(n: int = 30, seed: int = 42) -> list[np.ndarray]:
    """Generate mid-game boards (8-20 moves played)."""
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        s = ConnectState()
        for _ in range(rng.randint(8, 20)):
            free = s.get_free_cols()
            if not free or s.is_final():
                break
            s = s.transition(rng.choice(free))
        if not s.is_final():
            boards.append(s.board.astype(np.float32))
    return boards


def _to_current_player(board: np.ndarray) -> np.ndarray:
    b = board.copy()
    if np.sum(b == -1) == np.sum(b == 1):
        b = -b
    return b


def _time_python(boards: list[np.ndarray], depth: int) -> float:
    pol = group_b.MinimaxPolicy.__new__(group_b.MinimaxPolicy)
    pol.difficulty = 3  # not used, we override depth via _DEPTH
    # Monkeypatch depth
    pol._DEPTH = {3: depth}
    pol.difficulty = 3

    times = []
    for board in boards:
        t0 = time.perf_counter()
        pol.act(board)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000


def _time_mojo(boards: list[np.ndarray], depth: int, use_tt: bool) -> float:
    fn = negamax_mojo.act_tt if use_tt else negamax_mojo.act_plain
    times = []
    for board in boards:
        b = _to_current_player(board).ravel()
        t0 = time.perf_counter()
        fn(b, depth)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000


def main() -> None:
    boards = _make_boards(30)
    print(f"Boards: {len(boards)} mid-game positions\n")

    header = f"{'Depth':>6}  {'Python (ms)':>12}  {'Mojo plain (ms)':>16}  {'Speedup':>8}  {'Mojo+TT (ms)':>13}  {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for depth in [2, 4, 6, 8]:
        py   = _time_python(boards, depth)
        mojo_plain = _time_mojo(boards, depth, use_tt=False)
        mojo_tt    = _time_mojo(boards, depth, use_tt=True)

        sp_plain = py / mojo_plain if mojo_plain > 0 else float("inf")
        sp_tt    = py / mojo_tt    if mojo_tt    > 0 else float("inf")

        print(
            f"{depth:>6}  {py:>12.2f}  {mojo_plain:>16.3f}  {sp_plain:>7.0f}x"
            f"  {mojo_tt:>13.3f}  {sp_tt:>7.0f}x"
        )


if __name__ == "__main__":
    main()
