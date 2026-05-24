"""
Play Connect 4 against the Mojo negamax engine.

Run from the project root:
    uv run python groups/reference/play.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from connect4.connect_state import ConnectState

import mojo.importer  # noqa: F401
import negamax_mojo   # isort: skip

# ── ANSI ──────────────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

DISC   = {-1: f"{RED}●{RESET}", 1: f"{YELLOW}●{RESET}", 0: " "}
NAME   = {-1: f"{RED}Red{RESET}", 1: f"{YELLOW}Yellow{RESET}"}
SYMBOL = {-1: f"{RED}●{RESET}", 1: f"{YELLOW}●{RESET}"}


def clear() -> None:
    print("\033[2J\033[H", end="")


def draw(board: np.ndarray, last_col: int | None, status: str, engine_info: str = "") -> None:
    clear()
    print(f"\n  {BOLD}Connect 4{RESET}  —  Mojo negamax + TT\n")

    # Arrow row pointing to last played column
    arrow_row = "    "
    for c in range(7):
        arrow_row += (f"{CYAN}↓{RESET}  " if c == last_col else "   ")
    print(arrow_row)

    # Top border
    print("  ┌───┬───┬───┬───┬───┬───┬───┐")

    # Rows
    for r, row in enumerate(board):
        line = "  │"
        for cell in row:
            line += f" {DISC[cell]} │"
        print(line)
        if r < 5:
            print("  ├───┼───┼───┼───┼───┼───┼───┤")

    # Bottom border + column numbers
    print("  └───┴───┴───┴───┴───┴───┴───┘")
    print(f"    {'   '.join(str(c) for c in range(7))}")

    # Status line
    print(f"\n  {status}")
    if engine_info:
        print(f"  {engine_info}")
    print()


def board_to_float(board: np.ndarray, current: int) -> np.ndarray:
    b = board.astype(np.float32)
    return -b if current == -1 else b


def engine_move(state: ConnectState, depth: int) -> tuple[int, float]:
    b = board_to_float(state.board, state.player)
    t0 = time.perf_counter()
    col = int(negamax_mojo.act_tt(b.ravel(), depth))
    ms = (time.perf_counter() - t0) * 1000
    return col, ms


def human_move(state: ConnectState, free: list[int]) -> int:
    while True:
        raw = input(f"  Drop in column [{' '.join(str(c) for c in free)}]: ").strip()
        try:
            col = int(raw)
            if col in free:
                return col
            print(f"  {GRAY}Column {col} is full or invalid.{RESET}")
        except ValueError:
            print(f"  {GRAY}Enter a column number.{RESET}")


def pick_side() -> int:
    print(f"\n  {RED}Red ●{RESET} always moves first.\n")
    while True:
        c = input(f"  Play as  [{RED}R{RESET}]ed  or  [{YELLOW}Y{RESET}]ellow ?  ").strip().lower()
        if c in ("r", "red"):
            return -1
        if c in ("y", "yellow"):
            return 1
        print(f"  {GRAY}Type r or y.{RESET}")


def pick_depth() -> int:
    print(f"\n  Engine strength (search depth):\n")
    # Timings measured on this machine (mid-game positions, TT enabled)
    levels = [(4,  "easy",       "~0.1 ms / move"  ),
              (6,  "medium",     "~0.5 ms / move"  ),
              (8,  "hard",       "~2 ms / move"    ),
              (10, "impossible", "~10 ms / move"   ),
              (12, "god",        "~50 ms / move"   ),
              (14, "why",        "~2 s / move"     ),  # measured: ~2 s
              (16, "send help",  "~37 s / move"    )]  # measured: ~37 s
    for d, label, speed in levels:
        print(f"    {BOLD}{d}{RESET}  {label:<8} {GRAY}{speed}{RESET}")
    print()
    while True:
        raw = input("  Choose depth [4 / 6 / 8 / 10 / 12 / 14 / 16, default 6]: ").strip() or "6"
        try:
            d = int(raw)
            if 1 <= d <= 16:
                return d
            print(f"  {GRAY}Pick 1–16.{RESET}")
        except ValueError:
            print(f"  {GRAY}Enter a number.{RESET}")


def play_game(human: int, depth: int) -> None:
    state       = ConnectState()
    last_col    = None
    engine_info = ""

    while not state.is_final():
        free    = state.get_free_cols()
        current = state.player

        if current == human:
            status = f"Your turn  {SYMBOL[human]}  — pick a column"
            draw(state.board, last_col, status, engine_info)
            col = human_move(state, free)
            last_col = col
        else:
            draw(state.board, last_col, f"{GRAY}Engine thinking…{RESET}", engine_info)
            col, ms = engine_move(state, depth)
            last_col    = col
            engine_info = f"{GRAY}Engine took {BOLD}{ms:.1f} ms{RESET}{GRAY} on last move (col {col}, depth {depth}){RESET}"
            draw(state.board, last_col, f"Engine played column {BOLD}{col}{RESET}", engine_info)
            input(f"  {GRAY}Press Enter to continue…{RESET}")

        state = state.transition(col)

    winner = state.get_winner()
    if winner == 0:
        outcome = f"{BOLD}Draw!{RESET}"
    elif winner == human:
        outcome = f"{BOLD}You win! 🎉{RESET}"
    else:
        outcome = f"{BOLD}Engine wins.{RESET}"

    draw(state.board, last_col, outcome, engine_info)


def main() -> None:
    clear()
    print(f"\n  {BOLD}Connect 4{RESET}  —  Mojo negamax + TT\n")
    human = pick_side()
    depth = pick_depth()

    while True:
        play_game(human, depth)
        again = input("\n  Play again? [y/n]: ").strip().lower()
        if again not in ("y", "yes"):
            break
        human = pick_side()
        depth = pick_depth()

    print("\n  Goodbye!\n")


if __name__ == "__main__":
    main()
