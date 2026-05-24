#!/usr/bin/env python3
"""
Custom stats runner: my-solution vs Group B (MinimaxPolicy).

Run from the project root:
    uv run python groups/runner/run.py
    uv run python groups/runner/run.py --games 40 --difficulty 3
    uv run python groups/runner/run.py --games 50 --difficulty 4 --save-plot results.png
    uv run python groups/runner/run.py --sweep          # all difficulty levels
    uv run python groups/runner/run.py --no-plots       # console only
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy_class(path: Path) -> type[Policy]:
    """Load the first Policy subclass found in a policy.py file."""
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_policy_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for obj in vars(mod).values():
        if isinstance(obj, type) and issubclass(obj, Policy) and obj is not Policy:
            return obj
    raise RuntimeError(f"No Policy subclass found in {path}")


# ---------------------------------------------------------------------------
# Game data
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    result: str          # 'win' | 'loss' | 'draw'  (from my-solution's perspective)
    my_went_first: bool  # True = my-solution played as Red (-1, first mover)
    moves: int
    my_ms_per_move: float   # avg milliseconds per move for my-solution
    opp_ms_per_move: float  # avg milliseconds per move for opponent


# ---------------------------------------------------------------------------
# Game execution
# ---------------------------------------------------------------------------

def play_one_game(
    my_cls: type[Policy],
    opp_cls: type[Policy],
    my_goes_first: bool,
) -> GameRecord:
    my_sign = -1 if my_goes_first else 1
    first_cls, second_cls = (my_cls, opp_cls) if my_goes_first else (opp_cls, my_cls)

    first_p = first_cls()
    second_p = second_cls()
    first_p.mount()
    second_p.mount()

    state = ConnectState()
    moves = 0
    my_total_ms  = 0.0
    opp_total_ms = 0.0
    my_move_count  = 0
    opp_move_count = 0

    while not state.is_final():
        is_first_player = state.player == -1
        current = first_p if is_first_player else second_p

        # Decide whether this move belongs to my-solution or opponent
        is_my_move = (my_goes_first and is_first_player) or (not my_goes_first and not is_first_player)

        t0 = time.perf_counter()
        action = current.act(state.board)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if is_my_move:
            my_total_ms  += elapsed_ms
            my_move_count += 1
        else:
            opp_total_ms  += elapsed_ms
            opp_move_count += 1

        state = state.transition(action)
        moves += 1

    winner = state.get_winner()
    if winner == 0:
        result = "draw"
    else:
        result = "win" if winner == my_sign else "loss"

    return GameRecord(
        result=result,
        my_went_first=my_goes_first,
        moves=moves,
        my_ms_per_move=my_total_ms  / my_move_count  if my_move_count  else 0.0,
        opp_ms_per_move=opp_total_ms / opp_move_count if opp_move_count else 0.0,
    )


def run_series(
    my_cls: type[Policy],
    opp_cls: type[Policy],
    n_games: int,
    verbose: bool,
) -> list[GameRecord]:
    """
    Run n_games, alternating who goes first each game.
    Games 0, 2, 4… → my-solution plays as Red (first).
    Games 1, 3, 5… → my-solution plays as Yellow (second).
    """
    records: list[GameRecord] = []

    for i in range(n_games):
        my_first = (i % 2 == 0)
        try:
            rec = play_one_game(my_cls, opp_cls, my_first)
        except Exception as exc:
            print(f"  [!] game {i+1} errored: {exc}")
            rec = GameRecord("loss", my_first, 0, 0.0, 0.0)

        records.append(rec)

        if verbose:
            wins   = sum(1 for r in records if r.result == "win")
            losses = sum(1 for r in records if r.result == "loss")
            draws  = sum(1 for r in records if r.result == "draw")
            played = len(records)
            wr     = wins / played * 100
            badge  = {"win": "W", "loss": "L", "draw": "D"}[rec.result]
            side   = "Red  " if my_first else "Yellow"
            print(
                f"  game {played:>4}/{n_games}  [my={side}]  {badge}"
                f"  |  {wins}W {losses}L {draws}D  ({wr:.1f}% wr)"
                f"  |  me {rec.my_ms_per_move:>6.1f}ms/move"
                f"  opp {rec.opp_ms_per_move:>6.1f}ms/move"
            )

    return records


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _pct(x: int, total: int) -> str:
    return f"{x / total * 100:.1f}%" if total else "—"


def _avg(vals: list[float]) -> str:
    return f"{sum(vals) / len(vals):.1f}" if vals else "—"


def print_summary(records: list[GameRecord], elapsed: float, opp_label: str) -> None:
    n      = len(records)
    wins   = sum(1 for r in records if r.result == "win")
    losses = sum(1 for r in records if r.result == "loss")
    draws  = sum(1 for r in records if r.result == "draw")

    as_red    = [r for r in records if r.my_went_first]
    as_yellow = [r for r in records if not r.my_went_first]

    my_times  = [r.my_ms_per_move  for r in records if r.my_ms_per_move  > 0]
    opp_times = [r.opp_ms_per_move for r in records if r.opp_ms_per_move > 0]
    lengths   = [r.moves for r in records if r.moves > 0]

    sep = "─" * 58
    print(f"\n{sep}")
    print(f"  my-solution  vs  {opp_label}")
    print(sep)
    print(f"  Games    : {n}   wall time {elapsed:.1f}s  ({elapsed / n * 1000:.0f} ms/game avg)")
    print(sep)
    print(f"  Overall  :  {wins}W  {losses}L  {draws}D")
    print(f"  Win rate :  {_pct(wins, n)}    "
          f"Loss: {_pct(losses, n)}    "
          f"Draw: {_pct(draws, n)}")
    print(sep)

    def side_row(label: str, subset: list[GameRecord]) -> None:
        k = len(subset)
        w = sum(1 for r in subset if r.result == "win")
        l = sum(1 for r in subset if r.result == "loss")
        d = sum(1 for r in subset if r.result == "draw")
        print(f"  {label} ({k:>3} games):  {w}W  {l}L  {d}D   wr {_pct(w, k)}")

    side_row("As Red    (1st mover)", as_red)
    side_row("As Yellow (2nd mover)", as_yellow)
    print(sep)

    print(f"  Move speed — my-solution : avg {_avg(my_times)} ms/move")
    print(f"  Move speed — opponent    : avg {_avg(opp_times)} ms/move")
    if my_times and opp_times:
        ratio = sum(my_times) / len(my_times) / (sum(opp_times) / len(opp_times))
        faster = "my-solution" if ratio < 1 else "opponent"
        print(f"  Speed ratio : {ratio:.2f}x  ({faster} is faster)")
    print(sep)

    if lengths:
        print(f"  Game length: avg {sum(lengths)/len(lengths):.1f}  "
              f"min {min(lengths)}  max {max(lengths)}  moves")
        print(sep)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_single(
    records: list[GameRecord],
    title: str,
    save_path: Path | None,
) -> None:
    n      = len(records)
    wins   = sum(1 for r in records if r.result == "win")
    losses = sum(1 for r in records if r.result == "loss")
    draws  = sum(1 for r in records if r.result == "draw")

    as_red    = [r for r in records if r.my_went_first]
    as_yellow = [r for r in records if not r.my_went_first]

    results   = [r.result for r in records]
    lengths   = [r.moves  for r in records if r.moves > 0]
    my_times  = [r.my_ms_per_move  for r in records if r.my_ms_per_move  > 0]
    opp_times = [r.opp_ms_per_move for r in records if r.opp_ms_per_move > 0]

    window = max(5, n // 10)
    rolling: list[float] = []
    for i in range(n):
        chunk = results[max(0, i - window + 1): i + 1]
        rolling.append(chunk.count("win") / len(chunk) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Plot 1: grouped bar (overall / as Red / as Yellow) ───────────────
    ax = axes[0, 0]
    categories = ["Overall", "As Red\n(1st)", "As Yellow\n(2nd)"]
    groups = [
        [wins, losses, draws],
        [sum(1 for r in as_red    if r.result == "win"),
         sum(1 for r in as_red    if r.result == "loss"),
         sum(1 for r in as_red    if r.result == "draw")],
        [sum(1 for r in as_yellow if r.result == "win"),
         sum(1 for r in as_yellow if r.result == "loss"),
         sum(1 for r in as_yellow if r.result == "draw")],
    ]
    x, bar_w = np.arange(len(categories)), 0.25
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    labels = ["Win", "Loss", "Draw"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + (i - 1) * bar_w, [g[i] for g in groups], bar_w,
               label=label, color=color, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Games")
    ax.set_title("Results by side")
    ax.legend()

    # ── Plot 2: rolling win rate ──────────────────────────────────────────
    ax = axes[0, 1]
    xs = range(1, n + 1)
    ax.plot(xs, rolling, color="#3498db", linewidth=1.8, label="win rate")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="50%")
    ax.fill_between(xs, rolling, 50,
                    where=[v >= 50 for v in rolling], alpha=0.15, color="#2ecc71")
    ax.fill_between(xs, rolling, 50,
                    where=[v <  50 for v in rolling], alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Win rate (%)")
    ax.set_title(f"Rolling win rate  (window = {window} games)")
    ax.set_ylim(0, 100)
    ax.legend()

    # ── Plot 3: game length distribution ─────────────────────────────────
    ax = axes[1, 0]
    if lengths:
        bins = range(min(lengths), max(lengths) + 2)
        ax.hist(lengths, bins=bins, color="#9b59b6", edgecolor="white", alpha=0.85)
        mean_l = sum(lengths) / len(lengths)
        ax.axvline(mean_l, color="orange", linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_l:.1f}")
        ax.legend()
    ax.set_xlabel("Moves")
    ax.set_ylabel("Games")
    ax.set_title("Game length distribution")

    # ── Plot 4: per-move speed comparison ────────────────────────────────
    ax = axes[1, 1]
    if my_times and opp_times:
        ax.scatter(range(len(my_times)),  my_times,  s=18, alpha=0.6,
                   color="#3498db", label="my-solution")
        ax.scatter(range(len(opp_times)), opp_times, s=18, alpha=0.6,
                   color="#e67e22", label="opponent")
        avg_my  = sum(my_times)  / len(my_times)
        avg_opp = sum(opp_times) / len(opp_times)
        ax.axhline(avg_my,  color="#3498db", linestyle="--", linewidth=1.2,
                   label=f"me avg {avg_my:.1f} ms")
        ax.axhline(avg_opp, color="#e67e22", linestyle="--", linewidth=1.2,
                   label=f"opp avg {avg_opp:.1f} ms")
        ax.legend(fontsize=8)
    ax.set_xlabel("Game #")
    ax.set_ylabel("ms / move")
    ax.set_title("Move speed per game")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_sweep(
    sweep_data: list[tuple[int, list[GameRecord]]],
    n_games: int,
    save_path: Path | None,
) -> None:
    difficulties = [d for d, _ in sweep_data]
    win_rates  = [sum(1 for r in recs if r.result == "win")  / len(recs) * 100 for _, recs in sweep_data]
    loss_rates = [sum(1 for r in recs if r.result == "loss") / len(recs) * 100 for _, recs in sweep_data]
    draw_rates = [sum(1 for r in recs if r.result == "draw") / len(recs) * 100 for _, recs in sweep_data]
    my_speeds  = [sum(r.my_ms_per_move  for r in recs if r.my_ms_per_move  > 0) /
                  max(1, sum(1 for r in recs if r.my_ms_per_move  > 0))
                  for _, recs in sweep_data]
    opp_speeds = [sum(r.opp_ms_per_move for r in recs if r.opp_ms_per_move > 0) /
                  max(1, sum(1 for r in recs if r.opp_ms_per_move > 0))
                  for _, recs in sweep_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"my-solution vs MinimaxPolicy — difficulty sweep  ({n_games} games each)",
                 fontsize=13, fontweight="bold")

    # Win/loss/draw by difficulty
    x, bar_w = np.arange(len(difficulties)), 0.25
    ax1.bar(x - bar_w, win_rates,  bar_w, label="Win",  color="#2ecc71", edgecolor="white")
    ax1.bar(x,         loss_rates, bar_w, label="Loss", color="#e74c3c", edgecolor="white")
    ax1.bar(x + bar_w, draw_rates, bar_w, label="Draw", color="#95a5a6", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Diff {d}" for d in difficulties])
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Win / Loss / Draw rate by difficulty")
    ax1.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.legend()
    ax1.set_ylim(0, 110)

    # Speed by difficulty
    ax2.plot(difficulties, my_speeds,  "o-", color="#3498db", linewidth=1.8,
             label="my-solution avg ms/move")
    ax2.plot(difficulties, opp_speeds, "s-", color="#e67e22", linewidth=1.8,
             label="opponent avg ms/move")
    ax2.set_xlabel("Difficulty")
    ax2.set_ylabel("Avg ms / move")
    ax2.set_title("Move speed vs difficulty")
    ax2.legend()
    ax2.set_xticks(difficulties)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sweep plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_opp_cls(base_cls: type, difficulty: int) -> type[Policy]:
    diff = difficulty

    class _Opp(base_cls):
        def __init__(self):
            super().__init__(difficulty=diff)

    _Opp.__name__ = f"{base_cls.__name__}(difficulty={diff})"
    return _Opp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Custom stats runner: my-solution vs Group B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Games per run — should be even so each player starts half as Red (default: 20)",
    )
    parser.add_argument(
        "--difficulty", type=int, default=3, choices=[1, 2, 3, 4, 5],
        help="Group B difficulty 1–5 (default: 3)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run --games games at every difficulty level and compare",
    )
    parser.add_argument("--no-plots",  action="store_true", help="Skip matplotlib output")
    parser.add_argument("--save-plot", type=Path, default=None,
                        help="Save plot to this path instead of displaying it")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the final summary (no per-game lines)")
    parser.add_argument(
        "--solution", type=str, default="my-solution",
        help="Subfolder name inside groups/ to use as the solution (default: my-solution)",
    )
    args = parser.parse_args()

    if args.games % 2 != 0:
        print(f"[warn] --games {args.games} is odd; one side will play one more game as Red.")

    groups = ROOT / "groups"
    print(f"\nLoading {args.solution} ...")
    my_cls = load_policy_class(groups / args.solution / "policy.py")
    print(f"  → {my_cls.__name__}")

    print("Loading Group B ...")
    opp_base = load_policy_class(groups / "Group B" / "policy.py")
    print(f"  → {opp_base.__name__}\n")

    if args.sweep:
        sweep_data: list[tuple[int, list[GameRecord]]] = []
        for d in range(1, 6):
            opp_cls   = make_opp_cls(opp_base, d)
            opp_label = opp_cls.__name__
            print(f"── Difficulty {d} ─────────────────────────────────────────")
            t0 = time.perf_counter()
            records = run_series(my_cls, opp_cls, args.games, verbose=not args.quiet)
            elapsed = time.perf_counter() - t0
            print_summary(records, elapsed, opp_label)
            sweep_data.append((d, records))

            if not args.no_plots:
                single_path: Path | None = None
                if args.save_plot:
                    p = args.save_plot
                    single_path = p.parent / f"{p.stem}_d{d}{p.suffix}"
                plot_single(records, f"my-solution vs {opp_label}  ({args.games} games)", single_path)

        if not args.no_plots:
            sweep_path: Path | None = None
            if args.save_plot:
                p = args.save_plot
                sweep_path = p.parent / f"{p.stem}_sweep{p.suffix}"
            plot_sweep(sweep_data, args.games, sweep_path)

    else:
        opp_cls   = make_opp_cls(opp_base, args.difficulty)
        opp_label = opp_cls.__name__

        print(f"Running {args.games} games  ({args.solution} vs {opp_label})")
        print(f"  Games 1,3,5,…  → {args.solution} plays as Red   (1st mover)")
        print(f"  Games 2,4,6,…  → {args.solution} plays as Yellow (2nd mover)\n")

        t0 = time.perf_counter()
        records = run_series(my_cls, opp_cls, args.games, verbose=not args.quiet)
        elapsed = time.perf_counter() - t0

        print_summary(records, elapsed, opp_label)

        if not args.no_plots:
            save_path = args.save_plot
            if save_path is None:
                save_path = Path(__file__).parent / f"results_{args.solution}_d{args.difficulty}.png"
            plot_single(records, f"{args.solution} vs {opp_label}  ({args.games} games)", save_path)


if __name__ == "__main__":
    main()
