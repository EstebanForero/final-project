"""Generate all report figures for the Connect-4 agent analysis."""
from __future__ import annotations

import importlib
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import math

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from connect4.connect_state import ConnectState  # noqa: E402

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "v0":      {"color": "#6b7280", "label": "v0 (no heuristic)"},
    "v1":      {"color": "#2563eb", "label": "v1 (heuristic)"},
    "current": {"color": "#059669", "label": "current (TT + adapt.)"},
}

# ─── Policy specs ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicySpec:
    name: str
    module: str
    cls: str
    source_dir: str | None = None


POLICIES: dict[str, PolicySpec] = {
    "random":  PolicySpec("random",  "groups.random-group.policy",   "RandomPolicy"),
    "v0":      PolicySpec("v0",      "groups.my-solution-v0.policy", "OhYes"),
    "v1":      PolicySpec("v1",      "groups.my-solution-v1.policy", "NegamaxHeuristics"),
    "current": PolicySpec("current", "groups.my-solution.policy",    "NegamaxAdaptativeDeepening"),
}


def load_policy_class(spec: PolicySpec):
    module = importlib.import_module(spec.module)
    return getattr(module, spec.cls)


def make_policy(spec: PolicySpec, depth: int | None = None):
    cls = load_policy_class(spec)
    policy = cls()
    if hasattr(policy, "mount"):
        policy.mount()
    if depth is not None and hasattr(policy, "search_depth"):
        policy.search_depth = depth
    return policy


def play_game(
    first_spec: PolicySpec,
    second_spec: PolicySpec,
    depth: int | None = None,
    max_moves: int = 42,
) -> dict:
    first  = make_policy(first_spec,  depth)
    second = make_policy(second_spec, depth)
    state  = ConnectState()
    n_moves = 0

    while not state.is_final() and n_moves < max_moves:
        policy = first if state.player == -1 else second
        action = int(policy.act(state.board))
        state  = state.transition(action)
        n_moves += 1

    winner_color = state.get_winner()
    if winner_color == -1:
        winner = first_spec.name
    elif winner_color == 1:
        winner = second_spec.name
    else:
        winner = "draw"

    return {"first": first_spec.name, "second": second_spec.name,
            "winner": winner, "moves": n_moves}


def match(a: PolicySpec, b: PolicySpec, games: int = 20, depth: int | None = None) -> pd.DataFrame:
    rows = []
    for i in range(games):
        row = play_game(a, b, depth=depth) if i % 2 == 0 else play_game(b, a, depth=depth)
        row["game"] = i + 1
        rows.append(row)
    return pd.DataFrame(rows)


def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.5, 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


# ─── Experiment 1: win rate vs depth (current vs random) ──────────────────────

def run_depth_sweep(depths=(2, 4, 6, 8), n_games=20) -> pd.DataFrame:
    rows = []
    spec = POLICIES["current"]
    rnd  = POLICIES["random"]
    for d in depths:
        print(f"  depth={d}...", flush=True)
        df = match(spec, rnd, games=n_games, depth=d)
        wins = (df["winner"] == "current").sum()
        wr, lo, hi = wilson_ci(int(wins), n_games)
        # timing: measure first-player decision time
        times_ms = []
        for _ in range(6):
            state = ConnectState()
            policy = make_policy(spec, d)
            while not state.is_final():
                t0 = time.perf_counter()
                action = int(policy.act(state.board))
                times_ms.append((time.perf_counter() - t0) * 1000)
                state = state.transition(action)
                break  # only first move per game
        med_ms = float(np.median(times_ms)) if times_ms else 0.0
        rows.append({"depth": d, "win_rate": wr, "ci_lo": lo, "ci_hi": hi,
                     "wins": int(wins), "n": n_games, "median_ms": med_ms})
    return pd.DataFrame(rows)


# ─── Experiment 2: latency sweep v1 vs current ────────────────────────────────

def run_latency_sweep(depths=(2, 4, 6, 8), n_positions=20) -> pd.DataFrame:
    rows = []
    state0 = ConnectState()
    for agent_name in ("v1", "current"):
        spec = POLICIES[agent_name]
        for d in depths:
            print(f"  latency {agent_name} depth={d}...", flush=True)
            policy = make_policy(spec, d)
            samples = []
            state = state0
            for _ in range(n_positions):
                if state.is_final():
                    state = ConnectState()
                t0 = time.perf_counter()
                action = int(policy.act(state.board))
                samples.append((time.perf_counter() - t0) * 1000)
                state = state.transition(action)
            arr = np.array(samples)
            rows.append({
                "agent": agent_name, "depth": d,
                "median_ms": float(np.median(arr)),
                "p25_ms": float(np.quantile(arr, 0.25)),
                "p75_ms": float(np.quantile(arr, 0.75)),
            })
    return pd.DataFrame(rows)


# ─── Experiment 3: version comparison at fixed depth ──────────────────────────

def run_version_comparison(depth=8, n_games=30) -> pd.DataFrame:
    rows = []
    rnd = POLICIES["random"]
    for name in ("v0", "v1", "current"):
        spec = POLICIES[name]
        print(f"  {name} vs random...", flush=True)
        df = match(spec, rnd, games=n_games, depth=depth)
        wins = (df["winner"] == name).sum()
        wr, lo, hi = wilson_ci(int(wins), n_games)
        rows.append({"version": name, "win_rate": wr, "ci_lo": lo, "ci_hi": hi,
                     "wins": int(wins), "n": n_games})
    return pd.DataFrame(rows)


# ─── Experiment 4: color analysis (first vs second player) ────────────────────

def run_color_analysis(depth=8, n_per_color=15) -> pd.DataFrame:
    rows = []
    rnd = POLICIES["random"]
    for name in ("v0", "v1", "current"):
        spec = POLICIES[name]
        print(f"  color {name}...", flush=True)
        # As first player (Red, -1)
        p1_games = [play_game(spec, rnd, depth=depth) for _ in range(n_per_color)]
        p1_wins  = sum(g["winner"] == name for g in p1_games)
        p1_wr, p1_lo, p1_hi = wilson_ci(p1_wins, n_per_color)
        # As second player (Yellow, +1)
        p2_games = [play_game(rnd, spec, depth=depth) for _ in range(n_per_color)]
        p2_wins  = sum(g["winner"] == name for g in p2_games)
        p2_wr, p2_lo, p2_hi = wilson_ci(p2_wins, n_per_color)

        rows.append({"version": name, "role": "First (P1)", "win_rate": p1_wr,
                     "ci_lo": p1_lo, "ci_hi": p1_hi, "wins": p1_wins, "n": n_per_color})
        rows.append({"version": name, "role": "Second (P2)", "win_rate": p2_wr,
                     "ci_lo": p2_lo, "ci_hi": p2_hi, "wins": p2_wins, "n": n_per_color})
    return pd.DataFrame(rows)


# ─── Experiment 5: self-play ──────────────────────────────────────────────────

def run_self_play(depth=8, n_games=20) -> pd.DataFrame:
    spec_a = POLICIES["current"]
    # Create a renamed spec so first/second are distinguishable
    spec_b = PolicySpec("current_B", "groups.my-solution.policy",
                        "NegamaxAdaptativeDeepening")
    rows = []
    for i in range(n_games):
        if i % 2 == 0:
            row = play_game(spec_a, spec_b, depth=depth)
            row["nominal_first"] = "current"
        else:
            row = play_game(spec_b, spec_a, depth=depth)
            row["nominal_first"] = "current_B"
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_depth_winrate(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(df["depth"], df["win_rate"], marker="o", color=STYLE["current"]["color"],
            linewidth=2, markersize=6, label="current vs random")
    ax.fill_between(df["depth"], df["ci_lo"], df["ci_hi"],
                    alpha=0.2, color=STYLE["current"]["color"], label="95% Wilson CI")
    ax.axhline(0.5, linestyle="--", linewidth=0.8, color="gray", label="random baseline")
    ax.set_xlabel("Search depth")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(df["depth"])
    ax.set_title("Win rate vs search depth\n(current agent vs random)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_depth_winrate.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_depth_winrate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_depth_winrate")


def plot_latency(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    colors = {"v1": STYLE["v1"]["color"], "current": STYLE["current"]["color"]}
    labels = {"v1": "v1 (no TT)", "current": "current (TT)"}
    for agent, sub in df.groupby("agent"):
        sub = sub.sort_values("depth")
        ax.plot(sub["depth"], sub["median_ms"], marker="s", color=colors[agent],
                linewidth=2, markersize=6, label=labels[agent])
        ax.fill_between(sub["depth"], sub["p25_ms"], sub["p75_ms"],
                        alpha=0.15, color=colors[agent])
    ax.set_xlabel("Search depth")
    ax.set_ylabel("Median decision time (ms)")
    ax.set_xticks(sorted(df["depth"].unique()))
    ax.set_title("Decision time vs depth\n(v1 no-TT vs current with TT)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_latency_depth.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_latency_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_latency_depth")


def plot_version_comparison(df: pd.DataFrame, depth: int = 8):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    versions = df["version"].tolist()
    x = np.arange(len(versions))
    colors = [STYLE[v]["color"] for v in versions]
    ax.bar(x, df["win_rate"], color=colors, width=0.5, alpha=0.85)
    for xi, (_, row) in zip(x, df.iterrows()):
        ax.errorbar(xi, row["win_rate"],
                    yerr=[[row["win_rate"] - row["ci_lo"]], [row["ci_hi"] - row["win_rate"]]],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
        ax.text(xi, row["ci_hi"] + 0.03, f"{row['wins']}/{row['n']}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([STYLE[v]["label"] for v in versions], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Win rate vs random")
    ax.set_title(f"Version comparison (depth={depth})\n(95% Wilson CI, {df['n'].iloc[0]} games each)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_version_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_version_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_version_comparison")


def plot_color_analysis(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    versions = df["version"].unique().tolist()
    roles    = ["First (P1)", "Second (P2)"]
    n_v      = len(versions)
    x        = np.arange(n_v)
    width    = 0.35
    role_colors = {"First (P1)": "#1d4ed8", "Second (P2)": "#b45309"}
    hatches     = {"First (P1)": "", "Second (P2)": "//"}

    for j, role in enumerate(roles):
        sub = df[df["role"] == role].set_index("version").reindex(versions)
        offset = (j - 0.5) * width
        rects = ax.bar(x + offset, sub["win_rate"], width, label=role,
                       color=role_colors[role], alpha=0.8, hatch=hatches[role])
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.errorbar(x[i] + offset, row["win_rate"],
                        yerr=[[row["win_rate"] - row["ci_lo"]], [row["ci_hi"] - row["win_rate"]]],
                        fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([STYLE[v]["label"] for v in versions], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Win rate vs random")
    ax.set_title("Win rate by player color\n(P1=Red / P2=Yellow, 95% Wilson CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_color_analysis.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_color_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_color_analysis")


def plot_self_play(df: pd.DataFrame):
    first_wins  = (df["winner"] == df["first"]).sum()
    second_wins = (df["winner"] == df["second"]).sum()
    draws       = (df["winner"] == "draw").sum()
    n           = len(df)

    fp_wr, fp_lo, fp_hi = wilson_ci(int(first_wins), n)
    sp_wr, sp_lo, sp_hi = wilson_ci(int(second_wins), n)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    roles = ["First player", "Second player", "Draw"]
    vals  = [first_wins, second_wins, draws]
    colors_sp = ["#1d4ed8", "#b45309", "#9ca3af"]
    bars = ax.bar(roles, [v / n for v in vals], color=colors_sp, alpha=0.85, width=0.5)
    ax.errorbar(0, fp_wr, yerr=[[fp_wr - fp_lo], [fp_hi - fp_wr]],
                fmt="none", color="black", capsize=5)
    ax.errorbar(1, sp_wr, yerr=[[sp_wr - sp_lo], [sp_hi - sp_wr]],
                fmt="none", color="black", capsize=5)
    for i, v in enumerate(vals):
        ax.text(i, v / n + 0.04, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Fraction of games")
    ax.set_title(f"Self-play: current vs current\n(depth=8, n={n} games)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_self_play.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_self_play.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_self_play")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.dpi": 150,
    })

    print("=== Depth sweep (current vs random) ===")
    depth_df = run_depth_sweep(depths=[2, 4, 6, 8], n_games=20)
    print(depth_df[["depth", "wins", "n", "win_rate", "median_ms"]].to_string(index=False))
    plot_depth_winrate(depth_df)

    print("\n=== Latency sweep (v1 vs current) ===")
    latency_df = run_latency_sweep(depths=[2, 4, 6, 8], n_positions=20)
    print(latency_df.to_string(index=False))
    plot_latency(latency_df)

    print("\n=== Version comparison at depth=8 ===")
    version_df = run_version_comparison(depth=8, n_games=30)
    print(version_df.to_string(index=False))
    plot_version_comparison(version_df, depth=8)

    print("\n=== Color analysis ===")
    color_df = run_color_analysis(depth=8, n_per_color=15)
    print(color_df.to_string(index=False))
    plot_color_analysis(color_df)

    print("\n=== Self-play (current vs current) ===")
    sp_df = run_self_play(depth=8, n_games=20)
    fw = (sp_df["winner"] == sp_df["first"]).sum()
    sw = (sp_df["winner"] == sp_df["second"]).sum()
    dr = (sp_df["winner"] == "draw").sum()
    print(f"  first player wins: {fw}, second: {sw}, draws: {dr}")
    plot_self_play(sp_df)

    print("\nAll figures saved to", FIGURE_DIR)
