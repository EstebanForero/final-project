import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mojo.importer  # noqa: F401

import act_hybrid_mojo  # isort: skip


class HybridCnn(Policy):
    """
    Negamax with two improvements over plain CNN negamax:

    1. Move ordering by heuristic at every internal node (no CNN call needed)
       — better alpha-beta pruning — same depth, fewer nodes visited.

    2. Hybrid leaf evaluation:
       0.6 * CNN_best_Q + 0.4 * tanh(heuristic / 50)
       — CNN covers long-range strategy, heuristic covers immediate threats.
    """

    def mount(self, timeout=None) -> None:
        self.timeout = timeout
        self.search_depth = 4

        model_path = _THIS_DIR / "connect4_qnet.bin"
        self.weights = np.frombuffer(model_path.read_bytes(), dtype="<f4").copy()

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "weights"):
            self.mount()

        board = np.asarray(s, dtype=np.float32, order="C")

        # CNN expects current player's pieces as +1.
        if np.sum(board == -1) == np.sum(board == 1):
            board = -board

        return int(
            act_hybrid_mojo.act(
                board.ravel(),
                board.shape[0],
                board.shape[1],
                self.timeout,
                self.weights,
                self.search_depth,
            )
        )
