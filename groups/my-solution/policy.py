import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mojo.importer  # noqa: F401

import negamax_mojo  # isort: skip


class OhYes(Policy):
    def mount(self, timeout=None) -> None:
        self.timeout = timeout
        self.search_depth = 6

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "search_depth"):
            self.mount()

        board = np.asarray(s, dtype=np.float32, order="C")

        # Current player's pieces → +1, opponent → -1
        if np.sum(board == -1) == np.sum(board == 1):
            board = -board

        return int(negamax_mojo.act_tt(board.ravel(), self.search_depth))
