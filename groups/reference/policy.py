import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mojo.importer  # noqa: F401

import negamax_mojo  # isort: skip


def _current_player_board(s: np.ndarray) -> np.ndarray:
    board = np.asarray(s, dtype=np.float32, order="C")
    # Red=-1 Yellow=+1 in ConnectState; equal counts → Red to move → flip so current=+1
    if np.sum(board == -1) == np.sum(board == 1):
        board = -board
    return board


# class NegamaxMojo(Policy):
class NegamaxMojo:
    # class NegamaxMojo:
    """Pure negamax + alpha-beta in Mojo. Same heuristic as Python Group B."""

    def mount(self, timeout=None) -> None:
        self.timeout = timeout
        self.search_depth = 6

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "search_depth"):
            self.mount()
        board = _current_player_board(s)
        return int(negamax_mojo.act_plain(board.ravel(), self.search_depth))


class NegamaxTTMojo(Policy):
    """Negamax + alpha-beta + transposition table in Mojo."""

    def mount(self, timeout=None) -> None:
        self.timeout = timeout
        self.search_depth = 8

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "search_depth"):
            self.mount()
        board = _current_player_board(s)
        return int(negamax_mojo.act_tt(board.ravel(), self.search_depth))


# Default export for the runner
MinimaxMojoPolicy = NegamaxMojo
