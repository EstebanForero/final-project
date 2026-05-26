import sys
import time
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mojo.importer  # noqa: F401

import solution  # isort: skip  — compiles solution.mojo automatically


class Negamax(Policy):
    def mount(self, timeout=None) -> None:
        self.timeout = timeout
        self.search_depth = 14  # change this to control how deep your search goes
        self.last_move_time = None

        self.agent = solution.Agent()

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "search_depth"):
            self.mount()

        board = np.asarray(s, dtype=np.float32, order="C")

        # ConnectState stores Red=-1, Yellow=+1.
        # solution.mojo expects the current player's pieces as +1.
        # Equal piece counts means Red is to move, so flip the board.
        if np.sum(board == -1) == np.sum(board == 1):
            board = -board

        start = time.time()
        result = int(self.agent.act(np.flipud(board).ravel(), self.search_depth))
        elapsed = time.time() - start

        if self.timeout is not None:
            budget = self.timeout * 0.9
            projected_next = elapsed * 7.5
            if projected_next < budget:
                self.search_depth += 2

        self.last_move_time = elapsed
        return result
