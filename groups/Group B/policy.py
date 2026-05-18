import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mojo.importer  # noqa: F401

import act_mojo  # isort: skip


class Hola(Policy):
    def mount(self, timeout=None) -> None:
        self.timeout = timeout
        self.q_values_path = str(_THIS_DIR / "q_values")

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "q_values_path"):
            self.mount()

        board = np.asarray(s, dtype=np.float32, order="C")
        flat_board = board.ravel()

        return int(
            act_mojo.act(
                flat_board,
                board.shape[0],
                board.shape[1],
                self.timeout,
                self.q_values_path,
            )
        )
