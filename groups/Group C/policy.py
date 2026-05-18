import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mojo.importer  # noqa: F401

import act_cnn_mojo  # isort: skip


class OhYes(Policy):
    def mount(self, timeout=None) -> None:
        self.timeout = timeout
        self.model_path = str(_THIS_DIR / "connect4_qnet.bin")

        # Start with 0 to test pure CNN-greedy.
        # Then try 1, 2, 3.
        self.search_depth = 2

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "model_path"):
            self.mount()

        board = np.asarray(s, dtype=np.float32, order="C")
        flat_board = board.ravel()

        return int(
            act_cnn_mojo.act(
                flat_board,
                board.shape[0],
                board.shape[1],
                self.timeout,
                self.model_path,
                self.search_depth,
            )
        )
