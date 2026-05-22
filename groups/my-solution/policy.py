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
        self.search_depth = 4

        # Load weights once as a float32 numpy array.
        # Mojo reads this directly — no per-move struct parsing.
        model_path = _THIS_DIR / "connect4_qnet.bin"
        self.weights = np.frombuffer(model_path.read_bytes(), dtype="<f4").copy()

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "weights"):
            self.mount()

        board = np.asarray(s, dtype=np.float32, order="C")

        # CNN expects current player's pieces as +1, opponent as -1.
        # ConnectState stores Red=-1 and Yellow=+1 regardless of whose turn it is,
        # so flip signs when it's Red's turn (equal piece counts = Red to move).
        if np.sum(board == -1) == np.sum(board == 1):
            board = -board

        return int(
            act_cnn_mojo.act(
                board.ravel(),
                board.shape[0],
                board.shape[1],
                self.timeout,
                self.weights,
                self.search_depth,
            )
        )
