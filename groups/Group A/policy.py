import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import act_mojo
import mojo.importer  # noqa: F401


class Aha(Policy):
    def mount(self, timeout=None) -> None:
        self.timeout = timeout

    def act(self, s: np.ndarray) -> int:
        board = np.asarray(s, dtype=np.float32, order="C")
        return int(act_mojo.act(board, board.shape[0], board.shape[1]))
