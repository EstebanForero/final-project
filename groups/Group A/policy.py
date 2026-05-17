import sys
from pathlib import Path

import mojo.importer
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fast_sum

from connect4.policy import Policy


class Aha(Policy):
    def mount(self) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        board = np.asarray(s, dtype=np.float32, order="C")
        return int(fast_sum.act(board, board.shape[0], board.shape[1]))
