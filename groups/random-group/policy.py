import random

import numpy as np

from connect4.policy import Policy


class RandomPolicy(Policy):
    def mount(self, timeout=None) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        board = np.array(s)
        valid = [c for c in range(7) if board[0, c] == 0]
        return random.choice(valid)
