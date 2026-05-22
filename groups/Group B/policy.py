from __future__ import annotations

import random

import numpy as np

from connect4.policy import Policy

ROWS = 6
COLS = 7
# Center-first column ordering improves alpha-beta pruning
_COL_ORDER = [3, 2, 4, 1, 5, 0, 6]

# Score constants
_WIN_SCORE = 100_000
_DEPTH_BONUS = 1  # prefer faster wins / slower losses


class MinimaxPolicy(Policy):
    """
    Difficulty-adjustable Connect 4 opponent using negamax + alpha-beta pruning.

    difficulty=1  random moves
    difficulty=2  depth 2  (beginner)
    difficulty=3  depth 4  (medium)
    difficulty=4  depth 6  (strong)
    difficulty=5  depth 8  (very strong)
    """

    _DEPTH = {1: 0, 2: 2, 3: 4, 4: 6, 5: 8}

    def __init__(self, difficulty: int = 3) -> None:
        self.difficulty = max(1, min(5, difficulty))

    def mount(self, timeout=None) -> None:
        self.timeout = timeout

    def act(self, s: np.ndarray) -> int:
        board = np.array(s, dtype=np.int8)

        # Determine whose turn it is from piece counts
        red_count = int(np.sum(board == -1))
        yellow_count = int(np.sum(board == 1))
        me = -1 if red_count == yellow_count else 1

        valid = _valid_cols(board)
        if not valid:
            return 0

        depth = self._DEPTH[self.difficulty]

        if depth == 0:
            return random.choice(valid)

        best_col = valid[0]
        best_score = -_WIN_SCORE * 2

        for col in _COL_ORDER:
            if col not in valid:
                continue
            child = _drop(board, col, me)
            if _check_win(child, me):
                return col  # instant win
            score = -_negamax(child, -me, depth - 1, -_WIN_SCORE * 2, _WIN_SCORE * 2)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _valid_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == 0]


def _drop(board: np.ndarray, col: int, player: int) -> np.ndarray:
    new = board.copy()
    for r in reversed(range(ROWS)):
        if new[r, col] == 0:
            new[r, col] = player
            return new
    return new


def _check_win(board: np.ndarray, player: int) -> bool:
    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] != player:
                continue
            if c + 3 < COLS and all(board[r, c + i] == player for i in range(1, 4)):
                return True
            if r + 3 < ROWS and all(board[r + i, c] == player for i in range(1, 4)):
                return True
            if r + 3 < ROWS and c + 3 < COLS and all(board[r + i, c + i] == player for i in range(1, 4)):
                return True
            if r + 3 < ROWS and c - 3 >= 0 and all(board[r + i, c - i] == player for i in range(1, 4)):
                return True
    return False


def _window_score(window: list[int], player: int) -> int:
    opp = -player
    p = window.count(player)
    o = window.count(opp)
    if p > 0 and o > 0:
        return 0
    if p == 4:
        return 100
    if p == 3:
        return 5
    if p == 2:
        return 2
    if o == 3:
        return -4
    return 0


def _heuristic(board: np.ndarray, player: int) -> int:
    score = 0
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += _window_score([board[r, c + i] for i in range(4)], player)
    for r in range(ROWS - 3):
        for c in range(COLS):
            score += _window_score([board[r + i, c] for i in range(4)], player)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += _window_score([board[r + i, c + i] for i in range(4)], player)
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            score += _window_score([board[r + i, c - i] for i in range(4)], player)
    return score


def _negamax(board: np.ndarray, player: int, depth: int, alpha: int, beta: int) -> int:
    # The side that just moved is -player; check if they won.
    if _check_win(board, -player):
        return -(_WIN_SCORE + depth * _DEPTH_BONUS)

    valid = _valid_cols(board)
    if not valid:
        return 0  # draw

    if depth == 0:
        return _heuristic(board, player)

    for col in _COL_ORDER:
        if col not in valid:
            continue
        child = _drop(board, col, player)
        score = -_negamax(child, -player, depth - 1, -beta, -alpha)
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return alpha
