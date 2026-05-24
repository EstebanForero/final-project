"""
PUCT MCTS with CNN policy + value (AlphaZero-style).

No fixed depth — explores adaptively. The CNN serves as:
  - Policy prior: softmax(Q-values) guides which branches get more simulations
  - Value:        max(Q-values over legal moves) evaluates leaf positions
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent
ROWS, COLS = 6, 7


# ---------------------------------------------------------------------------
# Numpy CNN
# ---------------------------------------------------------------------------

def _load_weights(path: Path) -> tuple:
    data = np.frombuffer(path.read_bytes(), dtype="<f4")
    off = 0

    def take(shape):
        nonlocal off
        n = int(np.prod(shape))
        w = data[off: off + n].reshape(shape).copy()
        off += n
        return w

    w1  = take((16, 2,  3, 3)); b1  = take((16,))
    w2  = take((32, 16, 3, 3)); b2  = take((32,))
    wf1 = take((64, 32 * 6 * 7)); bf1 = take((64,))
    wf2 = take((7,  64));       bf2 = take((7,))
    return w1, b1, w2, b2, wf1, bf1, wf2, bf2


def _conv2d_relu(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    C_in, H, W = x.shape
    x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    wins  = sliding_window_view(x_pad, (3, 3), axis=(1, 2))
    flat  = wins.transpose(1, 2, 0, 3, 4).reshape(H * W, C_in * 9)
    out   = (flat @ w.reshape(w.shape[0], -1).T + b).T.reshape(w.shape[0], H, W)
    return np.maximum(0.0, out)


def _predict_q(weights: tuple, board_2ch: np.ndarray) -> np.ndarray:
    """Returns Q-values (7,) in (-1, 1) for each column."""
    w1, b1, w2, b2, wf1, bf1, wf2, bf2 = weights
    h1 = _conv2d_relu(board_2ch, w1, b1)
    h2 = _conv2d_relu(h1, w2, b2)
    h3 = np.maximum(0.0, wf1 @ h2.reshape(-1) + bf1)
    return np.tanh(wf2 @ h3 + bf2)


def _encode(board: np.ndarray, player: int) -> np.ndarray:
    out = np.zeros((2, ROWS, COLS), dtype=np.float32)
    out[0] = (board == player).astype(np.float32)
    out[1] = (board == -player).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _valid_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == 0]


def _drop(board: np.ndarray, col: int, player: int) -> np.ndarray:
    nb = board.copy()
    for r in reversed(range(ROWS)):
        if nb[r, col] == 0:
            nb[r, col] = player
            return nb
    return nb


def _terminal(board: np.ndarray) -> int | None:
    """Returns winner (1/-1), 0 for draw, None if ongoing."""
    for p in (1, -1):
        b = board == p
        for r in range(ROWS):
            for c in range(COLS):
                if not b[r, c]:
                    continue
                if c + 3 < COLS and all(b[r, c + i] for i in range(1, 4)):
                    return p
                if r + 3 < ROWS and all(b[r + i, c] for i in range(1, 4)):
                    return p
                if r + 3 < ROWS and c + 3 < COLS and all(b[r+i, c+i] for i in range(1, 4)):
                    return p
                if r + 3 < ROWS and c - 3 >= 0 and all(b[r+i, c-i] for i in range(1, 4)):
                    return p
    if not _valid_cols(board):
        return 0
    return None


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("board", "player", "parent", "col", "prior",
                 "children", "visits", "value_sum", "valid_cols")

    def __init__(self, board, player, parent=None, col=None, prior: float = 1.0):
        self.board      = board
        self.player     = player
        self.parent     = parent
        self.col        = col
        self.prior      = prior
        self.children:  dict[int, _Node] = {}
        self.visits     = 0
        self.value_sum  = 0.0
        self.valid_cols = _valid_cols(board)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def puct(self, c_puct: float) -> float:
        u = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q + u

    def is_leaf(self) -> bool:
        return not self.children


# ---------------------------------------------------------------------------
# MCTS simulation
# ---------------------------------------------------------------------------

def _simulate(root: _Node, weights: tuple, c_puct: float) -> None:
    # Selection
    node = root
    while not node.is_leaf():
        t = _terminal(node.board)
        if t is not None:
            break
        node = max(node.children.values(), key=lambda n: n.puct(c_puct))

    # Expansion + Evaluation
    t = _terminal(node.board)
    if t is not None:
        value = 0.0 if t == 0 else (1.0 if t == node.player else -1.0)
    elif not node.valid_cols:
        value = 0.0
    else:
        q = _predict_q(weights, _encode(node.board, node.player))

        # Value = best Q among legal moves
        value = float(max(q[c] for c in node.valid_cols))

        # Policy prior = softmax of legal Q-values
        legal_q = np.array([q[c] for c in node.valid_cols], dtype=np.float64)
        legal_q -= legal_q.max()
        priors  = np.exp(legal_q)
        priors  /= priors.sum()

        for i, col in enumerate(node.valid_cols):
            child = _Node(
                _drop(node.board, col, node.player),
                -node.player,
                parent=node, col=col,
                prior=float(priors[i]),
            )
            node.children[col] = child

    # Backpropagation
    n, v = node, value
    while n is not None:
        n.visits    += 1
        n.value_sum += v
        v = -v
        n = n.parent


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class CnnMcts(Policy):
    """
    PUCT MCTS: CNN Q-values as policy prior + value. No fixed depth.
    Adaptively explores promising lines much deeper than fixed-depth negamax.
    """

    def __init__(self, simulations: int = 2000, c_puct: float = 1.5) -> None:
        self.simulations = simulations
        self.c_puct      = c_puct

    def mount(self, timeout=None) -> None:
        self.timeout  = timeout
        self._weights = _load_weights(_THIS_DIR / "connect4_qnet.bin")

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "_weights"):
            self.mount()

        board = np.array(s, dtype=np.int8)
        me    = -1 if np.sum(board == -1) == np.sum(board == 1) else 1
        valid = _valid_cols(board)
        if not valid:
            return 0

        # Always take immediate win
        for col in valid:
            if _terminal(_drop(board, col, me)) == me:
                return col

        root = _Node(board, me)
        for _ in range(self.simulations):
            _simulate(root, self._weights, self.c_puct)

        if not root.children:
            return valid[0]
        return max(root.children, key=lambda c: root.children[c].visits)
