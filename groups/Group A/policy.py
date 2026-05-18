import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import mojo.importer  # noqa: F401

import act_mojo  # isort: skip


class Aha(Policy):
    def mount(self, timeout=None) -> None:
        self.timeout = timeout

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "timeout"):
            self.timeout = None

        board = np.asarray(s, dtype=np.float32, order="C")

        report = act_mojo.act(
            board,
            board.shape[0],
            board.shape[1],
            self.timeout,
        )

        # Benchmark mode: intentionally fail so Gradescope prints the report.
        raise RuntimeError(str(report))
