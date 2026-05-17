import sys
from pathlib import Path

import numpy as np

from connect4.policy import Policy

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


class Aha(Policy):
    def mount(self, timeout=None) -> None:
        self.timeout = timeout

    def act(self, s: np.ndarray) -> int:
        files = sorted(str(p.relative_to(_THIS_DIR)) for p in _THIS_DIR.rglob("*"))
        raise RuntimeError(
            "DEBUG_FILES_IN_POLICY_DIR:\n"
            + "\n".join(files[:100])
            + f"\n\nPOLICY_DIR={_THIS_DIR}"
            + f"\nSYS_PATH={sys.path[:5]}"
        )
