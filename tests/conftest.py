"""Pytest configuration: add `src/` to sys.path so tests can import `medembed`.

This avoids requiring a `pip install -e .` just to run the test suite in CI.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
