"""pytest bootstrap — make the repo importable and pin the working directory.

Auto-loaded by pytest. Guarantees that ``import src...`` works and that the
working directory is the repo root during collection and test runs, regardless
of where ``pytest`` was launched from (PowerShell's CWD can drift to the parent
dir under OneDrive). No test dependencies — purely a path/CWD guard.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if Path.cwd().resolve() != ROOT:
    os.chdir(ROOT)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
