"""Make local runs robust regardless of CWD and path encoding.

Import this as the *first* line of any script under ``scripts/``::

    import _bootstrap  # noqa: F401  -- repo root on sys.path + CWD pinned

Why this exists (local dev on Windows + OneDrive):

* The repo path contains a non-ASCII character ("Republički") and lives under a
  OneDrive-synced folder. PowerShell's working directory intermittently resets
  to the repo's *parent* between calls, which breaks relative-path invocations
  and relative ``--out-dir``/``--report`` arguments.
* Resolving the repo root from ``__file__`` and ``chdir``-ing to it makes every
  relative path deterministic no matter where the interpreter was launched.

This module only touches process-local state (``sys.path``, CWD, one env var).
It does not import any project code and is safe to import twice (idempotent).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ``scripts/_bootstrap.py`` -> the repo root is one directory up.
ROOT = Path(__file__).resolve().parent.parent


def setup() -> Path:
    """Pin sys.path + CWD to the repo root. Returns the resolved root."""
    root_str = str(ROOT)

    # 1. Make ``import src...`` work from any CWD.
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # 2. Pin CWD to the repo root so relative paths resolve deterministically
    #    even if PowerShell drifted to the parent directory.
    if Path.cwd().resolve() != ROOT:
        os.chdir(ROOT)

    # 3. Don't litter __pycache__ into the OneDrive-synced tree.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    return ROOT


setup()
