"""Path helpers for running the parallel RL examples from the repo checkout."""

from __future__ import annotations

import sys
from pathlib import Path


PARALLEL_RL_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARALLEL_RL_DIR.parents[1]
REPO_SRC = REPO_ROOT / "src"


def ensure_repo_src_on_path() -> None:
    """Make the in-tree ``soromox`` package importable without installation."""

    repo_src = str(REPO_SRC)
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
