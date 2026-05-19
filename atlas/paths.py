"""Repository-anchored paths.

Structural paths only (package root, results, paper). Per-experiment
paths come from the Hydra config, not from here.

`DATA_ROOT` is the source of truth for experimental artifacts. It reads
the `ATLAS_DATA_ROOT` env var if set, otherwise defaults to the user's
local mirror at `~/Data/New-Atlas`. On Colab, set the env var to the
mounted Drive path; on CI, to the downloaded HF dataset path.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root: this file lives at <repo>/atlas/paths.py.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Default location of the artifact mirror. Override via $ATLAS_DATA_ROOT.
_DEFAULT_DATA_ROOT = Path.home() / "Data" / "New-Atlas"
DATA_ROOT: Path = Path(os.environ.get("ATLAS_DATA_ROOT", _DEFAULT_DATA_ROOT))

# Where experiment runs write their outputs (per-run subdir, gitignored).
RESULTS_DIR: Path = REPO_ROOT / "results"

# Where the LaTeX paper + final figures live (tracked in git).
PAPER_DIR: Path = REPO_ROOT / "paper"
PAPER_FIGURES_DIR: Path = PAPER_DIR / "figures"
