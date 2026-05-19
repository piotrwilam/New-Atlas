"""Central matplotlib style.

Call `apply_style(name)` once at the top of every figure script.
Plotting functions never set style themselves — that keeps figures
consistent and lets the paper, poster, and slides versions diverge in
exactly one place.
"""

from __future__ import annotations

import matplotlib as mpl

_STYLES: dict[str, dict] = {
    "paper": {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.4,
    },
    "poster": {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "lines.linewidth": 2.0,
    },
    "slides": {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "lines.linewidth": 1.8,
    },
}


def apply_style(name: str = "paper") -> None:
    """Apply one of the predefined matplotlib styles globally.

    Parameters
    ----------
    name  one of {"paper", "poster", "slides"}.

    Side effects: mutates matplotlib rcParams.
    """
    assert name in _STYLES, f"unknown style {name!r}; choose from {sorted(_STYLES)}"
    mpl.rcParams.update(_STYLES[name])
