"""Bar chart of within-group Jaccard cohesion against a permutation null.

Used for paper Figure 6: per-group observed mean Jaccard plotted against
the null distribution from `permutation_within_group_p_value`. A null
band runs from the null mean to the 95th percentile so significance
reads off visually — bars that clear the band's upper edge are unlikely
to be drawn from the random-group distribution.
"""

from __future__ import annotations

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch


class GroupResult(TypedDict):
    name: str
    observed: float
    null_mean: float
    null_p95: float
    p_value: float


_SIG_COLOR = "#d62728"
_NS_COLOR = "#7f7f7f"
_NULL_BAND_COLOR = "lightgray"


def _significance_label(p: float) -> str:
    if p < 0.001:
        return "***\np < 0.001"
    if p < 0.01:
        return f"**\np = {p:.3f}"
    if p < 0.05:
        return f"*\np = {p:.3f}"
    return f"n.s.\np = {p:.3f}"


def plot_group_coherence(
    results: list[GroupResult],
    *,
    title: str = "",
    figsize: tuple[float, float] = (9, 5),
    alpha: float = 0.05,
) -> Figure:
    """Bar chart of observed Jaccard cohesion per group with null overlay.

    Parameters
    ----------
    results  list of per-group records (observed mean Jaccard, null mean,
             null 95th percentile, one-sided p-value).
    title    axes title; "" to omit.
    figsize  (width, height) inches.
    alpha    significance threshold for the bar colour. Bars below alpha
             are red; bars at or above are gray.

    Returns
    -------
    matplotlib Figure. Caller saves and closes.
    """
    assert len(results) >= 1, "need at least one group to plot"
    assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"

    n = len(results)
    x = np.arange(n)
    observed = np.array([r["observed"] for r in results])
    null_mean = np.array([r["null_mean"] for r in results])
    null_p95 = np.array([r["null_p95"] for r in results])
    p_values = [r["p_value"] for r in results]
    names = [r["name"] for r in results]

    fig, ax = plt.subplots(figsize=figsize)

    # Null band: rectangle from null_mean to null_p95 for each group.
    for i in range(n):
        ax.fill_between(
            [x[i] - 0.42, x[i] + 0.42],
            [null_mean[i], null_mean[i]],
            [null_p95[i], null_p95[i]],
            color=_NULL_BAND_COLOR,
            alpha=0.7,
            zorder=1,
        )

    # Observed bars, coloured by significance.
    colors = [_SIG_COLOR if p < alpha else _NS_COLOR for p in p_values]
    ax.bar(x, observed, color=colors, alpha=0.85, zorder=2, width=0.84)

    # Annotate each bar with significance + p-value.
    headroom = 0.04 * max(observed.max(), null_p95.max())
    for i, p in enumerate(p_values):
        ax.text(
            x[i],
            max(observed[i], null_p95[i]) + headroom,
            _significance_label(p),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Mean pairwise Jaccard")
    ax.set_ylim(0, max(observed.max(), null_p95.max()) * 1.25)
    if title:
        ax.set_title(title)

    legend_elements = [
        Patch(color=_NULL_BAND_COLOR, alpha=0.7,
              label="null mean → 95th percentile"),
        Patch(color=_SIG_COLOR, alpha=0.85,
              label=f"observed (p < {alpha})"),
        Patch(color=_NS_COLOR, alpha=0.85,
              label=f"observed (p ≥ {alpha})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              frameon=True, framealpha=0.95)

    fig.tight_layout()
    return fig
