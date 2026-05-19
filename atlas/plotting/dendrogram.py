"""Dendrogram rendering.

Plots a scipy hierarchical-clustering linkage as a coloured dendrogram.
Used for paper Figure 5 (Rust concept clustering, Qwen vs DeepSeek) and
Figure 13 (Python dendrogram, Qwen).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram as _dendrogram


def plot_dendrogram(
    linkage: np.ndarray,
    labels: list[str],
    *,
    title: str = "",
    color_threshold: float | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """Render a hierarchical-clustering dendrogram.

    Parameters
    ----------
    linkage          (n-1, 4) linkage matrix from scipy.cluster.hierarchy.linkage.
    labels           list of leaf labels, length = number of leaves.
    title            axes title (top of the plot). Use "" to omit.
    color_threshold  distance below which branches share a colour. None lets
                     scipy pick the default; the default (0.7 * max distance)
                     gives the four-colour cut for the paper's Rust/Qwen plot.
    figsize          (width, height) inches.

    Returns
    -------
    A matplotlib Figure. Caller is responsible for saving / closing.

    Pure aside from creating a Figure object.
    """
    assert linkage.ndim == 2 and linkage.shape[1] == 4, (
        f"linkage must be (n-1, 4), got shape {linkage.shape}"
    )
    assert linkage.shape[0] == len(labels) - 1, (
        f"linkage has {linkage.shape[0]} rows but labels has {len(labels)} entries "
        f"(expected {linkage.shape[0] + 1})"
    )

    fig, ax = plt.subplots(figsize=figsize)
    _dendrogram(
        linkage,
        labels=labels,
        color_threshold=color_threshold,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(False)
    fig.tight_layout()
    return fig
