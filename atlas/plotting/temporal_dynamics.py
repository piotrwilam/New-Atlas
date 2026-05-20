"""Per-layer circuit-size line plot, faceted by concept.

Used for paper Figure 4 ("The 'How' axis"): same atomicity concepts
plotted as circuit size vs layer, revealing each model's distinct
temporal signature (Qwen `two_phase` — spike-collapse-re-explode;
DeepSeek `build_and_hold` — monotonic gradual growth).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_temporal_dynamics(
    sizes_by_concept: dict[str, dict[int, int]],
    *,
    concepts: list[str],
    title: str = "",
    highlight_layers: list[tuple[int, int]] | None = None,
    ylabel: str = "Circuit size (in neurons)",
    xlabel: str = "Layer",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> Figure:
    """Render per-layer circuit sizes for a selected concept subset.

    Parameters
    ----------
    sizes_by_concept  dict of concept name → dict of layer → size, as
                      produced by `load_concept_sizes_by_layer`.
    concepts          ordered list of concept names to plot (one line each).
                      Order controls legend order and colour assignment.
    title             axes title; "" to omit.
    highlight_layers  list of (start, end) inclusive layer ranges to shade
                      as vertical green bands. Use to highlight the
                      two_phase early- and late-burst windows on Qwen.
    ylabel, xlabel    axis labels.
    ax                pre-existing axes to plot into. If None, a new
                      Figure + Axes is created at `figsize`.
    figsize           used only when `ax` is None.

    Returns
    -------
    The matplotlib Figure (either freshly created or the one owning `ax`).
    """
    assert concepts, "need at least one concept to plot"
    missing = [c for c in concepts if c not in sizes_by_concept]
    assert not missing, f"concepts missing from data: {missing}"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if highlight_layers:
        for start, end in highlight_layers:
            assert start <= end, f"highlight range {start}..{end} is backwards"
            ax.axvspan(start, end, color="green", alpha=0.12, zorder=0)

    for concept in concepts:
        sizes = sizes_by_concept[concept]
        layers = sorted(sizes.keys())
        values = [sizes[L] for L in layers]
        ax.plot(layers, values, label=concept, linewidth=1.4)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95)
    return fig
