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


# Default per-flow-type colours for F9–F12. Matches the agent's earlier
# validation PNGs: two_phase = green, build_and_hold = blue,
# late_emergence = purple, unclassified = gray.
_DEFAULT_FLOW_COLORS: dict[str, str] = {
    "two_phase": "#2ca02c",
    "build_and_hold": "#1f77b4",
    "late_emergence": "#9467bd",
    "unclassified": "#bbbbbb",
}


def plot_circuit_size_overlay(
    sizes_by_concept: dict[str, dict[int, int]],
    group_by_concept: dict[str, str],
    *,
    group_colors: dict[str, str] | None = None,
    title: str = "",
    ylabel: str = "Circuit size",
    xlabel: str = "Layer",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 5),
    alpha: float = 0.45,
    linewidth: float = 0.9,
) -> Figure:
    """Per-concept circuit-size curves overlaid, one thin line per
    concept, coloured by group. Legend shows group name and the count
    of concepts in the group.

    Parameters
    ----------
    sizes_by_concept  concept → layer → size (from load_concept_sizes_by_layer).
    group_by_concept  concept → group label (e.g. flow type).
    group_colors      group label → matplotlib colour. Falls back to
                      _DEFAULT_FLOW_COLORS for the four standard flow
                      types, then to matplotlib's default cycle for
                      anything else.
    """
    assert sizes_by_concept, "sizes_by_concept is empty"
    assert group_by_concept, "group_by_concept is empty"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Resolve colours per group, in a deterministic order.
    groups = sorted(set(group_by_concept.values()))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    resolved: dict[str, str] = {}
    cycle_i = 0
    for g in groups:
        if group_colors and g in group_colors:
            resolved[g] = group_colors[g]
        elif g in _DEFAULT_FLOW_COLORS:
            resolved[g] = _DEFAULT_FLOW_COLORS[g]
        else:
            resolved[g] = cycle[cycle_i % len(cycle)]
            cycle_i += 1

    counts: dict[str, int] = {g: 0 for g in groups}
    plotted_label: set[str] = set()
    for concept, sizes in sizes_by_concept.items():
        group = group_by_concept.get(concept)
        if group is None:
            continue
        counts[group] += 1
        layers = sorted(sizes.keys())
        values = [sizes[L] for L in layers]
        label = group if group not in plotted_label else None
        plotted_label.add(group)
        ax.plot(
            layers, values,
            color=resolved[group], alpha=alpha, linewidth=linewidth,
            label=label,
        )

    handles = []
    from matplotlib.lines import Line2D
    for g in groups:
        handles.append(Line2D(
            [0], [0],
            color=resolved[g], linewidth=2.0,
            label=f"{g} ({counts[g]})",
        ))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, framealpha=0.95)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
