"""Plotting: style + per-figure-type rendering functions.

Plotting functions take data and return a `matplotlib.figure.Figure`.
They never read or write files; that's the experiment script's job.
"""

from atlas.plotting.style import apply_style
from atlas.plotting.dendrogram import plot_dendrogram
from atlas.plotting.group_coherence import plot_group_coherence

__all__ = ["apply_style", "plot_dendrogram", "plot_group_coherence"]
