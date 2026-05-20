"""Cross-validate the neuron view against the probe-direction view.

Paper §7.3: at each layer, compute pairwise Jaccard similarity between
concept-only neuron sets and pairwise cosine similarity between the
corresponding linear-probe direction vectors. If the two views agree on
the coarse structure — which concepts cluster with which — their
per-pair values should correlate.

The paper's headline result: Pearson r peaks at 0.645 at L20 on
P × Qwen. The moderate (not unity) correlation is informative — it says
the two views agree on the head of the distribution (high-amplitude
neurons) but diverge in the tails (sub-threshold distributed signal
that probes pick up but binary masks don't).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import pearsonr

from atlas.analysis.jaccard import jaccard_sets


def pairwise_cosine_vs_jaccard(
    weights: dict[str, np.ndarray],
    masks: dict[str, set[int]],
) -> tuple[list[float], list[float], float]:
    """Pairwise (Jaccard, cosine) data and their Pearson correlation.

    For every pair of concepts present in *both* `weights` and `masks`,
    compute the Jaccard of their concept-only neuron sets and the
    cosine similarity of their L2-normalised probe direction vectors.

    Parameters
    ----------
    weights
        concept → probe direction vector (1-D ndarray, length = MLP
        hidden dim). Typically the output of
        `atlas.io.load_probe_weights(..., layer=L)` for a single layer.
    masks
        concept → concept-only neuron set. Typically the output of
        `atlas.io.load_neuron_lists(..., layer=L)` for the same layer.

    Returns
    -------
    `(jaccards, cosines, pearson_r)` — two same-length lists (one entry
    per concept pair, in deterministic name-sorted order) and the
    Pearson correlation between them.
    """
    shared = sorted(set(weights) & set(masks))
    assert len(shared) >= 2, (
        f"need at least 2 concepts in both inputs, got {len(shared)}"
    )

    norms = {}
    for c in shared:
        v = weights[c]
        norm = np.linalg.norm(v)
        assert norm > 0, f"probe vector for concept {c!r} has zero norm"
        norms[c] = v / norm

    jaccards: list[float] = []
    cosines: list[float] = []
    for a, b in combinations(shared, 2):
        jaccards.append(jaccard_sets(masks[a], masks[b]))
        cosines.append(float(np.dot(norms[a], norms[b])))

    r, _ = pearsonr(jaccards, cosines)
    return jaccards, cosines, float(r)
