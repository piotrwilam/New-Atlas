"""Hierarchical clustering and group-coherence tests on neuron-set masks.

Wraps scipy's linkage with a fixed, documented choice of method and
distance metric so the dendrogram is reproducible across scipy versions
(within the limits of scipy's tie-breaking; if scipy changes that, the
linkage array changes, which is why the golden-numbers test exists).
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster as _scipy_fcluster
from scipy.cluster.hierarchy import linkage as _scipy_linkage
from scipy.spatial.distance import squareform

from atlas.analysis.jaccard import jaccard_sets, pairwise_jaccard_matrix


def ward_linkage_from_jaccard(jaccard_matrix: np.ndarray) -> np.ndarray:
    """Ward-linkage hierarchical clustering on (1 - Jaccard) distance.

    Parameters
    ----------
    jaccard_matrix  (n, n) symmetric similarity matrix in [0, 1].

    Returns
    -------
    linkage matrix Z of shape (n-1, 4) as produced by scipy.cluster.hierarchy.linkage.
    Compatible with scipy.cluster.hierarchy.dendrogram.

    Pure. Convention: distance = 1 - similarity. Diagonal of input must be 1.
    """
    assert jaccard_matrix.ndim == 2 and jaccard_matrix.shape[0] == jaccard_matrix.shape[1], (
        f"expected square 2-D matrix, got shape {jaccard_matrix.shape}"
    )
    assert np.allclose(jaccard_matrix, jaccard_matrix.T), "matrix must be symmetric"

    distance_matrix = 1.0 - jaccard_matrix
    np.fill_diagonal(distance_matrix, 0.0)  # guard against any rounding-induced ~0
    condensed = squareform(distance_matrix, checks=False)
    return _scipy_linkage(condensed, method="ward")


def cut_dendrogram_at_k_clusters(
    linkage: np.ndarray,
    labels: list[str],
    k: int,
) -> dict[str, int]:
    """Cut a Ward linkage dendrogram into exactly `k` flat clusters.

    Used in §6.2 to discover the four-cluster Rust meta-structure:
    after computing pairwise Jaccard and applying Ward linkage,
    cutting at k=4 produces the {type-system traits, memory/ownership,
    data definition, control-flow/module} groupings that the F6
    permutation test then validates.

    Returns
    -------
    dict mapping leaf label → cluster id (1-indexed, as scipy reports).
    """
    assert linkage.ndim == 2 and linkage.shape[1] == 4, (
        f"linkage must be (n-1, 4), got shape {linkage.shape}"
    )
    assert linkage.shape[0] == len(labels) - 1, (
        f"linkage has {linkage.shape[0]} rows but labels has {len(labels)} entries "
        f"(expected {linkage.shape[0] + 1})"
    )
    assert k >= 1, f"k must be ≥ 1, got {k}"
    assert k <= len(labels), (
        f"cannot cut into more clusters ({k}) than leaves ({len(labels)})"
    )

    cluster_ids = _scipy_fcluster(linkage, t=k, criterion="maxclust")
    return {label: int(cid) for label, cid in zip(labels, cluster_ids)}


def mean_pairwise_jaccard(members: list[str], masks: dict[str, set[int]]) -> float:
    """Mean pairwise Jaccard within a named group of concepts. Pure."""
    assert len(members) >= 2, f"need at least 2 members, got {len(members)}"
    assert all(m in masks for m in members), (
        f"missing masks for: {[m for m in members if m not in masks]}"
    )

    pairs = list(combinations(members, 2))
    return sum(jaccard_sets(masks[a], masks[b]) for a, b in pairs) / len(pairs)


def permutation_within_group_p_value(
    members: list[str],
    masks: dict[str, set[int]],
    *,
    n_perm: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """One-sided permutation test for within-group Jaccard cohesion.

    Null: members are no more similar to each other than a same-size
    random draw from `masks`. The test statistic is the mean pairwise
    Jaccard within the group.

    Parameters
    ----------
    members  list of names whose cohesion we test.
    masks    dict of name → set; supplies the universe of names to sample from.
    n_perm   number of permutations (default 10,000).
    seed     RNG seed; locked at the function level so all callers get
             identical p-values from identical inputs.

    Returns
    -------
    dict with keys: observed, null_mean, null_p95, p_value. p_value is the
    one-sided fraction of nulls ≥ observed.
    """
    assert len(members) >= 2, f"need at least 2 members, got {len(members)}"
    assert n_perm >= 100, f"n_perm must be at least 100 for a meaningful p-value, got {n_perm}"

    rng = random.Random(seed)
    universe = list(masks.keys())
    n = len(members)

    observed = mean_pairwise_jaccard(members, masks)
    nulls = [mean_pairwise_jaccard(rng.sample(universe, n), masks) for _ in range(n_perm)]
    nulls.sort()
    p95_index = int(0.95 * n_perm)
    return {
        "observed": observed,
        "null_mean": float(np.mean(nulls)),
        "null_p95": nulls[p95_index],
        "p_value": sum(1 for v in nulls if v >= observed) / n_perm,
    }
