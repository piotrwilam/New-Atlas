"""Test for atlas.analysis.meta_circuits.cut_dendrogram_at_k_clusters.

The four-cluster discovery in §6.2 — given pairwise Jaccards, build a
Ward linkage, then cut into K flat clusters.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.analysis import (
    cut_dendrogram_at_k_clusters,
    pairwise_jaccard_matrix,
    ward_linkage_from_jaccard,
)


def test_cut_into_two_clusters_separates_disjoint_groups() -> None:
    # Two pairs of identical concepts {A=B} and {C=D}, with the two pairs
    # disjoint from each other. Cutting at K=2 should put A,B together
    # and C,D together.
    masks = {
        "A": {1, 2, 3},
        "B": {1, 2, 3},
        "C": {10, 11, 12},
        "D": {10, 11, 12},
    }
    names, j = pairwise_jaccard_matrix(masks)
    linkage = ward_linkage_from_jaccard(j)
    clusters = cut_dendrogram_at_k_clusters(linkage, names, k=2)

    # A and B share a cluster; C and D share a cluster; the two clusters differ.
    assert clusters["A"] == clusters["B"]
    assert clusters["C"] == clusters["D"]
    assert clusters["A"] != clusters["C"]


def test_cut_into_one_cluster_puts_everything_together() -> None:
    masks = {"A": {1}, "B": {2}, "C": {3}}
    names, j = pairwise_jaccard_matrix(masks)
    linkage = ward_linkage_from_jaccard(j)
    clusters = cut_dendrogram_at_k_clusters(linkage, names, k=1)
    assert len(set(clusters.values())) == 1


def test_cut_into_n_clusters_each_singleton() -> None:
    masks = {"A": {1}, "B": {2}, "C": {3}}
    names, j = pairwise_jaccard_matrix(masks)
    linkage = ward_linkage_from_jaccard(j)
    clusters = cut_dendrogram_at_k_clusters(linkage, names, k=3)
    assert len(set(clusters.values())) == 3


def test_invalid_k_rejected() -> None:
    masks = {"A": {1}, "B": {2}}
    names, j = pairwise_jaccard_matrix(masks)
    linkage = ward_linkage_from_jaccard(j)
    with pytest.raises(AssertionError, match="k must be"):
        cut_dendrogram_at_k_clusters(linkage, names, k=0)
    with pytest.raises(AssertionError, match="cannot cut"):
        cut_dendrogram_at_k_clusters(linkage, names, k=3)


def test_mismatched_labels_rejected() -> None:
    linkage = np.array([[0., 1., 0.5, 2.]])
    with pytest.raises(AssertionError, match="labels has"):
        cut_dendrogram_at_k_clusters(linkage, ["only_one_label"], k=1)
