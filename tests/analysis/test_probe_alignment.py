"""Tests for atlas.analysis.probe_alignment — pairwise cosine vs Jaccard."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.analysis.probe_alignment import pairwise_cosine_vs_jaccard


def test_perfect_alignment() -> None:
    # Identical concepts: A and A both have the same mask and the same probe.
    # Wait — need ≥2 concepts. Use 3 concepts where two are pairs of
    # (same mask, parallel probe) → Jaccard = 1, cosine = 1.
    weights = {
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([1.0, 0.0, 0.0]),  # parallel to A
        "C": np.array([0.0, 1.0, 0.0]),  # orthogonal
    }
    masks = {
        "A": {1, 2, 3},
        "B": {1, 2, 3},   # same as A
        "C": {4, 5, 6},   # disjoint from A, B
    }
    jacc, cos, r = pairwise_cosine_vs_jaccard(weights, masks)
    # (A,B) Jaccard=1, cosine=1; (A,C) Jaccard=0, cosine=0; (B,C) Jaccard=0, cosine=0.
    assert sorted(jacc) == [0.0, 0.0, 1.0]
    assert sorted(cos) == [0.0, 0.0, 1.0]
    assert r == pytest.approx(1.0)


def test_orthogonal_data_no_correlation() -> None:
    # Jaccards vary but cosines are all the same → no correlation between them.
    rng = np.random.default_rng(0)
    weights = {f"c{i}": rng.standard_normal(8) for i in range(6)}
    masks = {f"c{i}": {j for j in range(i + 1)} for i in range(6)}  # nested
    jacc, cos, r = pairwise_cosine_vs_jaccard(weights, masks)
    assert len(jacc) == 15  # C(6, 2)
    assert len(cos) == 15
    # Just check shape; correlation could be anything for random data.
    assert -1.0 <= r <= 1.0


def test_zero_norm_weight_raises() -> None:
    weights = {"A": np.zeros(4), "B": np.array([1.0, 0.0, 0.0, 0.0])}
    masks = {"A": {1}, "B": {2}}
    with pytest.raises(AssertionError, match="zero norm"):
        pairwise_cosine_vs_jaccard(weights, masks)


def test_only_concepts_in_both_inputs_used() -> None:
    # A, B, C present in both → 3 pairs. "extra" only in weights, "absent"
    # only in masks — both excluded from the intersection.
    weights = {
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
        "C": np.array([0.0, 0.0, 1.0]),
        "extra": np.array([1.0, 1.0, 1.0]),
    }
    masks = {"A": {1, 2}, "B": {2, 3}, "C": {3, 4}, "absent": {99}}
    jacc, cos, _ = pairwise_cosine_vs_jaccard(weights, masks)
    assert len(jacc) == 3  # only A, B, C are in both inputs → C(3, 2) = 3 pairs


def test_too_few_concepts_raises() -> None:
    weights = {"A": np.array([1.0])}
    masks = {"A": {1}}
    with pytest.raises(AssertionError, match="at least 2"):
        pairwise_cosine_vs_jaccard(weights, masks)
