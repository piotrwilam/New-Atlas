"""Tests for atlas.analysis.jaccard."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.analysis.jaccard import jaccard_sets, pairwise_jaccard_matrix


def test_jaccard_sets_disjoint() -> None:
    assert jaccard_sets({1, 2, 3}, {4, 5}) == 0.0


def test_jaccard_sets_identical() -> None:
    assert jaccard_sets({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_sets_partial(trio_masks: dict) -> None:
    # Hand-computed in conftest docstring.
    assert jaccard_sets(trio_masks["alpha"], trio_masks["beta"]) == pytest.approx(0.5)
    assert jaccard_sets(trio_masks["alpha"], trio_masks["gamma"]) == 0.0
    assert jaccard_sets(trio_masks["beta"], trio_masks["gamma"]) == pytest.approx(0.25)


def test_jaccard_sets_both_empty(empty_pair: dict) -> None:
    assert jaccard_sets(empty_pair["a"], empty_pair["b"]) == 0.0


def test_jaccard_sets_one_empty() -> None:
    assert jaccard_sets({1, 2}, set()) == 0.0


def test_pairwise_jaccard_matrix_shape_and_values(trio_masks: dict) -> None:
    names, matrix = pairwise_jaccard_matrix(trio_masks)
    assert names == ["alpha", "beta", "gamma"]
    assert matrix.shape == (3, 3)
    # symmetry
    assert np.allclose(matrix, matrix.T)
    # spot-check values vs the hand-computed Jaccards in the fixture
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[0, 2] == 0.0
    assert matrix[1, 2] == pytest.approx(0.25)
    # diagonal = 1 for non-empty sets
    assert np.allclose(np.diag(matrix), [1.0, 1.0, 1.0])
