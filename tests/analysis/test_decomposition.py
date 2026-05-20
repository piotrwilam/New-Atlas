"""Tests for atlas.analysis.decomposition — three-way set decomposition and concept fraction."""

from __future__ import annotations

import pytest

from atlas.analysis.decomposition import concept_fraction, decompose_sets


def test_decompose_sets_disjoint() -> None:
    co, sh, to = decompose_sets({1, 2, 3}, {4, 5, 6})
    assert co == {1, 2, 3}
    assert sh == set()
    assert to == {4, 5, 6}


def test_decompose_sets_identical() -> None:
    co, sh, to = decompose_sets({1, 2, 3}, {1, 2, 3})
    assert co == set()
    assert sh == {1, 2, 3}
    assert to == set()


def test_decompose_sets_partial_overlap() -> None:
    co, sh, to = decompose_sets({1, 2, 3, 4}, {3, 4, 5, 6})
    assert co == {1, 2}
    assert sh == {3, 4}
    assert to == {5, 6}


def test_decompose_sets_returns_disjoint_partition() -> None:
    A, B = {1, 2, 3, 4, 5}, {3, 4, 5, 6, 7}
    co, sh, to = decompose_sets(A, B)
    # Three partitions must be pairwise disjoint and cover A ∪ B.
    assert co & sh == set() and co & to == set() and sh & to == set()
    assert co | sh | to == A | B


def test_decompose_sets_inputs_unchanged() -> None:
    A, B = {1, 2, 3}, {2, 3, 4}
    A_copy, B_copy = set(A), set(B)
    decompose_sets(A, B)
    assert A == A_copy and B == B_copy


def test_concept_fraction_partial() -> None:
    # concept_only=10, universal=40 → 0.25
    assert concept_fraction(10, 40) == pytest.approx(0.25)


def test_concept_fraction_full() -> None:
    # All universal neurons are concept-only (B is empty).
    assert concept_fraction(50, 50) == pytest.approx(1.0)


def test_concept_fraction_none() -> None:
    # No concept-only signal (every universal neuron is also in B).
    assert concept_fraction(0, 50) == pytest.approx(0.0)


def test_concept_fraction_empty_universal() -> None:
    # Convention: 0.0 when universal is empty.
    assert concept_fraction(0, 0) == 0.0


def test_concept_fraction_rejects_invalid() -> None:
    with pytest.raises(AssertionError):
        concept_fraction(10, 5)  # concept_only > universal
    with pytest.raises(AssertionError):
        concept_fraction(-1, 10)
