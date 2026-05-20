"""Tests for atlas.analysis.cross_language — equivalence-class pooling + sharing."""

from __future__ import annotations

import pytest

from atlas.analysis.cross_language import (
    EQUIVALENCE_CLASSES,
    cross_language_sharing_fraction,
    pool_neurons,
)


def test_pool_neurons_basic() -> None:
    masks = {"For": {1, 2, 3}, "While": {3, 4, 5}, "Loop": {5, 6}}
    pool = pool_neurons(masks, ["For", "While", "Loop"])
    assert pool == {1, 2, 3, 4, 5, 6}


def test_pool_neurons_missing_silently_skipped() -> None:
    masks = {"For": {1, 2}}
    pool = pool_neurons(masks, ["For", "While", "Loop"])  # Only For exists.
    assert pool == {1, 2}


def test_pool_neurons_empty_members() -> None:
    assert pool_neurons({"a": {1}}, []) == set()


def test_sharing_fraction_identical_pools() -> None:
    assert cross_language_sharing_fraction({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)


def test_sharing_fraction_disjoint_pools() -> None:
    assert cross_language_sharing_fraction({1, 2}, {3, 4}) == 0.0


def test_sharing_fraction_partial_normalised_by_smaller() -> None:
    # py_pool = {1, 2, 3, 4} (4), rs_pool = {1, 2} (2). shared = {1, 2} (2).
    # min_pool = 2 → 2/2 = 1.0.
    assert cross_language_sharing_fraction({1, 2, 3, 4}, {1, 2}) == pytest.approx(1.0)


def test_sharing_fraction_empty_pool() -> None:
    assert cross_language_sharing_fraction(set(), {1, 2, 3}) == 0.0
    assert cross_language_sharing_fraction({1, 2, 3}, set()) == 0.0


def test_default_equivalence_classes_has_paper_classes() -> None:
    # The 7 classes that show up in F3 must be present (Iteration, Branching,
    # Function def, Loop control, Module import, Return, Type def).
    expected = {"Iteration", "Branching", "Function def", "Loop control",
                "Module import", "Return", "Type def"}
    assert expected.issubset(EQUIVALENCE_CLASSES.keys())


def test_default_equivalence_classes_have_both_langs() -> None:
    for name, classes in EQUIVALENCE_CLASSES.items():
        assert "P" in classes, f"{name} missing Python members"
        assert "R" in classes, f"{name} missing Rust members"
        assert classes["P"], f"{name} has empty Python member list"
        assert classes["R"], f"{name} has empty Rust member list"
