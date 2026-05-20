"""Shared pytest fixtures for the atlas test suite.

Use tiny synthetic inputs so tests are fast, deterministic, and obvious.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def trio_masks() -> dict[str, set[int]]:
    """Three small concept-only sets with hand-computed Jaccard values.

    Pairwise expected:
        alpha ∩ beta  = {2, 3}        |alpha ∪ beta|  = 4 → J = 2/4 = 0.5
        alpha ∩ gamma = {}            |alpha ∪ gamma| = 5 → J = 0
        beta  ∩ gamma = {5}           |beta  ∪ gamma| = 4 → J = 1/4 = 0.25
    """
    return {
        "alpha": {1, 2, 3},
        "beta": {2, 3, 5},
        "gamma": {4, 5},
    }


@pytest.fixture
def empty_pair() -> dict[str, set[int]]:
    """Two empty sets. Jaccard convention: 0.0."""
    return {"a": set(), "b": set()}
