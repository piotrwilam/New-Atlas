"""Tests for circuits.probes — linear-probe training wrapper.

Skip if sklearn / torch isn't installed (these tests need sklearn).
"""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from circuits.probes import train_concept_probe  # noqa: E402


def test_perfectly_separable_data_high_accuracy() -> None:
    rng = np.random.default_rng(0)
    n, d = 80, 16
    # Class 1: cluster around +2 along dim 0; class 0: cluster around -2.
    # Wide separation → linearly separable up to noise.
    X_obj = rng.standard_normal((n, d)) + np.array([2.0] + [0.0] * (d - 1))
    X_chk = rng.standard_normal((n, d)) + np.array([-2.0] + [0.0] * (d - 1))
    out = train_concept_probe(X_obj, X_chk, seed=42)
    assert out is not None
    assert out["accuracy"] > 0.95
    # Weight along dim 0 should be the largest in magnitude.
    assert abs(out["weight_vector"][0]) > abs(out["weight_vector"][5])
    assert 0 in out["top10_neurons"][:3]


def test_returns_unit_normalised_weights() -> None:
    rng = np.random.default_rng(1)
    X_obj = rng.standard_normal((20, 8)) + np.array([1.0] + [0.0] * 7)
    X_chk = rng.standard_normal((20, 8))
    out = train_concept_probe(X_obj, X_chk, seed=42)
    assert out is not None
    assert np.linalg.norm(out["weight_vector"]) == pytest.approx(1.0, abs=1e-5)


def test_insufficient_samples_returns_none() -> None:
    X_obj = np.zeros((3, 8))
    X_chk = np.zeros((10, 8))
    out = train_concept_probe(X_obj, X_chk, min_samples_per_class=5)
    assert out is None


def test_dim_mismatch_rejected() -> None:
    with pytest.raises(AssertionError, match="feature dim mismatch"):
        train_concept_probe(np.zeros((10, 8)), np.zeros((10, 9)))


def test_seed_determinism() -> None:
    rng = np.random.default_rng(42)
    X_obj = rng.standard_normal((30, 8)) + 0.5
    X_chk = rng.standard_normal((30, 8)) - 0.5
    out1 = train_concept_probe(X_obj, X_chk, seed=42)
    out2 = train_concept_probe(X_obj, X_chk, seed=42)
    assert out1 is not None and out2 is not None
    np.testing.assert_allclose(out1["weight_vector"], out2["weight_vector"])
    assert out1["accuracy"] == out2["accuracy"]
