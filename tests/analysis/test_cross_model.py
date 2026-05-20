"""Tests for atlas.analysis.cross_model — Spearman/Pearson correlation wrapper."""

from __future__ import annotations

import pytest

from atlas.analysis.cross_model import cross_model_correlation


def test_spearman_perfect_rank_agreement() -> None:
    # Same rank order under both "models" → ρ = 1.0.
    qw = [0.1, 0.2, 0.3, 0.4, 0.5]
    ds = [10.0, 20.0, 30.0, 40.0, 50.0]
    rho, p = cross_model_correlation(qw, ds, method="spearman")
    assert rho == pytest.approx(1.0)
    assert p < 0.01


def test_spearman_perfect_anticorrelation() -> None:
    qw = [1, 2, 3, 4, 5]
    ds = [5, 4, 3, 2, 1]
    rho, _ = cross_model_correlation(qw, ds, method="spearman")
    assert rho == pytest.approx(-1.0)


def test_pearson_basic() -> None:
    qw = [1.0, 2.0, 3.0, 4.0, 5.0]
    ds = [2.0, 4.1, 6.0, 8.1, 10.0]  # nearly linear, slope ≈ 2
    r, _ = cross_model_correlation(qw, ds, method="pearson")
    assert r > 0.99


def test_length_mismatch_raises() -> None:
    with pytest.raises(AssertionError, match="length mismatch"):
        cross_model_correlation([1, 2, 3], [1, 2, 3, 4])


def test_invalid_method_raises() -> None:
    with pytest.raises(AssertionError, match="method must be"):
        cross_model_correlation([1, 2, 3], [1, 2, 3], method="kendall")  # type: ignore[arg-type]


def test_too_few_values_raises() -> None:
    with pytest.raises(AssertionError, match="at least 3"):
        cross_model_correlation([1, 2], [1, 2])
