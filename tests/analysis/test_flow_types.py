"""Tests for atlas.analysis.flow_types — circuit-size-curve classifier."""

from __future__ import annotations

import pytest

from atlas.analysis.flow_types import classify_all_flow_types, classify_flow_type


def test_empty_curve() -> None:
    assert classify_flow_type([0, 0, 0, 0, 0, 0, 0, 0]) == "empty"


def test_late_emergence_synthetic() -> None:
    # Quiet first half, peak in second half: classic late-emergence.
    sizes = [0, 0, 0, 0, 0, 0, 50, 200, 800, 3000, 3500, 3500]
    assert classify_flow_type(sizes) == "late_emergence"


def test_two_phase_synthetic() -> None:
    # Spike at L1-2, dip to near zero through the middle, late re-explosion
    # that peaks short of the endpoint (so the late peak is detectable as
    # a local maximum). Mirrors the Qwen two_phase shape.
    sizes = [0, 2000, 1800, 100, 50, 30, 20, 10, 5, 50, 3000, 3500, 3400, 100]
    assert classify_flow_type(sizes) == "two_phase"


def test_build_and_hold_synthetic() -> None:
    # Monotone gradual rise to a wide plateau.
    sizes = [0, 50, 100, 300, 800, 1500, 2200, 2800, 3200, 3500, 3700, 3700, 3700, 3700]
    assert classify_flow_type(sizes) == "build_and_hold"


def test_flash_synthetic() -> None:
    # One tall narrow spike, rest near zero. mean is small relative to peak.
    sizes = [0, 0, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert classify_flow_type(sizes) == "flash"


def test_unclassified_synthetic() -> None:
    # Noisy moderate curve that triggers no rule.
    sizes = [100, 150, 120, 180, 140, 160, 130, 170, 110, 190, 100]
    assert classify_flow_type(sizes) == "unclassified"


def test_short_curve_rejected() -> None:
    with pytest.raises(AssertionError, match="length ≥ 4"):
        classify_flow_type([1, 2, 3])


def test_batch_classify_preserves_concepts() -> None:
    sizes_by_concept = {
        "Empty":    {L: 0 for L in range(8)},
        "Late":     {0: 0, 1: 0, 2: 0, 3: 0, 4: 50, 5: 500, 6: 3000, 7: 3500},
    }
    out = classify_all_flow_types(sizes_by_concept)
    assert out["Empty"] == "empty"
    assert out["Late"] == "late_emergence"
