"""Tests for atlas.analysis.dissociation — §7.1 double-dissociation comparison."""

from __future__ import annotations

import pytest

from atlas.analysis.dissociation import compute_dissociation_score


def test_clean_dissociation_passes() -> None:
    # Concept-only ablation: concept prompts hurt a lot (-3), checker prompts barely (-0.1).
    # Random null: neither hurt much (-0.3, -0.2).
    # Concept-only dissociation = -3 - (-0.1) = -2.9 (strongly negative)
    # Random-null dissociation  = -0.3 - (-0.2) = -0.1
    # Margin = -0.1 - (-2.9) = +2.8 (positive → concept-only more dissociative)
    r = compute_dissociation_score(
        delta_concept_under_co=-3.0,
        delta_checker_under_co=-0.1,
        delta_concept_under_null=-0.3,
        delta_checker_under_null=-0.2,
    )
    assert r["concept_only_dissociation"] == pytest.approx(-2.9)
    assert r["random_null_dissociation"] == pytest.approx(-0.1)
    assert r["margin"] == pytest.approx(2.8)
    assert r["passes"] is True


def test_random_null_more_dissociative_fails() -> None:
    # Documented negative result (Break): random-null ablation produces a
    # more negative dissociation than concept-only ablation does.
    r = compute_dissociation_score(
        delta_concept_under_co=-0.032,
        delta_checker_under_co=0.0,
        delta_concept_under_null=-0.066,
        delta_checker_under_null=0.0,
    )
    # co_diss = -0.032, null_diss = -0.066. margin = -0.066 - (-0.032) = -0.034 < 0.
    assert r["concept_only_dissociation"] == pytest.approx(-0.032)
    assert r["random_null_dissociation"] == pytest.approx(-0.066)
    assert r["margin"] < 0
    assert r["passes"] is False


def test_positive_co_dissociation_fails() -> None:
    # Even if concept-only is more dissociative than the null, concept-only
    # ablation that helps concept prompts (positive Δ) shouldn't pass.
    r = compute_dissociation_score(
        delta_concept_under_co=0.1,    # ↑ prob under ablation (weird)
        delta_checker_under_co=0.0,
        delta_concept_under_null=-0.05,
        delta_checker_under_null=0.0,
    )
    assert r["concept_only_dissociation"] > 0
    assert r["passes"] is False


def test_margin_threshold_strict() -> None:
    # Tied co_diss and null_diss → margin = 0 → strict threshold (>0) fails.
    r = compute_dissociation_score(
        delta_concept_under_co=-1.0,
        delta_checker_under_co=0.0,
        delta_concept_under_null=-1.0,
        delta_checker_under_null=0.0,
    )
    assert r["margin"] == 0.0
    assert r["passes"] is False  # strict threshold


def test_margin_threshold_relaxed() -> None:
    r = compute_dissociation_score(
        delta_concept_under_co=-1.0,
        delta_checker_under_co=0.0,
        delta_concept_under_null=-1.0,
        delta_checker_under_null=0.0,
        margin_threshold=-0.01,  # allow ties to pass
    )
    assert r["passes"] is True
