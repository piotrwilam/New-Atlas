"""Double-dissociation comparison from §7.1 zero-ablation deltas.

Paper §7.1: a clean double dissociation requires that ablating
concept-only neurons hurts log-prob on concept prompts *more* than on
checker prompts, *and* that this asymmetry exceeds what random-null
ablation produces. Four concepts (Import, Try, While, Assert) pass on
Qwen × Python; one documented negative (Break) fails because the
random null hurts it just as much.

This module computes the comparison given the four Δ-log-prob inputs.
The ablation forward passes that produce those Δs live in
`circuits.ablation` — this is pure compute on the resulting numbers,
so it stays in `atlas/` with no torch dependency.
"""

from __future__ import annotations

from typing import TypedDict


class DissociationScore(TypedDict):
    concept_only_dissociation: float
    random_null_dissociation: float
    margin: float
    passes: bool


def compute_dissociation_score(
    delta_concept_under_co: float,
    delta_checker_under_co: float,
    delta_concept_under_null: float,
    delta_checker_under_null: float,
    *,
    margin_threshold: float = 0.0,
) -> DissociationScore:
    """Per-(concept, layer) double-dissociation score.

    Inputs are signed log-prob deltas relative to the unablated baseline
    — negative means the prompt got *less* likely under ablation.

    `concept_only_dissociation` = Δ(concept prompts under concept-only ablation)
                                 − Δ(checker prompts under concept-only ablation)
    `random_null_dissociation`  = Δ(concept prompts under random-null ablation)
                                 − Δ(checker prompts under random-null ablation)
    `margin`                    = random_null_dissociation − concept_only_dissociation
                                  (positive means concept-only ablation is
                                  *more* dissociative than the random null)

    A concept "passes" the double-dissociation test when both:
      - concept_only_dissociation is negative (concept prompts hurt more
        than checker prompts under concept-only ablation), and
      - margin > margin_threshold (concept-only is more dissociative
        than random-null).

    `margin_threshold` defaults to 0.0 (strictly more dissociative than
    null); the paper uses this strict form.
    """
    co_diss = delta_concept_under_co - delta_checker_under_co
    null_diss = delta_concept_under_null - delta_checker_under_null
    margin = null_diss - co_diss
    passes = (co_diss < 0) and (margin > margin_threshold)
    return {
        "concept_only_dissociation": co_diss,
        "random_null_dissociation": null_diss,
        "margin": margin,
        "passes": passes,
    }
