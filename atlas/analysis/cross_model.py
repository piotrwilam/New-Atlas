"""Cross-model correlation of per-concept measurements.

Paper §4.2: do Qwen and DeepSeek agree on *which* concepts are
syntactically distinctive? Pair each concept's mean concept-fraction
under one model with its value under the other, then test the
correlation. Spearman rank correlation is the paper's reported
statistic because the absolute magnitudes differ across models but
the rankings are expected to be conserved.

The §4.2 numbers from this function (with the frozen data):
    Python: ρ = 0.638, p ≈ 7e-8 (n = 58)
    Rust:   ρ = 0.673, p ≈ 1e-8 (n = 57)
"""

from __future__ import annotations

from typing import Literal

from scipy.stats import pearsonr, spearmanr


def cross_model_correlation(
    qw_values: list[float],
    ds_values: list[float],
    *,
    method: Literal["spearman", "pearson"] = "spearman",
) -> tuple[float, float]:
    """Correlation between paired per-concept measurements under two models.

    Parameters
    ----------
    qw_values, ds_values
        Per-concept measurements under the two models. Must be the same
        length and in the same concept order — caller is responsible for
        the pairing.
    method
        "spearman" (default — rank correlation, the paper's headline
        statistic) or "pearson" (linear correlation).

    Returns
    -------
    (statistic, p_value) — the correlation coefficient and its two-sided
    p-value.
    """
    assert len(qw_values) == len(ds_values), (
        f"length mismatch: qw={len(qw_values)}, ds={len(ds_values)}"
    )
    assert len(qw_values) >= 3, (
        f"need at least 3 paired values for a meaningful correlation, "
        f"got {len(qw_values)}"
    )
    assert method in {"spearman", "pearson"}, (
        f"method must be 'spearman' or 'pearson', got {method!r}"
    )
    if method == "spearman":
        rho, p = spearmanr(qw_values, ds_values)
    else:
        rho, p = pearsonr(qw_values, ds_values)
    return float(rho), float(p)
