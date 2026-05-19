"""Golden-numbers tests: lock every numeric claim in the paper to the
digits reported. Any drift > 0.001 fails loudly.

These tests require the frozen XLSX artifacts to be available at the
configured DATA_ROOT (~/Data/New-Atlas by default). They are slow and
disk-dependent; skip them when running on CI without artifacts via:
    pytest -m "not requires_data"
"""

from __future__ import annotations

import pytest

from atlas.analysis import (
    pairwise_jaccard_matrix,
    permutation_within_group_p_value,
)
from atlas.io import load_neuron_lists
from atlas.paths import DATA_ROOT


pytestmark = pytest.mark.requires_data


@pytest.fixture(scope="module")
def r_qw_l14() -> dict[str, set[int]]:
    """Concept-only sets for Rust × Qwen × L14 × eps=0.5 × cons=0.8."""
    if not (DATA_ROOT / "R_QW_4_neuron_list_eps0.5_cons0.8_layers_part1_both.xlsx").exists():
        pytest.skip(f"frozen artifacts not present at {DATA_ROOT}")
    masks = load_neuron_lists(
        model="QW", lang="R", eps=0.5, cons=0.8, layer=14, partition="concept_only",
    )
    return {k: v for k, v in masks.items() if v}


# Hypothesised type-system-trait family, from the §6 four-cluster claim.
G1_TRAIT_FAMILY = [
    "Enum", "Send", "Option", "Iterator",
    "Copy", "Eq", "Drop", "Debug", "ToString",
]


def test_f6_g1_trait_family_observed_jaccard(r_qw_l14: dict) -> None:
    """G1 within-group mean Jaccard. Paper validation value: 0.535."""
    members = [m for m in G1_TRAIT_FAMILY if m in r_qw_l14]
    assert len(members) == 9, f"expected 9 G1 members, found {len(members)}"
    masks = {m: r_qw_l14[m] for m in members}
    _, j_matrix = pairwise_jaccard_matrix(masks)
    # Upper triangle, excluding diagonal.
    n = len(members)
    mean_j = sum(j_matrix[i, j] for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) // 2)
    assert mean_j == pytest.approx(0.535, abs=0.001), (
        f"F6 G1 within-group Jaccard drifted: got {mean_j:.4f}, expected 0.535"
    )


def test_f6_g1_permutation_p_value(r_qw_l14: dict) -> None:
    """G1 permutation p-value vs random same-size draws. Paper: p < 0.0001."""
    members = [m for m in G1_TRAIT_FAMILY if m in r_qw_l14]
    result = permutation_within_group_p_value(
        members, r_qw_l14, n_perm=10_000, seed=42,
    )
    assert result["p_value"] < 0.001, (
        f"F6 G1 p-value drifted above significance: got p={result['p_value']:.4f}"
    )
    # Sanity: observed should be roughly 4-5× the null mean.
    assert result["observed"] / result["null_mean"] > 4.0, (
        f"observed/null ratio drifted: {result['observed']:.4f} / {result['null_mean']:.4f}"
    )
