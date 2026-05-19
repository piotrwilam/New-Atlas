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


# All four hypothesised groups from §6.2, with expected per-group statistics
# produced by experiments/fig6_four_cluster_test.py against the frozen
# R_QW × eps=0.5 × cons=0.8 × L14 artifacts. The permutation null samples
# from the FULL Rust concept pool (including concepts whose concept-only
# mask is empty at L14), so the null mean is shared across the four tests.
@pytest.fixture(scope="module")
def r_qw_l14_with_empties() -> dict[str, set[int]]:
    """Concept-only sets including those that are empty at L14. The null
    universe in the F6 test samples from this full pool."""
    if not (DATA_ROOT / "R_QW_4_neuron_list_eps0.5_cons0.8_layers_part1_both.xlsx").exists():
        pytest.skip(f"frozen artifacts not present at {DATA_ROOT}")
    return load_neuron_lists(
        model="QW", lang="R", eps=0.5, cons=0.8, layer=14, partition="concept_only",
    )


F6_GROUPS = {
    "G1_type_system_traits": (
        ["Enum", "Send", "Option", "Iterator", "Copy", "Eq", "Drop", "Debug", "ToString"],
        0.535,    # observed_J
        0.000,    # expected p_value (upper bound 0.001)
    ),
    "G2_memory_ownership": (
        ["Pin", "Mutex", "Arc", "Box", "Vec", "Impl", "Async", "Move", "Future"],
        0.190,
        0.044,
    ),
    "G3_data_definition": (
        ["Struct", "Let", "String", "Trait", "Default", "Display", "Hash", "Return", "Read"],
        0.199,
        0.035,
    ),
    "G4_control_flow_module": (
        ["Fn", "Pub", "Use", "Crate", "While", "Break", "Loop", "If", "Match"],
        0.129,
        0.292,
    ),
}


@pytest.mark.parametrize("group_name", list(F6_GROUPS.keys()))
def test_f6_four_cluster_observed_and_p_value(
    r_qw_l14_with_empties: dict, group_name: str,
) -> None:
    """Lock the §6.2 four-cluster claim: observed J and permutation p
    for each hypothesised group at R_QW × L14. Drift > 0.005 fails."""
    members, expected_j, expected_p = F6_GROUPS[group_name]
    missing = [m for m in members if m not in r_qw_l14_with_empties]
    assert not missing, f"{group_name}: members missing from data: {missing}"

    result = permutation_within_group_p_value(
        members, r_qw_l14_with_empties, n_perm=10_000, seed=42,
    )
    assert result["observed"] == pytest.approx(expected_j, abs=0.005), (
        f"{group_name} observed J drifted: got {result['observed']:.4f}, "
        f"expected {expected_j}"
    )
    if expected_p == 0.000:
        assert result["p_value"] < 0.001, (
            f"{group_name} p-value drifted above significance: "
            f"got p={result['p_value']:.4f}"
        )
    else:
        assert result["p_value"] == pytest.approx(expected_p, abs=0.005), (
            f"{group_name} p-value drifted: got p={result['p_value']:.4f}, "
            f"expected {expected_p}"
        )
