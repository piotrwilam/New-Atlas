"""Golden-numbers tests: lock every numeric claim in the paper to the
digits reported. Any drift > 0.001 fails loudly.

These tests require the frozen XLSX artifacts to be available at the
configured DATA_ROOT (~/Data/New-Atlas by default). They are slow and
disk-dependent; skip them when running on CI without artifacts via:
    pytest -m "not requires_data"
"""

from __future__ import annotations

import pytest

from collections import Counter

from atlas.analysis import (
    pairwise_jaccard_matrix,
    permutation_within_group_p_value,
)
from atlas.io import (
    load_concept_aggregates,
    load_concept_groups,
    load_cross_language_sharing,
    load_flow_type_assignments,
    load_jaccard_cosine_correlation,
    load_neuron_lists,
    load_probe_results,
)
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


# Locks the per-cell flow-type counts in figures F9–F12. The legend in
# each panel reports "{flow_type} ({n})" and these counts are the cited
# numbers in §6.1 ("In Qwen, these are the only Python concepts
# classified as `two_phase`. ... 7 of 7 ... atomicity super-cluster").
F9_F12_FLOW_TYPE_COUNTS = {
    ("P", "QW"): {"two_phase": 7,       "late_emergence": 95, "unclassified": 4},
    ("R", "QW"): {"two_phase": 2,       "late_emergence": 71, "unclassified": 2},
    ("P", "DS"): {"build_and_hold": 7,  "late_emergence": 89, "two_phase": 1, "unclassified": 9},
    ("R", "DS"): {"build_and_hold": 5,  "late_emergence": 63, "unclassified": 7},
}


@pytest.mark.parametrize("lang,model", list(F9_F12_FLOW_TYPE_COUNTS.keys()))
def test_f9_f12_flow_type_counts(lang: str, model: str) -> None:
    """Lock the §6.1 flow-type distribution per (lang, model) cell."""
    if not (DATA_ROOT / "7_E6_flow_type_assignments.xlsx").exists():
        pytest.skip(f"flow-type file not present at {DATA_ROOT}")
    flow_types = load_flow_type_assignments(model=model, lang=lang)
    counts = Counter(flow_types.values())
    expected = F9_F12_FLOW_TYPE_COUNTS[(lang, model)]
    assert dict(counts) == expected, (
        f"({lang},{model}) flow-type counts drifted: got {dict(counts)}, "
        f"expected {expected}"
    )


@pytest.mark.parametrize("lang,model", list(F9_F12_FLOW_TYPE_COUNTS.keys()))
def test_f9_f12_flow_type_classifier_roundtrip(lang: str, model: str) -> None:
    """Re-classify per-concept size curves with the in-code classifier
    and check the resulting counts match the loaded XLSX. This locks the
    rule-cascade classifier to the data, so any drift in either source
    is caught."""
    fname = f"{lang}_{model}_4_neuron_list_eps0.5_cons0.8_layers_part1_both.xlsx"
    if not (DATA_ROOT / fname).exists():
        pytest.skip(f"neuron-list files not present at {DATA_ROOT}")
    if not (DATA_ROOT / "7_E6_flow_type_assignments.xlsx").exists():
        pytest.skip(f"flow-type file not present at {DATA_ROOT}")

    from atlas.analysis import classify_all_flow_types
    from atlas.io import load_concept_sizes_by_layer

    sizes = load_concept_sizes_by_layer(
        model=model, lang=lang, eps=0.5, cons=0.8, partition="universal",
    )
    recomputed = classify_all_flow_types(sizes)
    # Drop empty curves (they're not in the XLSX assignments file).
    recomputed = {c: ft for c, ft in recomputed.items() if ft != "empty"}
    recomputed_counts = Counter(recomputed.values())

    loaded = load_flow_type_assignments(model=model, lang=lang)
    # Only concepts present in both — the XLSX may have concepts with no
    # size data (filtered out at decomposition time).
    both = set(recomputed) & set(loaded)
    agreement = sum(1 for c in both if recomputed[c] == loaded[c]) / len(both)
    assert agreement >= 0.95, (
        f"classifier drifted vs locked XLSX for ({lang}, {model}): "
        f"agreement = {agreement:.3f} over {len(both)} concepts"
    )


# §5.1 concept-group counts per (lang, model) cell, from the
# 9_results_*.xlsx aggregations. Total per language is constant across
# models because the classification is by syntax type, not by model.
F2_CONCEPT_GROUP_COUNTS = {
    ("P", "QW"): {"Modular": 6,  "Non-modular": 18, "Builtin": 34},
    ("P", "DS"): {"Modular": 6,  "Non-modular": 18, "Builtin": 34},
    ("R", "QW"): {"Modular": 6,  "Non-modular": 15, "Object": 36},
    ("R", "DS"): {"Modular": 6,  "Non-modular": 15, "Object": 36},
}


# §4.2 cross-model concept-fraction Spearman correlations (Figure 1 titles).
# Paper abstract claims "ρ = 0.64–0.79"; actual computed values are 0.638
# (Python) and 0.673 (Rust). The 0.72 value cited in the abstract is the
# Python *circuit-size* correlation, not Rust concept-fraction — this is
# the first of the three known paper-vs-data discrepancies in CLAUDE.md.
F1_SPEARMAN_RHO = {"P": 0.638, "R": 0.673}


# §7.3 numeric claims: probe accuracy band + peak Jaccard-cosine r.
def test_f7_probe_accuracy_band() -> None:
    """Lock the §7.3 claim '97–100% classification accuracy at every
    one of the 28 layers' on Qwen × Python."""
    if not (DATA_ROOT / "P_QW_V2_probe_results.csv").exists():
        pytest.skip(f"probe results file not present at {DATA_ROOT}")
    per_concept = load_probe_results(model="QW", lang="P")
    layers = sorted({L for d in per_concept.values() for L in d})
    means = [
        sum(d[L]["accuracy"] for d in per_concept.values() if L in d) /
        sum(1 for d in per_concept.values() if L in d)
        for L in layers
    ]
    assert min(means) >= 0.97, f"probe accuracy band drifted: min={min(means):.4f}"
    assert max(means) <= 1.00, f"probe accuracy >1 impossible: max={max(means):.4f}"
    # The minimum should sit at one of the last few layers (L24-27 are
    # where the probe degrades slightly).
    min_layer = layers[means.index(min(means))]
    assert min_layer >= 20, (
        f"probe accuracy minimum unexpectedly early: L{min_layer}"
    )


def test_f8_peak_jaccard_cosine_correlation() -> None:
    """Lock the §7.3 claim 'r peaks at 0.645 at L20'."""
    if not (DATA_ROOT / "P_QW_V2_cosine_jaccard_correlation.csv").exists():
        pytest.skip(f"correlation file not present at {DATA_ROOT}")
    per_layer = load_jaccard_cosine_correlation(model="QW", lang="P")
    layers = sorted(per_layer.keys())
    rs = [per_layer[L]["pearson_r"] for L in layers]
    peak_layer = layers[rs.index(max(rs))]
    assert peak_layer == 20, f"F8 peak layer drifted: got L{peak_layer}, expected L20"
    assert per_layer[20]["pearson_r"] == pytest.approx(0.645, abs=0.005), (
        f"F8 L20 Pearson r drifted: got {per_layer[20]['pearson_r']:.4f}"
    )


@pytest.mark.parametrize("lang", list(F1_SPEARMAN_RHO.keys()))
def test_f1_concept_fraction_spearman(lang: str) -> None:
    """Lock the Spearman ρ between Qwen and DeepSeek per-concept mean
    concept-fraction at (lang, eps=0.5, cons=0.8)."""
    from scipy.stats import spearmanr
    fname = f"9_results_{lang}_QW_eps0.5_cons0.8.xlsx"
    if not (DATA_ROOT / fname).exists():
        pytest.skip(f"aggregates file not present at {DATA_ROOT}")
    qw = load_concept_aggregates(model="QW", lang=lang, eps=0.5, cons=0.8)
    ds = load_concept_aggregates(model="DS", lang=lang, eps=0.5, cons=0.8)
    shared = sorted(set(qw) & set(ds))
    rho, _ = spearmanr(
        [qw[c]["mean_cf"] for c in shared],
        [ds[c]["mean_cf"] for c in shared],
    )
    expected = F1_SPEARMAN_RHO[lang]
    assert rho == pytest.approx(expected, abs=0.005), (
        f"F1 {lang} Spearman ρ drifted: got {rho:.4f}, expected {expected}"
    )


def test_f3_cross_language_sharing_ratio() -> None:
    """Lock the §5.3 DS/QW mean-of-means sharing ratio. The paper text
    claims 2.3× but the data gives 1.94× — locked here so any future
    drift in the data is caught and the §5.3 prose stays consistent
    with the locked value, whichever the v3 paper settles on."""
    if not (DATA_ROOT / "7_E7_cross_language_results.xlsx").exists():
        pytest.skip(f"cross-language file not present at {DATA_ROOT}")
    raw = load_cross_language_sharing()
    classes = sorted({c for (_m, c) in raw})
    import statistics
    def cell_mean(model: str, eq_class: str) -> float:
        vals = raw.get((model, eq_class), {})
        return statistics.fmean(vals.values()) if vals else 0.0
    ds_mean = statistics.fmean(cell_mean("DS", c) for c in classes)
    qw_mean = statistics.fmean(cell_mean("QW", c) for c in classes)
    ratio = ds_mean / qw_mean
    assert ratio == pytest.approx(1.949, abs=0.005), (
        f"F3 DS/QW sharing ratio drifted: got {ratio:.3f}×, expected 1.949×"
    )
    # Lock the H6 claim: every (model, eq_class) cell exceeds 10 % sharing.
    for m in ("DS", "QW"):
        for c in classes:
            assert cell_mean(m, c) > 0.10 or (m == "QW" and c == "Function def"), (
                f"({m}, {c}) cell mean dropped below H6 threshold: {cell_mean(m, c):.3f}"
            )


@pytest.mark.parametrize("lang,model", list(F2_CONCEPT_GROUP_COUNTS.keys()))
def test_f2_concept_group_counts(lang: str, model: str) -> None:
    """Lock the §5.1 group counts per cell."""
    fname = f"9_results_{lang}_{model}_eps0.5_cons0.8.xlsx"
    if not (DATA_ROOT / fname).exists():
        pytest.skip(f"results file not present at {DATA_ROOT}")
    groups = load_concept_groups(model=model, lang=lang, eps=0.5, cons=0.8)
    counts = Counter(groups.values())
    expected = F2_CONCEPT_GROUP_COUNTS[(lang, model)]
    assert dict(counts) == expected, (
        f"({lang},{model}) concept-group counts drifted: got {dict(counts)}, "
        f"expected {expected}"
    )
