# Paper Map — Atlas2x2

A section-by-section index of the paper claims, the code that produced
them, and the tests that lock the numbers. **Read this while writing
the paper.**

Audience: the paper author and reviewers asking "where does this number
come from?". Indexed by **paper section**, not by code structure
(unlike the [README](../README.md), which is the inverse).

Each block contains:
- **Claim** — what the paper section asserts
- **Numbers** — exact values + how the paper text reports them
- **Method** — the algorithm, parameters, and choices
- **Locked by** — the golden-numbers test that pins the value
- **Regenerate** — one-line command
- **Known divergence** — if the paper text disagrees with the data

---

## 0. Shared choices

Parameters and seeds that apply across the whole pipeline. Locked
implicitly because every downstream number was computed under them.

| Parameter | Value | Where set | Notes |
|---|---|---|---|
| Extraction ε (epsilon threshold) | **0.5** | `configs/default.yaml::extraction.eps` | At ε = 0.001 circuits are trivially full; at ε = 0.1 concept fractions are near-zero. Meaningful signal at 0.5. |
| Extraction cons (consistency threshold) | **0.8** | `configs/default.yaml::extraction.cons` | For both Qwen + DS, every neuron passing ε also fires on 100% of prompts → consistency is a no-op at this ε. |
| Permutation RNG seed | **42** | function-level in `permutation_within_group_p_value` and `train_concept_probe` | Locked at the function, not the script, so all callers get identical results. |
| Permutation count `n_perm` | **10 000** | `configs/paper/figure6_four_cluster.yaml` | F6 G1 p < 0.0001 reported as "p < 0.0001" in paper because the smallest resolvable p-value is 1/n_perm = 1e-4. |
| Probe CV folds | **5** | `circuits/probes.py::train_concept_probe` | Stratified `cross_val_score`; capped at the smaller class count. |
| Probe regularisation | **C = 1.0** | `circuits/probes.py` | LogisticRegression default. Lbfgs solver, max_iter=1000. |
| Reference layer for dendrograms | **L14** | `configs/paper/figure{5,5b,13}_dendrogram.yaml::layer` | Mid-network for both Qwen (28 layers) and DeepSeek (32 layers); chosen for stable concept-only sets. |
| Reference layer for F8 scatter | **L20** | `configs/paper/figure8_jaccard_cosine.yaml::focus_layer` | The layer at which Jaccard-cosine Pearson r peaks (0.645). |

**Model checkpoints:**

| Model | HF id | Layers | MLP hidden dim |
|---|---|---|---|
| Qwen | `Qwen/Qwen2.5-Coder-7B` | 28 | 3584 |
| DeepSeek | `deepseek-ai/deepseek-coder-6.7b-base` | 32 | 4096 |

**Data root:** `$ATLAS_DATA_ROOT` (default `~/Data/New-Atlas/`). See `atlas/paths.py`.

---

## §3.1 — Models

**Claim:** Two dense MLP-based code LMs (Qwen2.5-Coder-7B, DeepSeek-Coder-V1-6.7B); training differs by 2.75× data volume, layer count, hidden dim, release date.

**Numbers:**
- Qwen: 28 layers, 3584 MLP dim
- DeepSeek: 32 layers, 4096 MLP dim

**Method:** MLP hooks at every layer, last-token-position activations only. Hook pattern `model.layers.{i}.mlp` (Qwen/Llama-style).

**Code:** [`circuits/extraction.py::ActivationExtractor`](../circuits/extraction.py)

**Regenerate (GPU required):** `notebooks/1_artifact_generation/2_extraction.ipynb` (Python) + `R2_extraction.ipynb` (Rust)

---

## §3.2 — Concept Spaces

**Claim:** Concepts are "testable" if their keyword token can appear in non-structural contexts.

**Numbers:**
- **Python: 58 testable concepts** = 24 ast (`ast__Assert`, `ast__Break`, …, `ast__YieldFrom`) + 34 builtins (`builtin__sum`, `builtin__str`, …)
- **Rust: 57 testable concepts** (all `rust__*` prefix)
- Group breakdown (§5.1): Python = 6 Modular + 18 Non-modular + 34 Builtin; Rust = 6 Modular + 15 Non-modular + 36 Object

**Method:** §5.1 groups come from the `group` column of `9_results_{lang}_{model}_eps0.5_cons0.8.xlsx`. Same classification across both models.

**Locked by:** `tests/test_paper_numbers.py::test_f2_concept_group_counts[*]`

**Loaders:**
- `atlas.io.load_concept_groups(lang, model, eps=0.5, cons=0.8)` — concept → group
- `atlas.io.load_concept_aggregates(lang, model, eps, cons)` — concept → full per-row stats

---

## §3.3 — Contrastive Prompts

**Claim:** Per concept: object prompts (keyword in structural role) + matched checker prompts (keyword token outside structural role).

**Method:** Notebooks generate prompts with perplexity filter.

**Code:** `notebooks/1_artifact_generation/{1A,1B,R1A,R1B}_*.ipynb`

---

## §3.4 — Activation Extraction

**Claim:** Forward-hook MLP outputs at every layer, record activations at the last-token position.

**Method:** PyTorch forward hooks on `model.layers.{i}.mlp`. Per-prompt: tokenise with `add_special_tokens=False`, extract activations of shape `(batch, seq, mlp_dim)` at the last non-pad token.

**Code:** [`circuits/extraction.py::ActivationExtractor.extract_batch`](../circuits/extraction.py)

---

## §3.5 — Universal Circuit Computation

**Claim:** ε-threshold + consistency filter → binary masks; marginalise across the complementary dimension → universal masks A per concept per layer.

**Method:**
1. Binarise: `mask = abs(activation) > ε` (ε = 0.5)
2. Consistency filter: retain neurons firing on ≥ cons fraction of prompts (cons = 0.8; in practice a no-op at ε = 0.5)
3. Marginalise: for AST universals, intersect across all builtins. For builtin universals, intersect across all AST nodes. Each pair (ast_node, builtin_obj) contributes one row.

**Code:**
- [`circuits/binarization.py::PairRepresentationBuilder`](../circuits/binarization.py)
- [`circuits/marginalization.py::UniversalModuleComputer`](../circuits/marginalization.py)

**Regenerate:** `notebooks/1_artifact_generation/3_universals.ipynb` + `R3_universals.ipynb`

---

## §3.6 — Three-way Decomposition

**Claim:** Each concept's universal mask A and the checker mask B yield three disjoint partitions:
- `concept_only` = A \ B
- `shared` = A ∩ B
- `token_only` = B \ A

The **concept fraction** = `|concept_only| / |A|`.

**Method:** Pure set algebra.

**Code:**
- Set operation: [`atlas.analysis.decompose_sets`](../atlas/analysis/decomposition.py)
- Per-concept fraction: [`atlas.analysis.concept_fraction`](../atlas/analysis/decomposition.py)
- Convention: `concept_fraction(0, 0) = 0.0` (no detectable circuit)

**Locked by:** `tests/analysis/test_decomposition.py` (9 synthetic tests)

---

## §4.1 — Calibration

**Claim:** The atomicity super-cluster discovered in CSP-Atlas replicates in both dense models. Python: `{Assert, Break, Continue, Import, ImportFrom, Pass}` are the only `two_phase` Python concepts in Qwen. Rust: `{Super, Use}` are the only `two_phase` Rust concepts in Qwen. Python builtins have near-zero concept fractions (mean 0.024 in Qwen, 0.068 in DeepSeek).

**Numbers:** See §6.1 for the full flow-type table. Builtin means:
| Group | Qwen mean cf | DS mean cf |
|---|---|---|
| Builtin | 0.024 | 0.068 |
| Modular | 0.314 | (see §5.1) |
| Non-modular | 0.090 | (see §5.1) |

**Method:** Per-concept mean concept fraction across all layers (the `mean_cf` column of `9_results_*.xlsx`), averaged within group.

**Locked by:** `tests/test_paper_numbers.py::test_f2_concept_group_counts[*]` locks group membership; mean values are computed at experiment time by `experiments/fig2_concept_fraction_profile.py`.

---

## §4.2 — The "What" Is Conserved

**Claim:** Concept-fraction rankings correlate across two independently trained models. Spearman ρ = 0.64–0.79, all p < 10⁻⁷.

**Numbers:**

| Reported in paper | Actual value | Source |
|---|---|---|
| ρ(P) = 0.64 | **0.6384** | F1a, Python (n = 58) |
| ρ(R) = **0.72** | **0.6730** (paper abstract is wrong — see "Known divergences" below) | F1b, Rust (n = 57) |
| ρ(circuit_size, P) = 0.72 | 0.7216 | F1, secondary correlation (not in v3) |

**Method:** Per-concept mean concept-fraction at ε=0.5, cons=0.8 (Qwen vs DeepSeek). Spearman rank correlation via `scipy.stats.spearmanr`. Two-sided analytic p-values.

**Code:** [`atlas.analysis.cross_model_correlation`](../atlas/analysis/cross_model.py)

**Locked by:** `tests/test_paper_numbers.py::test_f1_concept_fraction_spearman[P|R]` (±0.005)

**Regenerate:** `python experiments/fig1_concept_scatter.py --config-name paper/figure1_concept_scatter`

**Known divergence:** The paper abstract says "Rust concept fraction ρ = 0.72". That value is the *Python circuit-size* correlation, not the Rust concept-fraction correlation. Use **0.673 (Rust)** and **0.638 (Python)** in v3.

---

## §4.3 — The "Where" Diverges

**Claim:** Qwen processes both Python and Rust at L19; DeepSeek processes both at L6–7. DeepSeek concentrates concept-specific processing 12–13 layers earlier than Qwen.

**Numbers:**
- Qwen Python peak: ~L19 (concept-fraction profile)
- Qwen Rust peak: ~L17 (slightly earlier than Python)
- DeepSeek peak: ~L6–7

**Method:** Per-(group, model) mean concept-fraction by layer; the peak layer is the argmax of the line.

**Code:** [`experiments/fig2_concept_fraction_profile.py`](../experiments/fig2_concept_fraction_profile.py) — writes `peak_per_line` to `run_info.json`.

**Loader:** `atlas.io.load_concept_sizes_by_layer(partition="concept_only"|"universal")` for the per-layer sizes; `atlas.analysis.concept_fraction` for the ratio.

**Plot:** F2.

---

## §4.4 — The "How" Diverges

**Claim:**
- **Qwen: `two_phase`.** Sharp spike at layers 2–3, collapse to near-zero, then re-explosion at L20+.
- **DeepSeek: `build_and_hold`.** Monotonic gradual growth from early layers, no collapse.

Same atomicity concepts in both models receive early-onset circuits — different temporal shapes.

**Numbers:** See §6.1 table. The §4.4 claim follows directly from the flow-type assignments at the (lang=P, model={QW,DS}) cells.

**Method:** Rule-cascade classifier on the per-layer universal-mask size for each concept. See §6.1 for the algorithm.

**Code:**
- [`atlas.analysis.classify_flow_type(sizes)`](../atlas/analysis/flow_types.py) — the rule cascade
- [`atlas.analysis.classify_all_flow_types(sizes_by_concept)`](../atlas/analysis/flow_types.py) — batch helper

**Locked by:** `tests/test_paper_numbers.py::test_f9_f12_flow_type_classifier_roundtrip[*]` verifies the in-code classifier reproduces ≥95% of the published assignments.

**Plot:** F4 (atomicity super-cluster only), F9–F12 (all concepts).

**Known caveat (not yet a paper change):** F4's Qwen panel shows the universal-mask plateau at L26–27 = 3584 (the full MLP hidden dim — every neuron fires). The agent's earlier validation PNG dropped these to zero as "saturated and meaningless". Current code plots them as-is. If §4.4 prose ends up describing the L26–27 collapse, add a `saturation_mask` flag to `plot_temporal_dynamics`.

---

## §5.1 — Rust vs Python: 2–3× More Circuitry

**Claim:** Rust's mean concept fraction is 2–3× Python's in both models. The effect is robust across architectures.

**Numbers:**
| Model | Python mean cf | Rust mean cf | Ratio |
|---|---|---|---|
| Qwen | 0.074 | 0.215 | **2.91×** |
| DeepSeek | 0.097 | 0.201 | **2.07×** |

Tier breakdown (Python, Qwen):
| Group | Mean cf |
|---|---|
| Modular | **0.314** |
| Non-modular | **0.090** |
| Builtin | **0.024** |

Ratio between extremes: **13:1**.

In Rust, all three groups (Modular / Non-modular / Object) fall within 1.5:1 of each other — "Rust's stricter syntax makes all concepts structurally distinctive".

**Method:** Average of per-concept `mean_cf` from `9_results_*.xlsx` within each group.

**Code:** [`atlas.io.load_concept_aggregates`](../atlas/io/xlsx.py) → group by the `group` column → mean of `mean_cf`.

**Plot:** F2 (per-layer profile by group × model).

---

## §5.2 — Boost Is Largest for Python's Weakest Concepts

**Claim:** Within Qwen, equivalent concepts that are weak in Python (Return, For, If — token-ambiguous, syntactically flexible) show the largest Rust boost. Import/Use is the only reversal (Python's `import` has uniquely distinctive syntax).

**Numbers:**
- Return: 7.2×
- For: 5.1×
- If: 4.4×

**Method:** Per-equivalent-pair ratio of Rust `mean_cf` over Python `mean_cf` at (Qwen, ε=0.5, cons=0.8). The 9 paper-cited equivalence pairs are in [`atlas.analysis.EQUIVALENCE_CLASSES`](../atlas/analysis/cross_language.py).

**Locked by:** Indirectly, via `test_f2_concept_group_counts` (group means are stable).

---

## §5.3 — Cross-Language Neuron Sharing

**Claim:** DeepSeek shares 2.3× more neurons between Python and Rust than Qwen, with all 7 of 7 equivalence classes passing the 10 % sharing threshold.

**Numbers:**

| Reported in paper | Actual value | Notes |
|---|---|---|
| "2.3×" | **1.949×** | Mean-of-means DS / mean-of-means QW. Paper is wrong by ~15% — see "Known divergences". |
| 7/7 classes ≥ 10% | 7/7 for DS; 6/7 for QW (Function def at 0.038 fails) | Paper claim about 7/7 holds for DS only. |

**Per-class (model, equivalence_class) mean sharing fractions:**

| Class | DS | QW |
|---|---|---|
| Branching | 0.517 | 0.270 |
| Function def | 0.149 | **0.038** (fails 10%) |
| Iteration | 0.525 | 0.340 |
| Loop control | 0.330 | 0.289 |
| Module import | 0.250 | 0.129 |
| Return | 0.503 | 0.177 |
| Type def | 0.434 | 0.146 |

**Method:**
1. For each equivalence class (e.g. Iteration = {Python: For, While; Rust: For, Loop, While}), pool concept-only neurons across language-specific members.
2. Sharing fraction = `|py_pool ∩ rs_pool| / min(|py_pool|, |rs_pool|)` per layer.
3. Bar is the mean across all layers in the data file (n = 32 per cell).
4. DS/QW ratio = mean-of-means DS / mean-of-means QW.

**Code:**
- [`atlas.analysis.pool_neurons`](../atlas/analysis/cross_language.py)
- [`atlas.analysis.cross_language_sharing_fraction`](../atlas/analysis/cross_language.py)
- [`atlas.analysis.EQUIVALENCE_CLASSES`](../atlas/analysis/cross_language.py) — the 9-class default mapping

**Locked by:** `tests/test_paper_numbers.py::test_f3_cross_language_sharing_ratio` locks ratio = 1.949× ± 0.005 and the 10% threshold per cell.

**Regenerate:** `python experiments/fig3_cross_language_sharing.py --config-name paper/figure3_cross_language_sharing`

**Plot:** F3.

**Known divergence:** Paper §5.3 says "2.3×". Recomputation gives 1.949× under simple mean-of-means. Use **1.94×** in v3 (or recompute with a defended alternative aggregation and update the locked test).

---

## §6.1 — The Atomicity Super-Cluster

**Claim:**
- In Qwen, the Python `two_phase` concepts are `{Assert, Break, Continue, Import, ImportFrom, Pass, Starred}` — the atomicity super-cluster from CSP-Atlas.
- In DeepSeek, the same functional group appears as `build_and_hold`.
- Flow-type agreement between models is 88 % for Python, 84 % for Rust.

**Numbers — flow-type distribution per (lang, model) cell:**

| Cell | two_phase | build_and_hold | late_emergence | unclassified | Total |
|---|---|---|---|---|---|
| P × QW | **7** | — | 95 | 4 | 106 |
| R × QW | **2** | — | 71 | 2 | 75 |
| P × DS | 1 | **7** | 89 | 9 | 106 |
| R × DS | — | **5** | 63 | 7 | 75 |

(106 / 75 totals include concepts with no testable signal at L14 — filtered out by the neuron-list XLSX but counted in the flow-type assignments.)

**Method:** Rule cascade on per-layer universal-mask size (`load_concept_sizes_by_layer(partition="universal")`):

1. `empty` if max = 0
2. `flash` if peak > 5× mean AND high-plateau width ≤ 3 (no concept matches in practice)
3. `late_emergence` if first half max < 10% of overall max AND peak past midpoint
4. `two_phase` if ≥ 2 local maxima above 30% of max, with a trough between them dropping below 50% of the smaller peak, peaks ≥ 3 layers apart
5. `build_and_hold` if longest non-decreasing run ≥ 4 AND plateau width ≥ n/3
6. otherwise `unclassified`

First-match-wins ordering. Thresholds are calibrated to reproduce the counts above and are exposed as module constants in `flow_types.py`.

**Code:** [`atlas.analysis.classify_flow_type`](../atlas/analysis/flow_types.py)

**Locked by:**
- `tests/test_paper_numbers.py::test_f9_f12_flow_type_counts[*]` — locks the count distribution per cell
- `tests/test_paper_numbers.py::test_f9_f12_flow_type_classifier_roundtrip[*]` — re-classifies size curves and verifies ≥ 95 % agreement with the loaded XLSX

**Plot:** F9–F12.

---

## §6.2 — Semantic Clustering (Four-Cluster Claim)

**Claim:** With 57 Rust concepts at Qwen × L14, hierarchical clustering of concept-only neuron sets produces four semantically coherent groups:
1. Type-system traits: Enum, Send, Option, Iterator, Copy, Eq, Drop, Debug, ToString
2. Memory/ownership: Pin, Mutex, Arc, Box, Vec, Impl, Async, Move, Future
3. Data definition: Struct, Let, String, Trait, Default, Display, Hash, Return, Read
4. Control-flow/module: Fn, Pub, Use, Crate, While, Break, Loop, If, Match

**Refined finding (locked in tests, reflected in CLAUDE.md but not yet in paper text):**

| Group | Observed mean Jaccard | Null mean | p-value | Verdict |
|---|---|---|---|---|
| G1 type-system traits | **0.535** | 0.112 | **< 0.001** | strongly significant |
| G2 memory/ownership | 0.190 | 0.112 | 0.044 | marginal (just below 0.05) |
| G3 data definition | 0.199 | 0.112 | 0.035 | marginal |
| G4 control-flow/module | 0.129 | 0.112 | 0.292 | indistinguishable from random |

So "four equal clusters" → **"1 strong + 2 marginal + 1 random"**.

**Method:**
1. Compute pairwise Jaccard between concept-only sets at (R, QW, L14, eps=0.5, cons=0.8). Universe = all 57 Rust concepts including ones empty at L14.
2. Ward linkage on (1 − Jaccard) distance.
3. For each hypothesised group, mean within-group Jaccard.
4. Permutation null: 10 000 random same-size draws from the full concept universe (seed = 42).
5. p-value = fraction of nulls ≥ observed.

**Code:**
- [`atlas.analysis.pairwise_jaccard_matrix`](../atlas/analysis/jaccard.py)
- [`atlas.analysis.ward_linkage_from_jaccard`](../atlas/analysis/meta_circuits.py)
- [`atlas.analysis.cut_dendrogram_at_k_clusters`](../atlas/analysis/meta_circuits.py) — original four-cluster discovery
- [`atlas.analysis.permutation_within_group_p_value`](../atlas/analysis/meta_circuits.py) — F6 validation

**Locked by:**
- `tests/test_paper_numbers.py::test_f6_g1_trait_family_observed_jaccard` (0.535 ± 0.001)
- `tests/test_paper_numbers.py::test_f6_g1_permutation_p_value` (p < 0.001)
- `tests/test_paper_numbers.py::test_f6_four_cluster_observed_and_p_value[G1|G2|G3|G4]` (all four locked)

**Regenerate:** `python experiments/fig6_four_cluster_test.py --config-name paper/figure6_four_cluster`

**Plots:** F5a (R × QW dendrogram), F5b (R × DS, two-cluster), F6 (group cohesion bar chart), F13 (P × QW dendrogram, appendix E).

**Known divergence:** The original "four equal clusters" framing is too strong. §6.2 must be rewritten in v3 to reflect the refined finding — only G1 is robust, G2/G3 marginal at α = 0.05, G4 random.

---

## §6.3 — Meta-Circuit Structure

**Claim:** DeepSeek produces 3× more statistically significant meta-circuit structure than Qwen. Rust meta-circuits are broader (5–7 significant layers L7–L20) than Python (0–7 significant layers).

**Numbers:** Per-(model, group, layer) significant-vs-random counts.

**Method:** Per-layer permutation test across multiple hypothesised concept groups. The aggregated table is in `7_E3_meta_circuit_results.xlsx`; the per-layer test uses `permutation_within_group_p_value`.

**Code:**
- [`atlas.analysis.permutation_within_group_p_value`](../atlas/analysis/meta_circuits.py) — per-layer significance
- The aggregation was done in [`notebooks/2_analysis/7_E3_meta_circuits.ipynb`](../notebooks/2_analysis/7_E3_meta_circuits.ipynb) — not yet a first-class atlas function. Promotion candidate.

---

## §7.1 — Causal Validation: Double Dissociation

**Claim:** Per-layer zero-ablation of concept-only neurons on Qwen × Python yields a clean double dissociation for **4 of 5 tested keyword concepts**: Import, Try, While, Assert (PASS). One documented negative: Break (FAIL — random-ablation null produces a more negative dissociation, −0.066, than concept-only ablation, −0.032). Concepts with concept-only sets below ~20 neurons (Lambda n=9, len n=2, If n=15) produce no measurable causal effect.

**Method per (concept, layer):**
1. Forward-pass on concept prompts under no ablation → baseline log-prob of target token.
2. Zero out concept-only neurons at the target layer via [`circuits.ablation.AblationHook`](../circuits/ablation.py). Forward-pass again → record Δ-log-prob on concept prompts.
3. Repeat for checker prompts → Δ-log-prob on checker prompts.
4. Repeat (2) and (3) with a size-matched **random-null** ablation drawn from neurons outside the concept-only, shared, and token-only partitions.
5. Compute dissociation score: `(Δ_concept_co − Δ_checker_co) − (Δ_concept_null − Δ_checker_null)`. Passes if concept-only side is more negative than the null side.

**Code:**
- [`circuits.ablation.AblationHook`](../circuits/ablation.py) — the forward-hook context manager (zeroes neurons at one MLP layer)
- [`atlas.analysis.compute_dissociation_score`](../atlas/analysis/dissociation.py) — the four-Δ comparison, returns `{co_dissociation, null_dissociation, margin, passes}`

**Tests:**
- `tests/test_ablation.py` — 5 tests on a toy nn.Module (no real model needed)
- `tests/analysis/test_dissociation.py` — 5 tests including the documented Break-fails case

**Notebooks (full pipeline):** [`notebooks/2_analysis/6_ablation.ipynb`](../notebooks/2_analysis/6_ablation.ipynb), [`notebooks/2_analysis/6b_ablation_double.ipynb`](../notebooks/2_analysis/6b_ablation_double.ipynb).

**Plot:** Original figure 6 (06_double_dissociation.png) — not yet migrated to `experiments/`. Promotion candidate.

---

## §7.2 — Geometric Validation: Linear Probes

**Claim:** Per-concept, per-layer logistic-regression probes (concept prompts vs checker prompts) on Qwen × Python achieve **97–100 % classification accuracy at every one of the 28 layers**, for every concept tested. The concept-versus-token distinction is linearly encoded from the earliest residual streams onward.

**Numbers:**
- Probe accuracy band: **0.976 (L27) to 0.997 (L3)**
- 24 ast concepts tested (no builtins).

**Method:**
1. For each concept × layer: load activations from `{lang}_{model}_V1_vectors.h5` (object/concept and object/checker groups).
2. StandardScaler → LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs').
3. 5-fold stratified cross-validation → accuracy.
4. Refit on all data, L2-normalise weights → release direction vector.

**Code:**
- [`circuits.probes.train_concept_probe`](../circuits/probes.py) — the sklearn wrapper
- [`atlas.io.load_probe_results`](../atlas/io/probe.py) — read pre-computed per-(concept, layer) accuracies

**Locked by:** `tests/test_paper_numbers.py::test_f7_probe_accuracy_band` (min ≥ 0.97, max ≤ 1.00).

**Regenerate (plot):** `python experiments/fig7_probe_accuracy.py --config-name paper/figure7_probe_accuracy`

**Regenerate (probes):** `notebooks/2_analysis/V2_probes.ipynb` — needs activations from V1.

**Plot:** F7.

---

## §7.3 — Jaccard–Cosine Cross-Validation

**Claim:** Pairwise probe-direction cosine similarity correlates with pairwise concept-only Jaccard similarity. **Pearson r peaks at 0.645 at L20**. The correlation is moderate (not unity) — the two views agree on coarse structure but diverge in low-amplitude tails.

**Numbers:**
- Peak r = 0.645 at L20
- 276 concept pairs (24 ast concepts)

**Method:** For each layer:
1. Load concept-only sets and probe weight vectors at that layer for all 24 ast concepts.
2. L2-normalise probe weights.
3. For each of the 276 pairs: Jaccard(set_A, set_B) and cosine(normalised_w_A, normalised_w_B).
4. Pearson r between the 276-element jaccard and cosine arrays.

**Code:** [`atlas.analysis.pairwise_cosine_vs_jaccard(weights, masks)`](../atlas/analysis/probe_alignment.py) → `(jaccards, cosines, pearson_r)`.

**Loaders:** [`atlas.io.load_probe_weights(lang, model, layer)`](../atlas/io/probe.py), [`atlas.io.load_neuron_lists(layer=L)`](../atlas/io/xlsx.py).

**Locked by:** `tests/test_paper_numbers.py::test_f8_peak_jaccard_cosine_correlation` (peak layer = 20, r = 0.645 ± 0.005).

**Regenerate:** `python experiments/fig8_jaccard_cosine.py --config-name paper/figure8_jaccard_cosine`

**Plot:** F8 (left = r per layer; right = L20 scatter).

---

## Appendix E — Python Dendrogram

**Claim:** For Python (Qwen, L14), the earliest merges in the dendrogram are (Assert, Pass) and (Continue, Break), confirming the atomicity grouping. Control-flow sub-clusters (Try, While, With) and (For, AsyncFor) emerge next. Builtins form a separate "poverty cluster" with high pairwise Jaccard driven by tiny concept-only sets.

**Code path:** Identical to F5 — same `pairwise_jaccard_matrix` + `ward_linkage_from_jaccard` + `plot_dendrogram` pipeline, different cell (P, QW, L14).

**Regenerate:** `python experiments/fig13_python_dendrogram.py --config-name paper/figure13_dendrogram`

**Plot:** F13.

---

## Figure cross-reference

| Figure | Paper § | Script | Config |
|---|---|---|---|
| F1 a/b | §4.2 | `fig1_concept_scatter.py` | `figure1_concept_scatter.yaml` |
| F2 | §4.3, §5.1 | `fig2_concept_fraction_profile.py` | `figure2_concept_fraction.yaml` |
| F3 | §5.3 | `fig3_cross_language_sharing.py` | `figure3_cross_language_sharing.yaml` |
| F4 | §4.4 | `fig4_temporal_dynamics.py` | `figure4_temporal_dynamics.yaml` |
| F5a | §6.2 | `fig5_rust_dendrogram.py` | `figure5_dendrogram.yaml` |
| F5b | §6.2 | `fig5_rust_dendrogram.py` | `figure5b_dendrogram.yaml` |
| F6 | §6.2 | `fig6_four_cluster_test.py` | `figure6_four_cluster.yaml` |
| F7 | §7.2 | `fig7_probe_accuracy.py` | `figure7_probe_accuracy.yaml` |
| F8 | §7.3 | `fig8_jaccard_cosine.py` | `figure8_jaccard_cosine.yaml` |
| F9 | §6.1 | `fig_circuit_size_by_flow_type.py` | `figure9_p_qw.yaml` |
| F10 | §6.1 | `fig_circuit_size_by_flow_type.py` | `figure10_r_qw.yaml` |
| F11 | §6.1 | `fig_circuit_size_by_flow_type.py` | `figure11_p_ds.yaml` |
| F12 | §6.1 | `fig_circuit_size_by_flow_type.py` | `figure12_r_ds.yaml` |
| F13 | App. E | `fig13_python_dendrogram.py` | `figure13_dendrogram.yaml` |

---

## Known divergences — consolidated list

Three places where the published v1/v2 paper text disagrees with the data. Two are locked in tests; one requires paper-text rewriting in v3.

| # | Section | Paper says | Data gives | Status |
|---|---|---|---|---|
| 1 | Abstract / §4.2 | ρ(Rust concept fraction) = 0.72 | ρ = **0.673** (P = 0.638) | Locked: `test_f1_concept_fraction_spearman[R]`. Use 0.673 in v3. |
| 2 | §5.3 | DS shares 2.3× more neurons than QW | Mean-of-means ratio = **1.949×** | Locked: `test_f3_cross_language_sharing_ratio`. Use 1.94× in v3 (or defend an alternative aggregation). |
| 3 | §6.2 | Four equal semantically coherent clusters | **G1 strong (p < 0.001), G2 marginal (p = 0.044), G3 marginal (p = 0.035), G4 random (p = 0.292)** | Locked: `test_f6_four_cluster_observed_and_p_value[*]`. §6.2 prose must be rewritten in v3 to reflect "1 strong + 2 marginal + 1 random". |

---

## Reproducibility recipe

For a reviewer reading this paper and wanting to reproduce a number:

```bash
git clone https://github.com/piotrwilam/New-Atlas
cd New-Atlas
git checkout v0.1.0           # the refactor-complete state
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Point at the data mirror (12 GB, HF Hub URL TBD).
export ATLAS_DATA_ROOT=~/Data/New-Atlas

# Verify every paper number at once (~3 s):
pytest tests/test_paper_numbers.py

# Reproduce any single figure:
python experiments/fig5_rust_dendrogram.py --config-name paper/figure5_dendrogram
# → results/atlas2x2_figure5_..._<timestamp>/figure5_rust_dendrogram_qwen.png
```

For re-running §7 validation (heavier — needs torch + sklearn):

```bash
uv pip install -e ".[extraction]"
pytest tests/test_ablation.py tests/test_probes.py
```

For re-extracting activations from a model (GPU required, ~30–60 min per cell):

See [`notebooks/1_artifact_generation/README.md`](../notebooks/1_artifact_generation/README.md).
