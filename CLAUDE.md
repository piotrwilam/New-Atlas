# CLAUDE.md — Atlas2x2 project notes

> Read this first in every session.

## What this is

The codebase for the **Atlas2x2** paper: *A Cross-Language Circuit Atlas for Code Models — The What/Where/How Dissociation in Syntactic Representation* (Wilam, 2026). A cross-model, cross-language mechanistic-interpretability study of syntactic concept circuits in Qwen2.5-Coder-7B and DeepSeek-Coder-V1-6.7B over Python and Rust.

- **Repo (this one, personal/working):** `piotrwilam/New-Atlas`
- **Public release repo (planned):** `piotrwilam/Atlas2x2` — to be created at publication
- **License (eventual):** Apache-2.0

## Pre-refactor archive

The pre-refactor state is frozen at the git tag **`archive-pre-refactor`** (created 2026-05-19). If anything goes wrong during the refactor, recover from there.

## Coding standards

Follow `coding_guidelines.md` at the repo root. **Read it at the start of every session.** Highlights:
- Build phase vs refactor phase — substantive functions go into the `atlas/` package once a stage is "done".
- Hydra configs, structured paths in `atlas/paths.py`, no hardcoded numbers or paths.
- pytest mirrored layout (`atlas/x/y.py` → `tests/x/test_y.py`).
- One script per figure in `experiments/`, plotting style centralised via `apply_style()`.
- Refactor-on-branch: any change > 50 lines / > 2 files lives on a feature branch and merges squashed.

## Project-specific quirks

- **Default extraction parameters: ε = 0.5, consistency = 0.8.** All paper-shipped results use this setting unless explicitly stated. Other (ε, cons) combinations exist in the data but are exploratory.
- **Two models, two languages, four cells:**
  - P_QW = Python × Qwen2.5-Coder-7B (28 layers)
  - R_QW = Rust × Qwen2.5-Coder-7B (28 layers)
  - P_DS = Python × DeepSeek-Coder-V1-6.7B (32 layers)
  - R_DS = Rust × DeepSeek-Coder-V1-6.7B (32 layers)
- **Concept-only / shared / token-only decomposition.** A *universal mask* A vs a *checker mask* B yields three disjoint partitions: concept-only (A \ B), shared (A ∩ B), token-only (B \ A). "Concept fraction" = |A \ B| / |A|.
- **Run naming convention:** `atlas2x2_{stage}_{variant}_{date}` (e.g. `atlas2x2_extraction_qw_eps05_260519`). Used for run IDs, output dirs, W&B names, artifact filenames.
- **Stages (in pipeline order):** `extraction` → `binarisation` → `decomposition` → `analysis` / `probes` → `plotting`.

## Data locations

The 11.8 GB of experimental artifacts (activations, masks, neuron lists, aggregated CSVs/XLSX) live in three mirrors:

| Tier | Location | Use |
|---|---|---|
| **Canonical for public release (TBD)** | Hugging Face Hub — dataset URL to be filled in here | What the README points reviewers/readers at |
| **Working storage** | Google Drive `gdrive_innest:DATA/New-Atlas/` | Where Colab experiments write |
| **Local working mirror** | `/Users/piotrwilam/Data/New-Atlas/` | What `atlas/paths.py` points `DATA_ROOT` at on this Mac |

`atlas/paths.py` must read `DATA_ROOT` from env / config, defaulting to the local mirror. Same code → works on Mac, Colab, and CI.

## Refactor status (Phase 2, top-down)

- **Frozen** (read-only inputs): extraction, binarisation, decomposition stages. Output artifacts already on Drive. Do NOT regenerate without explicit reason.
- **Refactored + rerun from frozen artifacts:** analysis, plotting.
- **Proof-of-concept figure:** Figure 5 (Rust dendrogram, Qwen at L14). Touches neuron-list loading, Jaccard, hierarchical clustering, dendrogram plotting — the most modules in one figure. Once it works end-to-end through the new package, the pattern clones to the other figures.

## Known issues to fix during refactor

1. **Two numeric discrepancies between PDF text and the source data** (surfaced in `../Papers/Atlas_v2_recovery/atlas_v2.md` validation section):
   - Paper abstract: "Rust concept fraction ρ = 0.72" — actual ρ = 0.673 (Spearman across 57 Rust concepts). The 0.72 figure refers to Python *circuit size*, not Rust concept fraction.
   - Paper §5.3: "DeepSeek shares 2.3× more neurons" — actual ratio from simple mean-of-means is **1.94×**. Aggregation method may differ; needs reconciling before v3.
2. **F6 four-cluster claim is overstated.** Only G1 (type-system traits: Enum/Send/Option/Iterator/Copy/Eq/Drop/Debug/ToString) is statistically significant (p < 0.0001 vs 10k permutations). G3 borderline (p = 0.038, but bleeds into G1). G2 fails (p = 0.052). G4 indistinguishable from random (p = 0.296). §6 reframes around trait-family recovery alone.
3. **§4.4 "How" axis prose is partly wrong.** DeepSeek's atomicity-concept dynamic is "smooth monotonic late-onset growth", not "monotonic growth from early layers" as the paper claims. Early-bias ratios for the atomicity concepts in DeepSeek are 0.05–0.11 (mostly late activity), versus 0.73–0.90 in Qwen. F4 (a new temporal plot in `original_assets/validation/`) is sharp enough to carry §4.4, but the prose needs adjusting.

## Reproducibility guarantees

- **Frozen-numbers test** (`tests/test_paper_numbers.py`) will lock every numeric claim in the paper to the digits reported. Any future drift > 0.001 fails loudly.
- **Pin model revision SHAs** (not just `Qwen2.5-Coder-7B`) in the Hydra config for any re-extraction.
- **All permutation tests seed `random.seed(42)`** at the function level — not in the script — so p-values stay identical across reruns.

## Don't

- Don't regenerate frozen artifacts during the refactor unless explicitly asked.
- Don't commit large data files to git. Drive / HF Hub for data; git for code only.
- Don't add `utils/` or `helpers/` directories. Functions go in named pipeline-stage modules.
- Don't put logic in notebooks once a stage is done (see `coding_guidelines.md` notebook contract).

## See also

- `coding_guidelines.md` — the full style and protocol doc.
- `../Papers/Atlas_v2_recovery/atlas_v2.md` — full PDF reverse-engineering + F4/F6 validation results.
- `../Papers/Atlas_v3/` — draft markdown for the next paper revision.
