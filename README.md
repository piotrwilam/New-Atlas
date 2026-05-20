# New-Atlas

Working repository for the **Atlas2x2** paper — *A Cross-Language Circuit Atlas for Code Models: The What / Where / How Dissociation in Syntactic Representation* (Wilam, 2026).

Cross-model, cross-language mechanistic interpretability of syntactic concept circuits in **Qwen2.5-Coder-7B** (28 layers) and **DeepSeek-Coder-V1-6.7B** (32 layers), over Python and Rust — a complete 2 × 2 matrix.

The public release (planned, Apache-2.0) will live at [`piotrwilam/Atlas2x2`](https://github.com/piotrwilam/Atlas2x2). This repo (`New-Atlas`) is the development mirror with full notebook history.

---

## Reading guide

The codebase is organised in three concentric layers. Pick the layer that matches what you want to understand:

### Layer 1 — How the artifacts were generated  (paper §3)

The extraction pipeline that produced the frozen XLSX / HDF5 / CSV / NPZ files everything else reads.

| Read | Then run | Paper section |
|---|---|---|
| [`circuits/extraction.py`](circuits/extraction.py) — `ActivationExtractor`        | [`notebooks/1_artifact_generation/2_extraction.ipynb`](notebooks/1_artifact_generation/2_extraction.ipynb) (GPU) | §3.1–§3.4 |
| [`circuits/marginalization.py`](circuits/marginalization.py) — `UniversalModuleComputer` | [`notebooks/1_artifact_generation/3_universals.ipynb`](notebooks/1_artifact_generation/3_universals.ipynb)       | §3.5      |
| [`circuits/binarization.py`](circuits/binarization.py) — pair representations    | [`notebooks/1_artifact_generation/4_neuron_list.ipynb`](notebooks/1_artifact_generation/4_neuron_list.ipynb)     | §3.6      |
| [`circuits/pipeline.py`](circuits/pipeline.py) — end-to-end orchestrator         | (used by 2_extraction.ipynb)                              | §3        |

See [`notebooks/1_artifact_generation/README.md`](notebooks/1_artifact_generation/README.md) for execution order, Rust variants (`R*`), and the probe-vector side-pipeline (`V1_*`).

### Layer 2 — The analysis tools  (paper §3.7+, §6, §7)

Domain-independent instruments: loaders, statistics, plotting primitives. 19 public functions in three sub-packages.

| Read | What it does | Paper section |
|---|---|---|
| [`atlas/io/xlsx.py`](atlas/io/xlsx.py)                  | 6 loaders for decomposition + aggregate stage XLSX files | §3.5–§3.6 |
| [`atlas/io/probe.py`](atlas/io/probe.py)                | 3 loaders for probe CSV / NPZ artifacts                  | §7.2–§7.3 |
| [`atlas/analysis/jaccard.py`](atlas/analysis/jaccard.py)         | Jaccard set similarity + pairwise matrices               | §3.7      |
| [`atlas/analysis/meta_circuits.py`](atlas/analysis/meta_circuits.py)   | Ward linkage + permutation-test for group cohesion       | §6        |
| [`atlas/plotting/`](atlas/plotting/)                | 5 rendering primitives (dendrogram, group-coherence bars, temporal-dynamics lines, circuit-size overlay, style preset) | — |

### Layer 3 — What each figure shows  (paper §4–§7)

One script + one Hydra config per paper figure. Each is a 3-step pipeline: load → analyze → plot.

| Figure | Script | Config | Paper section |
|---|---|---|---|
| F1  Concept fraction QW vs DS scatter | [`experiments/fig1_concept_scatter.py`](experiments/fig1_concept_scatter.py)              | [`configs/paper/figure1_concept_scatter.yaml`](configs/paper/figure1_concept_scatter.yaml)               | §4.2  |
| F2  Concept fraction profile by layer | [`experiments/fig2_concept_fraction_profile.py`](experiments/fig2_concept_fraction_profile.py)     | [`configs/paper/figure2_concept_fraction.yaml`](configs/paper/figure2_concept_fraction.yaml)             | §4.3  |
| F3  Cross-language sharing            | [`experiments/fig3_cross_language_sharing.py`](experiments/fig3_cross_language_sharing.py)       | [`configs/paper/figure3_cross_language_sharing.yaml`](configs/paper/figure3_cross_language_sharing.yaml) | §5.3  |
| F4  Atomicity temporal dynamics       | [`experiments/fig4_temporal_dynamics.py`](experiments/fig4_temporal_dynamics.py)            | [`configs/paper/figure4_temporal_dynamics.yaml`](configs/paper/figure4_temporal_dynamics.yaml)           | §4.4  |
| F5a Rust dendrogram × Qwen            | [`experiments/fig5_rust_dendrogram.py`](experiments/fig5_rust_dendrogram.py)              | [`configs/paper/figure5_dendrogram.yaml`](configs/paper/figure5_dendrogram.yaml)                         | §6.2  |
| F5b Rust dendrogram × DeepSeek        | (same script)                                       | [`configs/paper/figure5b_dendrogram.yaml`](configs/paper/figure5b_dendrogram.yaml)                       | §6.2  |
| F6  Four-cluster validation           | [`experiments/fig6_four_cluster_test.py`](experiments/fig6_four_cluster_test.py)            | [`configs/paper/figure6_four_cluster.yaml`](configs/paper/figure6_four_cluster.yaml)                     | §6.2  |
| F7  Probe accuracy by layer           | [`experiments/fig7_probe_accuracy.py`](experiments/fig7_probe_accuracy.py)               | [`configs/paper/figure7_probe_accuracy.yaml`](configs/paper/figure7_probe_accuracy.yaml)                 | §7.2  |
| F8  Jaccard–cosine cross-validation   | [`experiments/fig8_jaccard_cosine.py`](experiments/fig8_jaccard_cosine.py)               | [`configs/paper/figure8_jaccard_cosine.yaml`](configs/paper/figure8_jaccard_cosine.yaml)                 | §7.3  |
| F9–F12 Circuit size × 4 cells         | [`experiments/fig_circuit_size_by_flow_type.py`](experiments/fig_circuit_size_by_flow_type.py)     | `configs/paper/figure{9_p_qw,10_r_qw,11_p_ds,12_r_ds}.yaml` | §6.1 |
| F13 Python dendrogram (appendix E)    | [`experiments/fig13_python_dendrogram.py`](experiments/fig13_python_dendrogram.py)           | [`configs/paper/figure13_dendrogram.yaml`](configs/paper/figure13_dendrogram.yaml)                       | App. E |

Every figure writes a timestamped run directory under `results/` containing the PNG, the resolved Hydra config, a `run_info.json` provenance file, and the Hydra log.

---

## Reproducing the paper figures

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Point at the data mirror (frozen artifacts; ~12 GB).
export ATLAS_DATA_ROOT=~/Data/New-Atlas

# Reproduce one figure.
python experiments/fig5_rust_dendrogram.py --config-name paper/figure5_dendrogram

# Reproduce all golden-number claims in the paper (~1 s).
pytest tests/test_paper_numbers.py
```

The 39 tests under [`tests/`](tests/) lock every numeric claim in the paper to ±0.005, including:
- ρ = 0.638 / 0.673 (Python / Rust cross-model concept fraction)
- DS/QW cross-language sharing ratio = 1.949×
- F6 group p-values (G1 < 0.001, G2 = 0.044, G3 = 0.035, G4 = 0.292)
- F7 probe accuracy band [0.976, 0.997]
- F8 peak Jaccard-cosine r = 0.645 at L20
- F9–F12 flow-type counts per cell

## Project structure

```
New-Atlas/
├── circuits/             ← Layer 1: extraction pipeline (frozen, GPU-required to rerun)
├── atlas/                ← Layer 2: io / analysis / plotting tool library
├── experiments/          ← Layer 3: one entry-point script per paper figure
├── configs/              ← Hydra configs (one per figure, plus shared defaults)
├── tests/                ← unit tests + golden-numbers paper-claim tests
├── notebooks/
│   ├── 1_artifact_generation/   ← notebooks that produce frozen artifacts
│   └── 2_analysis/              ← exploratory analysis notebooks
├── docs/                 ← misc design notes
├── results/              ← (gitignored) per-run outputs from experiments/
├── pyproject.toml        ← setuptools + dependency declaration
├── .python-version       ← Python 3.12 (uv-managed)
├── CLAUDE.md             ← internal project notes
└── coding_guidelines.md  ← code style + workflow contract
```

## Data

The 12 GB of frozen artifacts (activations, masks, neuron lists, aggregated XLSX, probe outputs) live in three mirrors:

| Tier                                | Location                                      | Use                                  |
|---|---|---|
| **Hugging Face Hub**                | (URL TBD on public release)                   | Canonical for the public `Atlas2x2` repo |
| **Google Drive**                    | `gdrive_innest:DATA/New-Atlas/`              | Working storage written by Colab     |
| **Local mirror**                    | `~/Data/New-Atlas/`                          | What `ATLAS_DATA_ROOT` defaults to   |

The artifacts are **frozen** — re-extracting them requires running the Layer 1 notebooks on GPU and reproducing the seed regime documented in `CLAUDE.md`. The golden-numbers tests guard against any silent drift.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
