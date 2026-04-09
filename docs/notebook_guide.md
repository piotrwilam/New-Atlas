# Notebook Guide

21 notebooks across 10 stages. Python notebooks have numeric names, Rust notebooks have `R` prefix.

All data files use `{LANG}_{MODEL}_` prefix (e.g., `P_QW_`, `R_DS_`). Each notebook sets `LANG`, `MODEL`, `PREFIX` at the top.

## Model Configuration

All notebooks share a `MODEL_CONFIGS` dict. To add a new model, add one entry:

```python
MODEL_CONFIGS = {
    "QW": {"id": "Qwen/Qwen2.5-Coder-7B",                "n_layers": 28, "mlp_dim": 3584},
    "DS": {"id": "deepseek-ai/deepseek-coder-6.7b-base",  "n_layers": 32, "mlp_dim": 4096},
}
```

`N_LAYERS` and `MLP_DIM` are derived from this dict. No hardcoded dimensions.

## Adding a New Model (e.g., DeepSeek)

1. Run `0_tokeniser_validation` to check keyword tokenization
2. Reuse existing prompts (copy with new prefix):
   ```
   cp P_QW_1A_object_prompts.parquet P_DS_1A_object_prompts.parquet
   cp P_QW_1B_checker_prompts.parquet P_DS_1B_checker_prompts.parquet
   cp R_QW_1A_object_prompts.parquet R_DS_1A_object_prompts.parquet
   cp R_QW_1B_checker_prompts.parquet R_DS_1B_checker_prompts.parquet
   ```
3. Set `MODEL = "DS"` in notebooks 2-4 (and R2-R4) and run the pipeline
4. Rerun experiment notebooks 7_E3, 7_E6, 7_E7 (they load all combos automatically)
5. Run 10_E8 for cross-model comparison

## Stage 0 — Tokeniser Validation

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `0_tokeniser_validation` | No | Verify keywords are single tokens per model | Tokenizers only | Validation report (display) |

## Stage 1 — Prompt Generation (language-specific, model-independent except perplexity filter)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `1A_object_prompts` | Yes | Python (AST, builtin) prompt pairs with perplexity filter | Templates + Qwen | `{P}_1A_object_prompts.parquet` |
| `1B_checker_prompts` | No | Python token-without-concept prompts | Templates + tokenizer | `{P}_1B_checker_prompts.parquet` |
| `R1A_object_prompts` | Yes | Rust (construct, object) prompt pairs with perplexity filter | Templates + Qwen | `{P}_1A_object_prompts.parquet` |
| `R1B_checker_prompts` | No | Rust token-without-concept prompts | Templates + tokenizer | `{P}_1B_checker_prompts.parquet` |

Parquet schemas:
- 1A/R1A: `ast_node`/`construct`, `builtin_obj`/`object`, `variation_id`, `prompt_text`, `sequence_loss`, `token_length`, `ast_verified`/`tree_sitter_verified`
- 1B/R1B: `object`, `keyword`, `variation_id`, `prompt_text`

## Stage 2 — Activation Extraction (per language x model)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `2_extraction` | Yes | Extract raw MLP activations from Python prompts | `{P}_1A_*.parquet`, `{P}_1B_*.parquet` | `{P}_2_object_activations.h5`, `{P}_2_checker_activations.h5` |
| `R2_extraction` | Yes | Extract raw MLP activations from Rust prompts | `{P}_1A_*.parquet`, `{P}_1B_*.parquet` | `{P}_2_object_activations.h5`, `{P}_2_checker_activations.h5` |

HDF5 schema: `/activations/layer_{L}/{key}` (float32), `/firing_counts/layer_{L}/{key}` (int32), `/n_prompts/{key}` (scalar)

## Stage 3 — Threshold Sweep + Marginalization (per language x model)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `3_universals` | No | 9-combo threshold sweep, universal circuit intersection | `{P}_2_*.h5` | 9x `{P}_3_object_masks_eps{e}_cons{c}.h5`, 9x `{P}_3_checker_masks_eps{e}_cons{c}.h5` |
| `R3_universals` | No | Same for Rust | `{P}_2_*.h5` | 9x `{P}_3_*.h5` |

Thresholds: epsilon {0.001, 0.1, 0.5} x consistency {0.2, 0.5, 0.8}

## Stage 4 — Neuron Classification (per language x model)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `4_neuron_list` | No | Concept-only / shared / token-only partition. Loops all 9 settings. | `{P}_3_object_masks_*.h5`, `{P}_3_checker_masks_*.h5` | 9x `{P}_4_neuron_list_eps{e}_cons{c}_all_layers_both.csv` |
| `R4_neuron_list` | No | Same for Rust | `{P}_3_*.h5` | 9x `{P}_4_neuron_list_*_all_layers_both.csv` |

CSV columns: `object`, `layer`, `n_concept_only`, `n_both`, `n_token_only`, `concept_only`, `both`, `token_only`

## Stage 5 — Per-combo Analysis Tables (per language x model)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `5_writeup_tables` | No | Python concept fraction tables | `{P}_4_neuron_list_*.csv` | Display only |
| `R5_writeup_tables` | No | Rust concept fraction tables | `{P}_4_neuron_list_*.csv` | Display only |

## Stage 6 — Causal Validation

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `6_ablation` | Yes | Per-layer neuron ablation, log P(keyword) drop | `{P}_4_neuron_list_*.csv`, `{P}_1A_*.parquet` | `{P}_6_ablation_results.csv`, `{P}_6_ablation_stats.csv`, 4 PNG figures |

Design doc: `docs/6_ablation_design.md`

## Stage 7 — Core Experiments (cross-cutting, one notebook each)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `7_E3_meta_circuits` | No | Group intersection + permutation test | `{*}_4_neuron_list_*.csv` | `7_E3_meta_circuit_results.csv`, Jaccard heatmaps, dendrograms |
| `7_E6_layer_dynamics` | No | Flow vectors, circuit size curves, flow type classification | `{*}_3_object_masks_*.h5` | `7_E6_flow_vectors.csv`, `7_E6_circuit_sizes.csv`, `7_E6_flow_type_assignments.csv` |
| `7_E7_cross_language` | No | Python vs Rust neuron sharing per equivalence class | `{*}_4_neuron_list_*.csv` | `7_E7_cross_language_results.csv`, sharing heatmaps |

## Stage 8 — Optional Experiments (need 1C/1D prompts first)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `8_E4_wellformedness` | No | Validity vs error neurons (well-formed vs malformed) | `{*}_3_object_masks_*.h5`, `{*}_3_malformed_masks_*.h5` | `8_E4_wellformedness_results.csv` |
| `8_E5_composition` | No | Compositional residuals (pair co-occurrence vs union) | `{*}_3_object_masks_*.h5`, `{*}_3_composition_masks_*.h5` | `8_E5_composition_results.csv` |

Blocked on: `1C_malformed_prompts.ipynb` and `1D_composition_prompts.ipynb` (not yet implemented)

## Stage 9 — Stage 1 Results

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `9_results_stage1` | No | Consolidated E1+E2+E3+E6+E7 results, tables, figures | `{*}_4_neuron_list_*.csv`, `7_E3_*.csv`, `7_E6_*.csv`, `7_E7_*.csv` | `9_results_*.csv`, `9_E1_*.png`, `9_E2_*.png` |

## Stage 10 — Cross-Model (needs second model)

| Notebook | GPU | Purpose | Input | Output |
|---|---|---|---|---|
| `10_E8_cross_model` | No | Spearman correlations across Qwen/DeepSeek | All `{*}_4_neuron_list_*.csv`, `7_E6_*.csv` | `9_E8_cross_model_results.csv` |

Blocked on: DeepSeek pipeline run

---

## Execution Order

```
Wave 0 (prompt generation):
  1A, 1B           Python prompts (1A needs GPU)
  R1A, R1B         Rust prompts (R1A needs GPU)

Wave 1 (extraction, GPU):
  2                Python extraction
  R2               Rust extraction

Wave 2 (CPU, parallel):
  3, R3            Threshold sweep
  4, R4            Neuron lists (loops all 9 settings)
  5, R5            Per-combo tables

Wave 3 (CPU, parallel):
  6                Ablation (GPU, Python only for now)
  7_E3             Meta-circuits
  7_E6             Layer dynamics
  7_E7             Cross-language

Wave 4 (results):
  9_results_stage1 Consolidated results

Optional:
  8_E4, 8_E5       After 1C/1D are implemented
  10_E8            After DeepSeek pipeline
```

---

## Data File Naming

All files: `{LANG}_{MODEL}_{step}_{name}` where:
- `LANG`: `P` (Python), `R` (Rust)
- `MODEL`: `QW` (Qwen), `DS` (DeepSeek)

Experiment outputs (stages 7-10) have no lang/model prefix — they are cross-cutting.

## Internal Key Prefixes (inside HDF5)

| Pipeline | Universal mask keys | Checker mask keys |
|---|---|---|
| Python | `ast__For`, `builtin__list` | `ast__For`, `builtin__list` |
| Rust | `rust__For`, `rust__i32` | `rust__For`, `rust__Vec` |
