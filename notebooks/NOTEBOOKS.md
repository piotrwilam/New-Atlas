# Notebook Reference

Two parallel pipelines on the same model (Qwen2.5-Coder-7B, 28 layers, SwiGLU, 3584 MLP output dims):

- **Python pipeline** (1A → 5, 6) — AST nodes + Python builtins
- **Rust pipeline** (R1A → R5) — Rust syntactic constructs + standard library objects

---

## Pipeline Overview

```
                Python                          Rust
           ┌──────────────┐              ┌──────────────┐
Stage 1    │ 1A  Object   │              │ R1A Object   │
Prompts    │ 1B  Checker  │              │ R1B Checker  │
           └──────┬───────┘              └──────┬───────┘
                  │                             │
Stage 2    ┌──────┴───────┐              ┌──────┴───────┐
Extract    │ 2  Extraction│              │ R2 Extraction│   GPU
           └──────┬───────┘              └──────┬───────┘
                  │                             │
Stage 3    ┌──────┴───────┐              ┌──────┴───────┐
Threshold  │ 3  Universals│              │ R3 Universals│   CPU
           └──────┬───────┘              └──────┬───────┘
                  │                             │
Stage 4    ┌──────┴───────┐              ┌──────┴───────┐
Classify   │ 4  NeuronList│              │ R4 NeuronList│   CPU
           └──────┬───────┘              └──────┬───────┘
                  │                             │
Stage 5    ┌──────┴───────┐              ┌──────┴───────┐
Tables     │ 5  Tables    │              │ R5 Tables    │   CPU
           └──────────────┘              └──────────────┘
                  │
Stage 6    ┌──────┴───────┐
Ablation   │ 6  Ablation  │                                 GPU
           └──────────────┘
```

---

## Python Pipeline

### 1A — Object Prompts (`1A_object_prompts.ipynb`)

Generates Python code prompts containing specific (AST node, builtin) pairs.

| | |
|---|---|
| **GPU** | Yes (Qwen2.5-Coder-7B for perplexity filter) |
| **Input** | None (generates from templates) |
| **Output** | `1A_object_prompts.parquet` |
| **Parquet schema** | `ast_node`, `builtin_obj`, `variation_id`, `prompt_text`, `sequence_loss`, `token_length`, `ast_verified` |
| **Concept space** | ~40 AST nodes (12 families) x ~80 builtins (5 families) |
| **Validation** | `ast.parse()` + `ast.walk()` |
| **Modes** | test (25 pairs), small (~2000), full (~5000+) |
| **Variance** | 5 domains (finance, biology, gaming, physics, ecommerce), 3 wrapper types |

### 1B — Checker Prompts (`1B_checker_prompts.ipynb`)

Generates prompts containing a keyword token but NOT the corresponding AST concept.

| | |
|---|---|
| **GPU** | No (tokenizer only) |
| **Input** | None |
| **Output** | `1B_checker_prompts.parquet` |
| **Parquet schema** | `object`, `keyword`, `variation_id`, `prompt_text` |
| **Objects** | 62 (24 AST keywords + 38 builtin keywords) |
| **Categories** | A: string literal, B: comment, C: identifier substring, D: dict key, E: print |
| **Validation** | `ast.parse()` (concept absent) + tokenizer (token present) |
| **Object naming** | `ast__For`, `ast__Import`, `builtin__list`, `builtin__len` |

### 2 — Extraction (`2_extraction.ipynb`)

Extracts raw MLP activation statistics from both prompt sets.

| | |
|---|---|
| **GPU** | Yes |
| **Input** | `1A_object_prompts.parquet`, `1B_checker_prompts.parquet` |
| **Output** | `2_object_activations.h5`, `2_checker_activations.h5` |
| **HDF5 schema** | `/activations/layer_{L}/{ast}__{blt}` (float32), `/firing_counts/layer_{L}/{ast}__{blt}` (int32), `/n_prompts/{ast}__{blt}` |
| **Grouping** | By `(ast_node, builtin_obj)` for objects, by `object` (split on `__`) for checkers |
| **Runtime** | 20-40 hours full mode on A100 |

### 3 — Universals (`3_universals.ipynb`)

Applies 9 threshold combinations and computes universal circuits via intersection.

| | |
|---|---|
| **GPU** | No |
| **Input** | `2_object_activations.h5`, `2_checker_activations.h5` |
| **Output** | 9x `3_object_masks_eps{e}_cons{c}.h5`, 9x `3_checker_masks_eps{e}_cons{c}.h5` |
| **Thresholds** | epsilon: {0.001, 0.1, 0.5}, consistency: {0.2, 0.5, 0.8} |
| **Universal masks** | Intersection across all builtins for each AST node, and vice versa |
| **HDF5 keys** | `universal_masks/layer_{L}/ast__{name}`, `universal_masks/layer_{L}/builtin__{name}` |
| **Checker keys** | `token_checker_masks/layer_{L}/ast__{name}`, `token_checker_masks/layer_{L}/builtin__{name}` |

### 4 — Neuron List (`4_neuron_list.ipynb`)

Compares universal circuits against token checker circuits.

| | |
|---|---|
| **GPU** | No |
| **Input** | `3_object_masks_eps{e}_cons{c}.h5`, `3_checker_masks_eps{e}_cons{c}.h5` |
| **Output** | `4_neuron_list_eps{e}_cons{c}_L{layers}_{type}.csv` |
| **Set operations** | A = universal, B = checker. concept_only = A & ~B, both = A & B, token_only = ~A & B |
| **CSV columns** | `object`, `layer`, `n_concept_only`, `n_both`, `n_token_only`, `concept_only` (index list), `both`, `token_only` |
| **Neuron dim** | 3584 (hardcoded as 18944 in Python, but actual max index is 3583) |

### 5 — Writeup Tables (`5_writeup_tables.ipynb`)

Generates summary tables for publication.

| | |
|---|---|
| **GPU** | No |
| **Input** | `4_neuron_list_eps{e}_cons{c}_L{layers}_both.csv` (up to 9 files) |
| **Groups** | Modular AST (Import, Break, Pass, Continue, Assert, ImportFrom), Non-modular AST (rest), Builtin |
| **Table 1** | Concept fraction across 3x3 parameter grid |
| **Table 2** | Neuron counts at layer 14 (eps=0.5, cons=0.8) |
| **Table 3** | Layer-by-layer concept fraction profile (eps=0.001, cons=0.8) |

### 6 — Ablation (`6_ablation.ipynb`)

Causal validation via per-layer neuron ablation.

| | |
|---|---|
| **GPU** | Yes (A100 recommended, 2-4 hours) |
| **Input** | `4_neuron_list_eps0.5_cons0.8_L..._both.csv`, `1A_object_prompts.parquet` |
| **Output** | `6_ablation_results.csv`, `6_ablation_stats.csv`, 4 PNG figures |
| **Concepts** | 10 targets: Import, Assert, Break, For, While, Try, If, FunctionDef, Lambda, len, range |
| **Protocol** | Per-layer zero/mean ablation of concept-only neurons, measure log P(keyword) drop |
| **Controls** | Shared neurons (A and B), random neurons (outside A and B), matched count |
| **Stats** | Paired t-test + Wilcoxon signed-rank, Bonferroni correction for 28 layers |
| **Design doc** | `6_ablation_design.md` |

---

## Rust Pipeline

Identical architecture, adapted for Rust syntax. Uses `tree-sitter-rust` instead of Python's `ast` module.

### R1A — Rust Object Prompts (`R1A_object_prompts.ipynb`)

| | |
|---|---|
| **GPU** | Yes |
| **Input** | None |
| **Output** | `R1A_object_prompts.parquet` |
| **Parquet schema** | `construct`, `object`, `variation_id`, `prompt_text`, `sequence_loss`, `token_length`, `tree_sitter_verified` |
| **Concept space** | ~35 constructs (15 families) x ~45 objects (5 families) |
| **Constructs** | for_expression, while_expression, loop_expression, if_expression, match_expression, function_item, closure_expression, let_declaration, let_mut, const_item, static_item, struct_item, enum_item, type_alias, impl_item, trait_item, use_declaration, mod_item, return_expression, break_expression, continue_expression, async_block, await_expression, unsafe_block, reference_expression, mutable_reference, dereference, lifetime, macro_invocation, attribute_item, macro_invocation_question_mark |
| **Objects** | Primitives (i32, f64, bool, ...), prelude types (Vec, String, Box, Option, Result), prelude traits (Clone, Debug, Iterator, ...), common std (HashMap, Arc, Mutex, ...) |
| **Validation** | `tree-sitter-rust` parse + `verify_concept()` dispatcher |
| **Compound concepts** | let_mut (let + mutable_specifier), mutable_reference (reference + mutable_specifier), ? operator (try_expression) |
| **Design doc** | `R1_design.md` |

### R1B — Rust Checker Prompts (`R1B_checker_prompts.ipynb`)

| | |
|---|---|
| **GPU** | No |
| **Output** | `R1B_checker_prompts.parquet` |
| **Parquet schema** | `object`, `keyword`, `variation_id`, `prompt_text` |
| **Objects** | 37 (21 construct keywords + 16 object keywords) |
| **Categories** | A: string, B: comment, C: identifier, D: array of tuples (Rust dict analog), E: println! |
| **Validation** | `tree-sitter-rust` (concept absent via node types + identifier check) + tokenizer (token present) |
| **Object naming** | `rust__For`, `rust__Vec`, `rust__HashMap`, etc. (single `rust__` prefix for all) |

### R2 — Extraction (`R2_extraction.ipynb`)

| | |
|---|---|
| **GPU** | Yes |
| **Input** | `R1A_object_prompts.parquet`, `R1B_checker_prompts.parquet` |
| **Output** | `R2_object_activations.h5`, `R2_checker_activations.h5` |
| **Grouping** | By `(construct, object)` for objects, by `object` (split on `__`) for checkers |
| **Difference from Python** | Column names `construct`/`object` instead of `ast_node`/`builtin_obj` |

### R3 — Universals (`R3_universals.ipynb`)

| | |
|---|---|
| **GPU** | No |
| **Input** | `R2_object_activations.h5`, `R2_checker_activations.h5` |
| **Output** | 9x `R3_object_masks_eps{e}_cons{c}.h5`, 9x `R3_checker_masks_eps{e}_cons{c}.h5` |
| **Key difference** | Universal masks saved with `rust__` prefix (not `ast__`/`builtin__`) so keys align with R1B checker naming in R4 |
| **HDF5 keys** | `universal_masks/layer_{L}/rust__{name}` for both construct and object universals |
| **Note** | Does NOT use `save_atlas_hdf5()` — saves manually to control prefix |

### R4 — Neuron List (`R4_neuron_list.ipynb`)

| | |
|---|---|
| **GPU** | No |
| **Input** | `R3_object_masks_eps{e}_cons{c}.h5`, `R3_checker_masks_eps{e}_cons{c}.h5` |
| **Output** | `R4_neuron_list_eps{e}_cons{c}_L{layers}_{type}.csv` |
| **Same as Python** | Set operations, CSV schema identical |
| **Object naming** | `rust__For`, `rust__Vec`, etc. |

### R5 — Writeup Tables (`R5_writeup_tables.ipynb`)

| | |
|---|---|
| **GPU** | No |
| **Input** | `R4_neuron_list_eps{e}_cons{c}_L{layers}_both.csv` |
| **Groups** | Modular Construct (Use, Mod, Break, Continue, Return, Unsafe, Await), Non-modular Construct (For, If, Let, ...), Object (i32, Vec, ...) |
| **Tables** | Same 3 tables as Python, adapted for Rust grouping |

---

## Data Flow Summary

### File naming convention

| Python | Rust |
|---|---|
| `1A_object_prompts.parquet` | `R1A_object_prompts.parquet` |
| `1B_checker_prompts.parquet` | `R1B_checker_prompts.parquet` |
| `2_object_activations.h5` | `R2_object_activations.h5` |
| `2_checker_activations.h5` | `R2_checker_activations.h5` |
| `3_object_masks_eps{e}_cons{c}.h5` | `R3_object_masks_eps{e}_cons{c}.h5` |
| `3_checker_masks_eps{e}_cons{c}.h5` | `R3_checker_masks_eps{e}_cons{c}.h5` |
| `4_neuron_list_eps{e}_cons{c}_L{layers}_both.csv` | `R4_neuron_list_eps{e}_cons{c}_L{layers}_both.csv` |
| `6_ablation_results.csv` | (not yet implemented) |

### Object key prefixes

| Pipeline | Universal mask keys | Checker mask keys |
|---|---|---|
| Python | `ast__For`, `builtin__list` | `ast__For`, `builtin__list` |
| Rust | `rust__For`, `rust__i32` | `rust__For`, `rust__Vec` |

### GPU requirements

| Notebook | GPU | Estimated runtime (A100) |
|---|---|---|
| 1A / R1A | Yes | 5-40 hours (mode dependent) |
| 1B / R1B | No | Minutes |
| 2 / R2 | Yes | 20-40 hours |
| 3 / R3 | No | Minutes |
| 4 / R4 | No | Minutes |
| 5 / R5 | No | Seconds |
| 6 | Yes | 2-4 hours |

---

## Shared Infrastructure

All notebooks use the same `src/module2/` codebase:

| Module | Used by | Purpose |
|---|---|---|
| `extraction.py` | 2, R2, 6 | `ActivationExtractor` with forward hooks on `model.layers.{L}.mlp` |
| `binarization.py` | 2, R2 | `RawActivationCollector` for activation statistics |
| `marginalization.py` | 3, R3 | `UniversalModuleComputer` for set intersections |
| `metrics.py` | 3, R3, 5, R5 | Jaccard similarity, entanglement index |
| `io_utils.py` | 2, 3, R2 | HDF5 save/load. R3 saves manually (prefix issue) |
| `pipeline.py` | (not used by notebooks directly) | End-to-end orchestrator |

### Model

- **Qwen/Qwen2.5-Coder-7B** for all notebooks
- 28 layers, SwiGLU MLP, 3584 MLP output dimensions
- Loaded with `torch.float16`, `device_map="auto"`
- Handles both Python and Rust code natively
