# Layer 1 — Artifact generation

These notebooks produce the frozen data the rest of the codebase reads from.
They are **expensive to run** (each extraction notebook needs a GPU and
~30–60 min per (lang, model) cell) and their outputs are checked into the
`/Users/piotrwilam/Data/New-Atlas/` mirror, not regenerated under normal use.

Read the corresponding paper sections **§3.1–§3.6** alongside.

## Order of execution

Python pipeline (drop the `R` prefix):

1. `0_tokeniser_validation.ipynb` — sanity-check tokenisation behaviour against expected keyword sequences. One-off.
2. `1A_object_prompts.ipynb` — generate concept prompts: (AST node, builtin object) variations with perplexity filter.
3. `1B_checker_prompts.ipynb` — generate matched checker prompts where the keyword token appears outside its structural role.
4. `2_extraction.ipynb` — forward-hook the model, record per-layer last-token MLP activations for both prompt sets. Writes HDF5.
5. `3_universals.ipynb` — apply ε / consistency thresholds, marginalise across the complementary dimension → universal masks A.
6. `4_neuron_list.ipynb` — decompose into concept-only / shared / token-only sets, write the per-layer XLSX consumed by `atlas/io/xlsx.py`.

Rust pipeline runs the same shape on the `R*_*.ipynb` notebooks (R1A → R1B → R2 → R3 → R4).

`V1_extract_vectors.ipynb` is a separate side-pipeline: trains logistic-regression probes and saves their direction vectors (used by Figure 8 in the analysis layer).

## Code these notebooks import from

The `circuits/` top-level package — `ActivationExtractor`, `PairRepresentationBuilder`, `UniversalModuleComputer`, `Module2Pipeline`, the HDF5 readers/writers. See `../../circuits/README.md` (if present) or the module docstrings.

## What gets produced

Outputs land in the data mirror (`$ATLAS_DATA_ROOT`, default `~/Data/New-Atlas/`) as:
- `{lang}_{model}_2_activations.h5` — raw activations
- `{lang}_{model}_3_*_masks_eps{ε}_cons{c}.h5` — binarised + marginalised masks
- `{lang}_{model}_4_neuron_list_eps{ε}_cons{c}_layers_part{1,2}_both.xlsx` — the canonical XLSX consumed by the analysis layer

All paper-shipped results use **ε = 0.5, consistency = 0.8**.
