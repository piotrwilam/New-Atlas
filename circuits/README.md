# `circuits/` — Layer 1: artifact-generation pipeline

The extraction pipeline that produces the frozen artifacts the rest of the
codebase reads from. This code is **frozen** at `archive-pre-refactor` —
rerunning it requires a GPU and ~30–60 min per (lang, model) cell. Under
normal use, you don't touch this package: you read its outputs via the
loaders in [`../atlas/io/`](../atlas/io/).

Read this alongside paper sections **§3.1–§3.6**.

## Modules

| File | Class / function | Purpose |
|---|---|---|
| [`extraction.py`](extraction.py)        | `ActivationExtractor`           | Forward-hook a HuggingFace causal LM at every MLP layer; record last-token activations for a batch of prompts. Hook pattern is configurable per architecture (`blocks.{i}.mlp` for Qwen/Llama-style, etc.). |
| [`binarization.py`](binarization.py)      | `PairRepresentationBuilder`, `RawActivationCollector` | Build per-(ast-node, builtin-object) pair representations and binarise activations under (ε, consistency) thresholds. |
| [`marginalization.py`](marginalization.py)   | `UniversalModuleComputer`       | Intersect masks across the complementary dimension to produce the universal mask `A` for each concept. |
| [`metrics.py`](metrics.py)           | `jaccard_similarity`, `compute_jaccard_matrix` | Array-based Jaccard for boolean masks (the binarisation stage). Distinct from `atlas/analysis/jaccard.py`, which operates on `set[int]` from the decomposition-stage XLSX. |
| [`pipeline.py`](pipeline.py)          | `Module2Pipeline`               | End-to-end orchestrator: load Parquet → extract → binarise → checkpoint → marginalise. |
| [`io_utils.py`](io_utils.py)          | `save_atlas_hdf5`, `load_atlas_hdf5`, `save_activations_hdf5`, `load_activations_hdf5`, `save_checkpoint`, `load_checkpoint` | HDF5 readers / writers for the intermediate artifacts. |

## Runtime requirements

- Python 3.12 (managed via uv; pinned in `.python-version`)
- A GPU with ≥ 16 GB VRAM for the actual model forward passes (fp16 inference); CPU is fine for everything downstream
- `torch`, `transformers`, `accelerate` (declared as the `extraction` extras group in `pyproject.toml`):
  ```
  uv pip install -e ".[extraction]"
  ```

## How notebooks call this

See [`../notebooks/1_artifact_generation/`](../notebooks/1_artifact_generation/) — each notebook imports a handful of classes from here and orchestrates a single pipeline stage.

```python
from circuits.extraction import ActivationExtractor
from circuits.binarization import RawActivationCollector
from circuits.marginalization import UniversalModuleComputer
from circuits.io_utils import save_activations_hdf5
```

## Why this is separate from `atlas/`

`atlas/` is the **analysis & plotting** library — it depends only on the frozen artifacts on disk, never on PyTorch or HuggingFace. That separation means: a reader who just wants to reproduce a paper figure installs `uv pip install -e ".[dev]"` (no torch / transformers), points `ATLAS_DATA_ROOT` at the data, and runs `experiments/fig*.py`. The expensive extraction machinery in this package stays optional.
