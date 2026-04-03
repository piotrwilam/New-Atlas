# New-Atlas — Syntactic Circuit Discovery for Qwen2.5-Coder 7B

Replication and extension of the CSP-Atlas methodology on **Qwen/Qwen2.5-Coder-7B** (28-layer, SwiGLU, GQA, RoPE).

Extracts universal syntactic circuits (AST nodes + Python builtins) via activation intersection and marginalization.

---

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 1A | `1A_object_prompts.ipynb` | Generate (AST node, builtin) prompt variations with perplexity filter |
| 1B | `1B_checker_prompts.ipynb` | Generate token-without-concept prompts for token independence check |
| 2 | `2_extraction.ipynb` | Extract raw MLP activations from both prompt sets |
| 3 | `3_universals.ipynb` | Threshold sweep + marginalization → universal circuits |
| 4 | `4_neuron_list.ipynb` | Compare concept vs token circuits, output neuron indices |

Run in order: 1A/1B → 2 → 3 → 4.

## Project Structure

```
New-Atlas/
├── src/
│   └── module2/
│       ├── extraction.py       # ActivationExtractor (model hooks)
│       ├── binarization.py     # RawActivationCollector
│       ├── marginalization.py  # UniversalModuleComputer
│       ├── metrics.py          # Jaccard similarity
│       ├── io_utils.py         # HDF5 I/O
│       └── pipeline.py         # End-to-end orchestrator
├── notebooks/                  # 5 notebooks (see table above)
└── README.md
```

## Requirements

- Python 3.10+
- GPU with ≥16 GB VRAM (fp16 inference)
- `pip install torch transformers accelerate numpy pandas h5py tqdm pyarrow`
