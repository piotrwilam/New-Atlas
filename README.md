# CSP-Atlas — Module 1 Version 1

**Programmatic Prompt Generation & Comprehension Filtering ("The Atlas Builder")**

> Source of truth: Project Atlas 2, Sections 2 and 9

---

## Section 1: Architecture

### 4-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage A              Stage B              Stage C        Stage D    │
│  Concept Matrix  ───► Variance Engine ───► Perplexity ───► Parquet  │
│  (125 × 153 pairs)    (150 vars/pair)      Filter          Export   │
│                                            (keep top 100)           │
└─────────────────────────────────────────────────────────────────────┘

Inputs / Outputs at each stage:
  A → queue of (ast_node, builtin_obj) pairs
  B → list of raw prompt dicts with prompt_text, ast_verified, domain, wrapper
  C → scored & sorted list; catastrophic cells (avg loss > threshold) dropped
  D → validated_prompts.parquet + stats.json + per-N checkpoints
```

### Essence vs. Variance

Every prompt has an **Essence** (invariant core) and three layers of **Variance**.
The Essence is a structurally valid Python snippet guaranteed to contain the
target AST node applied to the target builtin — built programmatically via
`ast.parse()` / `ast.unparse()`, never by raw string concatenation.

Variance is injected in three orthogonal directions:

- **Lexical / Semantic** — variable names, class names, and literal values are
  drawn from radically different domains (finance, biology, gaming, physics,
  e-commerce). A `For` loop over a list looks completely different when the list
  is `ledger_entries` versus `dna_samples`.

- **Structural / Contextual** — the snippet is wrapped at global scope (~40%),
  inside a function (~30%), or inside a class method (~30%).

- **Padding** — unrelated assignments or `print()` calls are optionally
  prepended or appended.

Module 2 later **intersects** the activation patterns across all 100 variations
of one (AST, builtin) cell. Because the only shared invariant is the structural
essence, the intersection washes away neurons that fire on lexical or padding
noise and isolates the pure structural circuit — a **Universal Concept**.

### Categorization Protocol (end goal)

Once Module 2 has extracted the 278 Universal Modules, each is categorized:

- **Monolithic** — a single dense circuit (e.g., the `For`-loop iterator).
- **Archipelago** — multiple disjoint sub-circuits that always co-activate.
- **Entangled Fog** — circuits that cannot be separated; polysemantic heads.

This categorization drives the downstream compression and interpretability work.

### Output Schema

| Column          | Type    | Example                           | Notes                                            |
|-----------------|---------|-----------------------------------|--------------------------------------------------|
| `ast_node`      | String  | `"For"`, `"ListComp"`, `"Try"`    | AST class name (no `ast.` prefix)                |
| `builtin_obj`   | String  | `"list"`, `"dict"`, `"ValueError"`| Builtin name as string                           |
| `variation_id`  | Integer | `0` to `99`                       | Sequential within (ast_node, builtin_obj) group  |
| `prompt_text`   | String  | `ast.unparse()` output            | Executable Python; canonical PEP 8 spacing       |
| `sequence_loss` | Float   | `2.341`                           | Cross-entropy loss from CSP forward pass         |
| `token_length`  | Integer | `47`                              | Token count after CSP tokenizer encoding         |
| `ast_verified`  | Boolean | `True`                            | Confirms `ast.parse(prompt_text)` contains target|

### Module 1 → Module 2 Data Flow

```
Module 1 output
  └── validated_prompts.parquet
        │
        ▼  grouped by (ast_node, builtin_obj)
  Module 2 forward passes
        │  activations at every layer / head
        ▼
  Binarization (threshold at median activation)
        │
        ▼
  Consistency Score = |intersection of binary masks| / N_KEEP
        │
        ▼
  Marginalization across all 153 builtins
        │
        ▼
  278 Universal AST Modules → Categorization Protocol
```

---

## Section 2: Usage Manual

### Setup

```bash
git clone https://github.com/piotrwilam/CSP-Atlas.git
cd CSP-Atlas

# Create environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch transformers numpy pandas pyarrow tqdm huggingface_hub
pip install pytest matplotlib      # for tests and quality plots
```

### Running

Open `notebooks/module1_run.ipynb` in Jupyter or Google Colab.
Edit **Cell 1** (Configuration) to select your run mode, then run all cells.

```python
# Cell 1 — change this one line:
MODE = "test"   # or "small" or "full"
```

The notebook mounts Google Drive automatically when run in Colab.
Locally it falls back to the `CHECKPOINT_DIR` path you set.

### Run Modes

| Mode    | AST nodes | Builtins | N_GENERATE | N_KEEP | Pairs   | Prompts    | Est. runtime (T4) |
|---------|-----------|----------|-----------|--------|---------|------------|-------------------|
| `test`  | 5         | 5        | 15        | 10     | 25      | ~250       | 5–10 min          |
| `small` | ~40       | ~50      | 75        | 50     | ~2,000  | ~100K      | 4–8 hours         |
| `full`  | 125       | 153      | 150       | 100    | ~19,125 | ~1.9M      | 20–40 hours       |

### Adjustable Parameters

| Parameter                | Test       | Small      | Full            | Description                               |
|--------------------------|------------|------------|-----------------|-------------------------------------------|
| `MODE`                   | `"test"`   | `"small"`  | `"full"`        | Run mode                                  |
| `N_GENERATE`             | `15`       | `75`       | `150`           | Variations generated per pair             |
| `N_KEEP`                 | `10`       | `50`       | `100`           | Variations kept after filter              |
| `CATASTROPHIC_THRESHOLD` | `10.0`     | `10.0`     | `10.0`          | Max avg loss before discarding cell       |
| `CHECKPOINT_EVERY`       | `100`      | `100`      | `50`            | Checkpoint interval (pairs)               |
| `WRAPPER_WEIGHTS`        | `[.4,.3,.3]` | same     | same            | Global / function / class proportions     |
| `DOMAINS`                | `5`        | `5`        | `5` (or 8–10)   | Semantic domain count                     |
| `MODEL_NAME`             | `openai/circuit-sparsity` | same | same      | HuggingFace model ID                      |
| `CHECKPOINT_DIR`         | Drive path | Drive path | Drive path      | Output directory                          |

### Expanding the Pipeline

**Add an AST template** — add a new key/lambda to the `T` dict in
`ASTPromptGenerator._build_essence()` (Cell 8 / `src/module1/generators.py`).

**Add a domain** — extend the `DOMAINS` dict in Cell 5 / `src/module1/variance_schema.py`
with a new key containing `var_names` and `mock_data` sub-dicts.

**Tune thresholds** — adjust `CATASTROPHIC_THRESHOLD` or `N_KEEP` in Cell 1.

### HuggingFace Upload

The full-run output is automatically uploaded to `https://huggingface.co/CSP-Atlas`
when `MODE = "full"` (Cell 20). You will be prompted to log in via `huggingface_hub`.

For manual upload:

```python
from huggingface_hub import HfApi, login
login()
api = HfApi()
api.upload_file(
    path_or_fileobj="path/to/validated_prompts.parquet",
    path_in_repo="module1/validated_prompts.parquet",
    repo_id="CSP-Atlas",
    repo_type="model",
)
```

### Running Tests

```bash
pytest tests/test_module1.py -v
```

Tests cover Stages A, B, and C (with a mocked model) plus an end-to-end
pipeline smoke test. No GPU or model download required.

---

## Project Structure

```
CSP-Atlas/
├── src/
│   └── module1/
│       ├── __init__.py          # package exports
│       ├── concept_matrix.py    # Stage A — AST × builtin pairs
│       ├── variance_schema.py   # domains, wrappers, padding constants
│       ├── generators.py        # Stage B — ASTPromptGenerator
│       ├── filters.py           # Stage C — PerplexityFilter
│       └── pipeline.py          # Stage D — orchestration & Parquet export
├── notebooks/
│   └── module1_run.ipynb        # 22-cell self-contained notebook
├── configs/
│   └── module1_config.yaml      # reference config (notebook Cell 1 takes precedence)
├── tests/
│   └── test_module1.py          # pytest suite
└── README.md
```
