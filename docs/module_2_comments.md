# Module 2 — Session Notes

## Topics Covered

### 1. Module 2 Python source files (7 files)
- `extraction.py` — `ActivationExtractor` with manual hooks and `extract_batch()`
- `binarization.py` — `PairRepresentationBuilder` with running-sum memory protocol
- `marginalization.py` — `UniversalModuleComputer` (intersection across builtins/ASTs)
- `metrics.py` — Jaccard similarity/distance, Entanglement Index, Jaccard matrix
- `pipeline.py` — `Module2Pipeline` with checkpoint support
- `io_utils.py` — HDF5 save/load for atlas, pickle checkpoints
- `__init__.py` — public exports

### 2. module2_extraction.ipynb — Built and debugged
- **circuit_sparsity stub injection**: Evolved from empty stubs to downloading real `gpt.py`/`hook_utils.py` from HuggingFace via `hf_hub_download`, loading with `importlib.util`, and patching `GPTConfig` to filter unknown kwargs (e.g. `unembed_rank`). Same pattern as `module1_run.ipynb` Cell 1.
- **Colab Drive mount issues**: `force_remount=True` alone doesn't work if `/content/drive` has stale files. Fix: `fusermount -uz` + `shutil.rmtree` before `drive.mount()`.
- **Path fixes**: `COLAB_SRC = "/content/drive/MyDrive/CODE/CSP-Atlas/src"` (repo lives on Drive, not `/content/CSP-Atlas`). `PARQUET_PATH` under `/content/drive/MyDrive/DATA/CSP-Atlas/`.
- **`torch_dtype` deprecated**: Changed to `dtype=torch.float16` in `from_pretrained()`.
- **Pad token**: `tokenizer.pad_token = tokenizer.eos_token` required for batch tokenization.
- **MLP layer auto-detection**: Regex scan of `model.named_modules()` with pattern `r'^(.*?)(\d+)(\.mlp)$'` to find hook pattern automatically.
- **Hook cleanup**: `extractor.remove_hooks()` in final cell to prevent memory leaks.

### 3. module2_evaluation.ipynb — Built, reviewed, fixed
- UMAP guards for `< 3` modules (skip gracefully on test runs)
- Drive paths and mount cleanup (same pattern as extraction notebook)
- HDF5 metadata loading fix: `load_atlas_hdf5` now reads both attrs and dataset-stored metadata (list fields stored as datasets with `h5py.string_dtype()`)
- 5 evaluation protocols: Topology Map (UMAP), Circuit Overlap (Jaccard heatmaps), Layer Evolution, Compositionality (Entanglement Index), Marginalization Robustness

### 4. Performance optimization — 200x speedup
- **Before**: 15s/pair (single-prompt forward passes, `torch.cuda.empty_cache()` per prompt)
- **After**: 0.07s/pair (14 it/s) with GPU + fp16 + batch_size=64
- Key changes:
  - `extract_batch()` method: tokenizes multiple prompts with padding, single forward pass, extracts at last non-pad token position
  - Batch size 16 → 64 (T4 has plenty of VRAM for 1.68GB model)
  - `torch.float16` halves memory and doubles throughput on T4 tensor cores
  - `tokenizer.padding_side = "left"` for decoder models
  - Removed per-prompt `torch.cuda.empty_cache()` (was ~10-50ms overhead × 127,600 calls)

### 5. Conceptual explanations given
- **UMAP**: Non-linear dimensionality reduction preserving neighborhood structure, uses Jaccard metric on boolean masks. UMAP-1/UMAP-2 are arbitrary coordinate axes — only relative distances matter.
- **GPU batching**: `[16, seq_len, hidden_dim]` tensor processed in one forward pass. Sequences are independent (no cross-sequence attention in causal models). GPU SIMD parallelism means batch=16 takes almost same time as batch=1.
- **PCA vs UMAP**: PCA is linear, assumes continuous data. UMAP handles non-linear manifolds and custom metrics like Jaccard on binary vectors.

### 6. Module 1 stats analysis
- `small_40x50x50_stats.json`: 1276 successful pairs, 21 catastrophic failures, 63,800 total prompts
- Failures cluster around `AnnAssign` (7), `Delete` (5), `Call` (4), `Assign` (3)

## Colab Workflow
- Repo lives at `/content/drive/MyDrive/CODE/CSP-Atlas` (on Google Drive)
- Pull pattern: `git fetch` + `git reset --hard FETCH_HEAD` using `GH_TOKEN` from Colab secrets
- Data lives at `/content/drive/MyDrive/DATA/CSP-Atlas/`
- After local push, user pulls on Colab and restarts runtime

## Key Files Modified
- `src/module2/extraction.py` — added `extract_batch()`
- `src/module2/binarization.py` — batched loop, batch_size=64
- `src/module2/io_utils.py` — `h5py.string_dtype()`, metadata loading fix
- `notebooks/module2_extraction.ipynb` — 17 cells, circuit_sparsity injection v1.30
- `notebooks/module2_evaluation.ipynb` — 12 cells, 5 evaluation protocols
