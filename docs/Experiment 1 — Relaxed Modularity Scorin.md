Experiment 1 — Relaxed Modularity Scoring
Coding Instructions for Claude Code
Purpose: Compute a modularity score for all 106 universal circuits at four values of the relaxation parameter p = {0, 0.05, 0.10, 0.25}. Output: a single table of shape 106 × 4.
GitHub repo: https://github.com/piotrwilam/CSP-Atlas
Data source: The HDF5 atlas file produced by Module 2, stored in Google Drive at /content/drive/MyDrive/DATA/CSP-Atlas/. The filename ends with _dynamic_feature_atlas.h5. If multiple exist, use the most recent.

1. What This Notebook Does
The original modularity score (from the 4_modularity_scores notebook) asks: "Is circuit X's mean Jaccard similarity to ALL 105 other circuits significantly lower than expected by chance?" This is strict — one high-similarity neighbor can kill the score.
The relaxed modularity score asks: "After dropping the top p fraction of most-similar neighbors, is circuit X's mean Jaccard similarity to the REMAINING circuits significantly lower than expected by chance?" This allows a circuit to have a small natural family while still scoring as modular relative to the broader population.
At p=0 we recover the original strict modularity. At p=0.25 we allow up to ~26 "friends" before testing.

2. Data Loading
2.1 Locate and load the HDF5 file
pythonimport h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# --- Find the atlas HDF5 ---
DRIVE_DIR = Path("/content/drive/MyDrive/DATA/CSP-Atlas")
h5_candidates = sorted(DRIVE_DIR.glob("*_dynamic_feature_atlas.h5"))
assert len(h5_candidates) > 0, f"No atlas HDF5 found in {DRIVE_DIR}"
H5_PATH = h5_candidates[-1]  # most recent
print(f"Loading: {H5_PATH}")
2.2 Extract all 106 universal circuit masks
The HDF5 stores universal masks under the group universal_masks/. Each dataset is named layer_{L}/ast__{Name} or layer_{L}/builtin__{Name} and contains a boolean array of shape (2048,).
Build a dictionary: masks[object_name][layer_id] → numpy bool array of shape (2048,).
pythonatlas = h5py.File(H5_PATH, "r")
um = atlas["universal_masks"]

N_LAYERS = 8
masks = {}  # {name: {layer: np.array(bool, 2048)}}
obj_types = {}  # {name: "ast" or "builtin"}

for layer_id in range(N_LAYERS):
    layer_key = f"layer_{layer_id}"
    if layer_key not in um:
        continue
    for ds_name in um[layer_key]:
        # ds_name format: "ast__For" or "builtin__list"
        parts = ds_name.split("__", 1)
        obj_type = parts[0]   # "ast" or "builtin"
        obj_name = parts[1]   # "For", "list", etc.
        
        full_name = ds_name  # keep "ast__For" as unique key
        
        if full_name not in masks:
            masks[full_name] = {}
            obj_types[full_name] = obj_type
        
        masks[full_name][layer_id] = np.array(um[layer_key][ds_name], dtype=bool)

all_objects = sorted(masks.keys())
N_OBJECTS = len(all_objects)
print(f"Loaded {N_OBJECTS} universal circuits")
print(f"  AST:     {sum(1 for v in obj_types.values() if v == 'ast')}")
print(f"  Builtin: {sum(1 for v in obj_types.values() if v == 'builtin')}")

# Sanity check: each object should have all 8 layers
for obj in all_objects:
    assert len(masks[obj]) == N_LAYERS, f"{obj} has {len(masks[obj])} layers, expected {N_LAYERS}"
IMPORTANT — Key naming: Inspect the actual HDF5 to confirm the naming convention. The separator might be "__" (double underscore) or something else. Run this diagnostic first:
python# Diagnostic: print a few dataset names to confirm format
layer_key = "layer_0"
sample_names = list(um[layer_key].keys())[:5]
print("Sample dataset names:", sample_names)
# Adjust the parsing logic above if the format differs

3. Precompute All Pairwise Jaccard Similarities
To avoid redundant computation across p values, precompute the full Jaccard matrix at every layer.
pythondef jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard = |A ∩ B| / |A ∪ B|. Returns 0.0 if both empty."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

# Precompute: jaccard_matrices[layer] = np.array shape (N, N)
jaccard_matrices = {}
for layer_id in tqdm(range(N_LAYERS), desc="Jaccard matrices"):
    mat = np.zeros((N_OBJECTS, N_OBJECTS), dtype=np.float32)
    for i in range(N_OBJECTS):
        for j in range(i + 1, N_OBJECTS):
            sim = jaccard_similarity(
                masks[all_objects[i]][layer_id],
                masks[all_objects[j]][layer_id]
            )
            mat[i, j] = sim
            mat[j, i] = sim
        mat[i, i] = 1.0  # self-similarity
    jaccard_matrices[layer_id] = mat

print(f"Precomputed {N_LAYERS} Jaccard matrices of shape {mat.shape}")

4. Relaxed Modularity Score Computation
4.1 The trimmed mean Jaccard statistic
For a given circuit index i, layer l, and relaxation parameter p:

Get the row jaccard_matrices[l][i, :] — all 106 Jaccard values.
Exclude the self-similarity (index i).
Sort the remaining 105 values in descending order.
Drop the top k = floor(p * 105) values (the most similar neighbors).
Compute the mean of the remaining 105 - k values. This is the trimmed mean Jaccard for circuit i at layer l with parameter p.

pythondef trimmed_mean_jaccard(jaccard_row: np.ndarray, self_idx: int, p: float) -> float:
    """
    Compute mean Jaccard after dropping top-p fraction of most similar neighbors.
    
    Args:
        jaccard_row: Full row from Jaccard matrix, shape (N_OBJECTS,)
        self_idx: Index of the circuit itself (to exclude self-similarity)
        p: Fraction of nearest neighbors to drop (0 = strict, 0.25 = drop top 25%)
    
    Returns:
        Trimmed mean Jaccard similarity
    """
    # Remove self
    others = np.delete(jaccard_row, self_idx)
    n_others = len(others)
    
    # Number of neighbors to drop
    k = int(np.floor(p * n_others))
    
    if k > 0:
        # Sort descending, drop top k
        sorted_desc = np.sort(others)[::-1]
        trimmed = sorted_desc[k:]
    else:
        trimmed = others
    
    if len(trimmed) == 0:
        return 0.0
    
    return float(trimmed.mean())
4.2 Permutation test
For each circuit i, layer l, and parameter p:

Compute the observed trimmed mean Jaccard.
Run 500 permutations: randomly shuffle the neuron labels (permute the 2048-bit mask), recompute the full Jaccard row against all other (real) circuits, compute the trimmed mean.
The p-value is the fraction of permutations where the permuted trimmed mean is ≤ the observed trimmed mean.
If p-value < 0.05, the circuit is significantly distinctive at this layer.

CRITICAL — What gets permuted: We permute the NEURON INDICES of the test circuit's mask, NOT the object labels. This tests whether the specific neurons in the circuit are specially positioned relative to the population, or whether any random set of the same number of neurons would look equally distinct.
pythonN_PERMUTATIONS = 500
SIGNIFICANCE = 0.05

def permutation_test_one_circuit(
    circuit_idx: int,
    layer_id: int,
    p: float,
    n_perms: int = N_PERMUTATIONS
) -> tuple:
    """
    Test whether circuit_idx has significantly lower trimmed-mean Jaccard
    than expected by chance at the given layer and relaxation p.
    
    Returns:
        (observed_stat, pvalue, is_significant)
    """
    # Observed statistic
    observed = trimmed_mean_jaccard(
        jaccard_matrices[layer_id][circuit_idx], circuit_idx, p
    )
    
    # The circuit's mask
    real_mask = masks[all_objects[circuit_idx]][layer_id]
    n_active = real_mask.sum()
    
    # If circuit is empty, it can't be modular
    if n_active == 0:
        return (0.0, 1.0, False)
    
    # Permutation distribution
    perm_stats = np.zeros(n_perms, dtype=np.float32)
    
    for perm_i in range(n_perms):
        # Create a random mask with the same number of active neurons
        perm_mask = np.zeros(2048, dtype=bool)
        perm_indices = np.random.choice(2048, size=int(n_active), replace=False)
        perm_mask[perm_indices] = True
        
        # Compute Jaccard of permuted mask against all other REAL circuits
        perm_jaccard_row = np.zeros(N_OBJECTS, dtype=np.float32)
        for j in range(N_OBJECTS):
            if j == circuit_idx:
                perm_jaccard_row[j] = 1.0  # self
            else:
                other_mask = masks[all_objects[j]][layer_id]
                perm_jaccard_row[j] = jaccard_similarity(perm_mask, other_mask)
        
        perm_stats[perm_i] = trimmed_mean_jaccard(perm_jaccard_row, circuit_idx, p)
    
    # One-sided test: is observed LOWER than permutation distribution?
    pvalue = float((perm_stats <= observed).sum() / n_perms)
    
    return (observed, pvalue, pvalue < SIGNIFICANCE)
4.3 Compute all scores
The modularity score for circuit i at parameter p = number of layers (out of 8) where the permutation test is significant.
pythonP_VALUES = [0.0, 0.05, 0.10, 0.25]

# Results storage
# rows: objects, columns: p values
results = np.zeros((N_OBJECTS, len(P_VALUES)), dtype=int)

# Also store per-layer details for diagnostics
detail_records = []

for obj_idx in tqdm(range(N_OBJECTS), desc="Circuits"):
    obj_name = all_objects[obj_idx]
    
    for p_col, p_val in enumerate(P_VALUES):
        layer_score = 0
        
        for layer_id in range(N_LAYERS):
            observed, pvalue, is_sig = permutation_test_one_circuit(
                obj_idx, layer_id, p_val
            )
            
            if is_sig:
                layer_score += 1
            
            detail_records.append({
                "object": obj_name,
                "type": obj_types[obj_name],
                "layer": layer_id,
                "p_trim": p_val,
                "observed_trimmed_jaccard": round(observed, 4),
                "pvalue": round(pvalue, 4),
                "significant": is_sig
            })
        
        results[obj_idx, p_col] = layer_score

print("Done. Results shape:", results.shape)
RUNTIME WARNING: This is 106 objects × 4 p-values × 8 layers × 500 permutations = ~1.7M permutation tests. Each test computes 106 Jaccard similarities. Total: ~180M Jaccard computations.
Optimization — MANDATORY for feasibility: Vectorize the Jaccard computation. Instead of looping through objects inside the permutation loop, use matrix operations:
python# Faster Jaccard: all-vs-one using broadcasting
def fast_jaccard_one_vs_all(query_mask: np.ndarray, all_masks_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Jaccard similarity of one mask against a matrix of masks.
    
    Args:
        query_mask: shape (2048,) bool
        all_masks_matrix: shape (N_OBJECTS, 2048) bool
    
    Returns:
        shape (N_OBJECTS,) float32
    """
    intersection = np.logical_and(query_mask[np.newaxis, :], all_masks_matrix).sum(axis=1)
    union = np.logical_or(query_mask[np.newaxis, :], all_masks_matrix).sum(axis=1)
    result = np.zeros(len(all_masks_matrix), dtype=np.float32)
    nonzero = union > 0
    result[nonzero] = intersection[nonzero] / union[nonzero]
    return result
Build a stacked mask matrix per layer for fast lookup:
python# Precompute: mask_matrices[layer] = np.array shape (N_OBJECTS, 2048), bool
mask_matrices = {}
for layer_id in range(N_LAYERS):
    mat = np.zeros((N_OBJECTS, 2048), dtype=bool)
    for i, obj_name in enumerate(all_objects):
        mat[i] = masks[obj_name][layer_id]
    mask_matrices[layer_id] = mat
Then the inner permutation loop becomes:
python# Inside permutation_test_one_circuit, replace the inner j-loop with:
perm_jaccard_row = fast_jaccard_one_vs_all(perm_mask, mask_matrices[layer_id])
This should reduce runtime from hours to ~15-30 minutes on a T4 GPU (CPU-only since it's numpy).
ADDITIONAL OPTIMIZATION — batch all permutations at once:
pythondef permutation_test_one_circuit_fast(
    circuit_idx: int,
    layer_id: int,
    p: float,
    n_perms: int = N_PERMUTATIONS
) -> tuple:
    """Vectorized version: generate all permuted masks at once."""
    
    observed = trimmed_mean_jaccard(
        jaccard_matrices[layer_id][circuit_idx], circuit_idx, p
    )
    
    real_mask = masks[all_objects[circuit_idx]][layer_id]
    n_active = int(real_mask.sum())
    
    if n_active == 0:
        return (0.0, 1.0, False)
    
    all_others = mask_matrices[layer_id]  # (N_OBJECTS, 2048)
    
    perm_stats = np.zeros(n_perms, dtype=np.float32)
    
    for perm_i in range(n_perms):
        perm_mask = np.zeros(2048, dtype=bool)
        perm_mask[np.random.choice(2048, size=n_active, replace=False)] = True
        
        perm_jaccard_row = fast_jaccard_one_vs_all(perm_mask, all_others)
        perm_stats[perm_i] = trimmed_mean_jaccard(perm_jaccard_row, circuit_idx, p)
    
    pvalue = float((perm_stats <= observed).sum() / n_perms)
    return (observed, pvalue, pvalue < SIGNIFICANCE)
FURTHER OPTIMIZATION — reuse permutations across p values:
Since p only affects which neighbors get trimmed (not the permuted masks), you can generate permuted Jaccard rows ONCE per (circuit, layer) and then compute the trimmed mean for all four p values from the same data. This cuts total permutation work by 4x.
pythondef permutation_test_all_p_values(
    circuit_idx: int,
    layer_id: int,
    p_values: list,
    n_perms: int = N_PERMUTATIONS
) -> dict:
    """
    Run permutation test for one circuit at one layer, for ALL p values at once.
    
    Returns:
        {p_val: (observed, pvalue, is_significant)}
    """
    real_mask = masks[all_objects[circuit_idx]][layer_id]
    n_active = int(real_mask.sum())
    
    if n_active == 0:
        return {p: (0.0, 1.0, False) for p in p_values}
    
    all_others = mask_matrices[layer_id]
    
    # Observed statistics for each p
    observed_row = jaccard_matrices[layer_id][circuit_idx]
    observed_stats = {
        p: trimmed_mean_jaccard(observed_row, circuit_idx, p)
        for p in p_values
    }
    
    # Generate all permuted Jaccard rows once
    perm_rows = np.zeros((n_perms, N_OBJECTS), dtype=np.float32)
    for perm_i in range(n_perms):
        perm_mask = np.zeros(2048, dtype=bool)
        perm_mask[np.random.choice(2048, size=n_active, replace=False)] = True
        perm_rows[perm_i] = fast_jaccard_one_vs_all(perm_mask, all_others)
    
    # Compute trimmed mean for each p for each permutation
    results = {}
    for p in p_values:
        perm_stats = np.array([
            trimmed_mean_jaccard(perm_rows[perm_i], circuit_idx, p)
            for perm_i in range(n_perms)
        ], dtype=np.float32)
        
        obs = observed_stats[p]
        pval = float((perm_stats <= obs).sum() / n_perms)
        results[p] = (obs, pval, pval < SIGNIFICANCE)
    
    return results
MAIN LOOP using the optimized version:
pythonnp.random.seed(42)

results = np.zeros((N_OBJECTS, len(P_VALUES)), dtype=int)
detail_records = []

for obj_idx in tqdm(range(N_OBJECTS), desc="Circuits"):
    obj_name = all_objects[obj_idx]
    
    for layer_id in range(N_LAYERS):
        layer_results = permutation_test_all_p_values(
            obj_idx, layer_id, P_VALUES
        )
        
        for p_col, p_val in enumerate(P_VALUES):
            observed, pvalue, is_sig = layer_results[p_val]
            
            if is_sig:
                results[obj_idx, p_col] += 1
            
            detail_records.append({
                "object": obj_name,
                "type": obj_types[obj_name],
                "layer": layer_id,
                "p_trim": p_val,
                "observed_trimmed_jaccard": round(observed, 4),
                "pvalue": round(pvalue, 4),
                "significant": is_sig
            })

5. Output
5.1 Main results table (106 × 4)
pythonresults_df = pd.DataFrame(
    results,
    index=all_objects,
    columns=[f"p={p}" for p in P_VALUES]
)
results_df.insert(0, "type", [obj_types[obj] for obj in all_objects])

# Sort: by p=0 score descending, then by p=0.25 score descending
results_df = results_df.sort_values(
    [f"p={P_VALUES[0]}", f"p={P_VALUES[-1]}"],
    ascending=[False, False]
)

print(results_df.to_string())
5.2 Save outputs
pythonOUTPUT_DIR = Path("/content/drive/MyDrive/DATA/CSP-Atlas")

# Main table
results_path = OUTPUT_DIR / "relaxed_modularity_scores.csv"
results_df.to_csv(results_path)
print(f"Saved: {results_path}")

# Detailed per-layer results
detail_df = pd.DataFrame(detail_records)
detail_path = OUTPUT_DIR / "relaxed_modularity_detail.csv"
detail_df.to_csv(detail_path, index=False)
print(f"Saved: {detail_path}")
5.3 Summary statistics
pythonprint("\n=== SUMMARY ===\n")

for p_col, p_val in enumerate(P_VALUES):
    col = f"p={p_val}"
    n_above_zero = (results_df[col] > 0).sum()
    n_ast_above = ((results_df[col] > 0) & (results_df["type"] == "ast")).sum()
    n_blt_above = ((results_df[col] > 0) & (results_df["type"] == "builtin")).sum()
    max_score = results_df[col].max()
    top_scorer = results_df[col].idxmax()
    
    print(f"p = {p_val}:")
    print(f"  Circuits with score > 0: {n_above_zero} / {N_OBJECTS}")
    print(f"    AST:     {n_ast_above}")
    print(f"    Builtin: {n_blt_above}")
    print(f"  Max score: {max_score}/8 ({top_scorer})")
    print()

# Score changes from p=0 to p=0.25
col_strict = f"p={P_VALUES[0]}"
col_relaxed = f"p={P_VALUES[-1]}"
results_df["delta"] = results_df[col_relaxed] - results_df[col_strict]

increased = results_df[results_df["delta"] > 0]
decreased = results_df[results_df["delta"] < 0]
stable = results_df[results_df["delta"] == 0]

print(f"Score change (p=0 → p=0.25):")
print(f"  Increased: {len(increased)} circuits")
print(f"  Decreased: {len(decreased)} circuits")
print(f"  Stable:    {len(stable)} circuits")

if len(decreased) > 0:
    print(f"\n  Decreased circuits (possible token-driven):")
    for idx, row in decreased.sort_values("delta").iterrows():
        print(f"    {idx}: {row[col_strict]}/8 → {row[col_relaxed]}/8 (Δ={row['delta']})")

if len(increased) > 0:
    print(f"\n  Increased circuits (hidden modularity):")
    for idx, row in increased.sort_values("delta", ascending=False).iterrows():
        print(f"    {idx}: {row[col_strict]}/8 → {row[col_relaxed]}/8 (Δ={row['delta']})")

6. Validation Checks
Run these before trusting results.
6.1 Reproduce original modularity at p=0
The p=0 column should match the original modularity scores from the 4_modularity_scores notebook. The top scorers should be Import (7/8), Break (6/8), Pass (6/8), ImportFrom (5/8), Continue (4/8). If they don't match, debug the data loading (Section 2) first.
6.2 Monotonicity check
For each circuit, the score should be monotonically non-decreasing as p increases (dropping more neighbors can only make the remaining population easier to distinguish from). If any circuit has a score that DECREASES with increasing p, there is a bug.
pythonfor obj_idx, obj_name in enumerate(all_objects):
    scores = [results[obj_idx, p_col] for p_col in range(len(P_VALUES))]
    for i in range(1, len(scores)):
        if scores[i] < scores[i-1]:
            print(f"⚠ NON-MONOTONIC: {obj_name} scores={scores}")
            break
WAIT — monotonicity is NOT guaranteed. Trimming changes which neighbors are in the comparison set, and the permutation test has stochastic variance. A circuit could lose significance at one layer when the trimming removes a high-similarity neighbor that was dragging up the mean, making the remaining population's mean closer to the permuted baseline. This is unlikely but possible. If you see non-monotonicity, flag it but don't treat it as a bug — just report it. Increase N_PERMUTATIONS to 1000 for those specific circuits to reduce noise.
6.3 Sanity: p=0.25 should not make everything modular
If more than ~40 circuits score above 0 at p=0.25, the test is too lenient. The whole point is that most circuits are NOT modular — relaxation should surface a few hidden cases, not turn everything significant.

7. Notebook Structure
Create a single Colab notebook: experiment1_relaxed_modularity.ipynb
CellTypeContent0MarkdownTitle, purpose, parameter definitions1CodeConfiguration: H5_PATH, N_PERMUTATIONS=500, SIGNIFICANCE=0.05, P_VALUES, random seed2CodeImports3CodeMount Google Drive4CodeLoad HDF5, extract all universal masks (Section 2)5CodeDiagnostic: print sample HDF5 keys, confirm format, count objects6CodePrecompute mask_matrices and jaccard_matrices (Section 3)7CodeDefine helper functions: jaccard_similarity, fast_jaccard_one_vs_all, trimmed_mean_jaccard (Sections 3–4)8CodeDefine permutation_test_all_p_values (Section 4.3 optimized)9CodeMain loop: compute all scores (Section 4.3 main loop)10CodeBuild results_df, save CSVs (Section 5.1–5.2)11CodeSummary statistics and delta analysis (Section 5.3)12CodeValidation checks (Section 6)13CodeDisplay final table sorted by p=0 score

8. Critical Implementation Notes

Random seed: Set np.random.seed(42) before the main loop. Results must be reproducible.
HDF5 key format: Do NOT assume the key format. Inspect the file first (Cell 5). The separator between type and name might be "__" or "_". Adjust parsing accordingly.
Empty circuits: Some universal circuits may have zero active neurons at certain layers (especially layer 5 where compression is maximal). Handle gracefully — score as non-significant.
Memory: 106 × 2048 × 8 layers of bool arrays is tiny (~1.7 MB). The Jaccard matrices are 106 × 106 × 8 = ~720 KB. No memory issues.
Runtime estimate: With the vectorized implementation, expect ~20–40 minutes on Colab T4. The bottleneck is the 500 permutations × 106 circuits × 8 layers = 424,000 permuted Jaccard row computations. Each one is a vectorized (2048,) × (106, 2048) operation — fast in numpy.
Do not close the HDF5 file until all data is loaded into numpy arrays. Once masks are in memory, the file can be closed.
Output location: Save both CSVs to the same Drive directory as the HDF5 atlas. This keeps all project data together.