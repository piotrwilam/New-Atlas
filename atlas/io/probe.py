"""
Readers for probe-validation artifacts.

The §7 validation pipeline produced three artifact types per (lang, model)
cell, all under DATA_ROOT:

  - {lang}_{model}_V2_probe_results.csv
      per-(concept, layer) probe classification accuracy (with std).
  - {lang}_{model}_V2_cosine_jaccard_correlation.csv
      per-layer Pearson r between pairwise probe-cosine and pairwise
      neuron-set Jaccard.
  - {lang}_{model}_V2_weight_vectors.npz
      per-(concept, layer) unit-normalised probe direction vectors,
      used for the L20 scatter (Figure 8).

Only Python × Qwen has these files in the v1 release.
"""

from __future__ import annotations

import csv
from pathlib import Path

from atlas.paths import DATA_ROOT


def load_probe_results(
    *,
    model: str,
    lang: str,
    data_root: Path | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Per-concept, per-layer probe accuracy from the V2 probe CSV.

    Returns
    -------
    dict mapping concept name (no prefix — the CSV stores plain names) →
    dict mapping layer → {`accuracy`, `accuracy_std`, `n_object`, `n_checker`}.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / f"{lang}_{model}_V2_probe_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected probe-results file not found: {path}")

    out: dict[str, dict[int, dict[str, float]]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            concept = row["concept"]
            layer = int(float(row["layer"]))
            out.setdefault(concept, {})[layer] = {
                "accuracy": float(row["accuracy"]),
                "accuracy_std": float(row["accuracy_std"]),
                "n_object": float(row["n_object"]),
                "n_checker": float(row["n_checker"]),
            }
    return out


def load_jaccard_cosine_correlation(
    *,
    model: str,
    lang: str,
    data_root: Path | None = None,
) -> dict[int, dict[str, float]]:
    """Per-layer Pearson r and Spearman ρ between pairwise probe-cosine
    and pairwise concept-only Jaccard.

    Returns
    -------
    dict mapping layer → {`pearson_r`, `spearman_rho`, `p_pearson`,
    `p_spearman`, `n_pairs`}.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / f"{lang}_{model}_V2_cosine_jaccard_correlation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected correlation file not found: {path}")

    out: dict[int, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            layer = int(float(row["layer"]))
            # Saturated last-layer rows (Qwen L26-27, DeepSeek L30-31) have
            # blank pearson_r / spearman_rho because the underlying neuron
            # sets are full-MLP — Jaccard is constant, correlation undefined.
            if not row["pearson_r"]:
                continue
            out[layer] = {
                "pearson_r": float(row["pearson_r"]),
                "spearman_rho": float(row["spearman_rho"]),
                "p_pearson": float(row["p_pearson"]),
                "p_spearman": float(row["p_spearman"]),
                "n_pairs": float(row["n_pairs"]),
            }
    return out


def load_probe_weights(
    *,
    model: str,
    lang: str,
    layer: int | None = None,
    data_root: Path | None = None,
):
    """Per-(concept, layer) probe direction vectors from the V2 .npz.

    NPZ keys are formatted `{concept}_L{layer}`; values are 1-D float
    arrays of length = MLP hidden dim. Lazy import of numpy so the
    module doesn't pull numpy unless this loader is actually called.

    If `layer` is given, returns dict[concept] → ndarray for that layer.
    If `layer` is None, returns dict[(concept, layer)] → ndarray for
    every (concept, layer) pair stored in the file.
    """
    import numpy as np  # local import — numpy isn't a hard dep of io/probe

    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / f"{lang}_{model}_V2_weight_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(f"Expected probe-weights file not found: {path}")

    raw = np.load(path, allow_pickle=False)
    if layer is None:
        out_all: dict[tuple[str, int], "np.ndarray"] = {}
        for k in raw.files:
            concept, L_str = k.rsplit("_L", 1)
            out_all[(concept, int(L_str))] = raw[k]
        return out_all

    suffix = f"_L{layer}"
    out_layer: dict[str, "np.ndarray"] = {}
    for k in raw.files:
        if k.endswith(suffix):
            concept = k[: -len(suffix)]
            out_layer[concept] = raw[k]
    return out_layer
