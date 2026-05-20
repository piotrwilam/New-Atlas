"""Paper Figure 8: probe-cosine vs concept-only-Jaccard cross-validation.

Two panels:
  Left  — per-layer Pearson r from `*_V2_cosine_jaccard_correlation.csv`.
  Right — pairwise scatter at the focus layer (default L20):
            x = Jaccard of concept_only neuron sets
            y = Cosine of probe direction vectors
          one point per concept pair (24 ast concepts → C(24,2) = 276).

Run:
    python experiments/fig8_jaccard_cosine.py --config-name paper/figure8_jaccard_cosine
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr

from atlas.analysis import jaccard_sets
from atlas.io import (
    load_jaccard_cosine_correlation,
    load_neuron_lists,
    load_probe_weights,
)
from atlas.plotting import apply_style


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    # Left panel: per-layer Pearson r.
    per_layer = load_jaccard_cosine_correlation(
        model=cfg.model, lang=cfg.lang, data_root=Path(cfg.data_root),
    )
    layers = sorted(per_layer.keys())
    rs = [per_layer[L]["pearson_r"] for L in layers]

    # Right panel: pairwise (Jaccard, Cosine) at the focus layer.
    weights = load_probe_weights(
        model=cfg.model, lang=cfg.lang,
        layer=cfg.focus_layer, data_root=Path(cfg.data_root),
    )
    masks = load_neuron_lists(
        model=cfg.model, lang=cfg.lang,
        eps=cfg.extraction.eps, cons=cfg.extraction.cons,
        layer=cfg.focus_layer, partition="concept_only",
        data_root=Path(cfg.data_root),
    )
    shared = sorted(set(weights) & set(masks))
    assert len(shared) >= 2, f"need at least 2 concepts; got {len(shared)}"
    jacc: list[float] = []
    cos: list[float] = []
    norms = {c: weights[c] / np.linalg.norm(weights[c]) for c in shared}
    for a, b in combinations(shared, 2):
        jacc.append(jaccard_sets(masks[a], masks[b]))
        cos.append(float(np.dot(norms[a], norms[b])))
    r_focus, _ = pearsonr(jacc, cos)

    fig, axes = plt.subplots(1, 2, figsize=tuple(cfg.figsize))

    axes[0].plot(layers, rs, marker="o", color="#1f77b4", linewidth=1.6)
    axes[0].axhline(cfg.r_target, color="red", linestyle="--",
                    label=f"r = {cfg.r_target} target")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Pearson r (Jaccard vs Cosine)")
    axes[0].set_title(f"{cfg.lang}_{cfg.model}: Jaccard-Cosine Correlation")
    axes[0].legend(loc="center left")

    axes[1].scatter(jacc, cos, alpha=0.55, edgecolors="white", linewidths=0.4)
    axes[1].set_xlabel("Jaccard (neuron view)")
    axes[1].set_ylabel("Cosine (vector view)")
    axes[1].set_title(f"Layer {cfg.focus_layer} scatter (r = {r_focus:.3f})")

    fig.tight_layout()

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    print(f"L{cfg.focus_layer}: recomputed r = {r_focus:.4f}  "
          f"(CSV file says r = {per_layer[cfg.focus_layer]['pearson_r']:.4f})")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "figure_path": str(out_png),
                "focus_layer": cfg.focus_layer,
                "r_recomputed": round(r_focus, 4),
                "r_from_csv": round(per_layer[cfg.focus_layer]["pearson_r"], 4),
                "peak_layer": layers[rs.index(max(rs))],
                "peak_r": round(max(rs), 4),
                "n_pairs": len(jacc),
            },
            f, indent=2,
        )


if __name__ == "__main__":
    main()
