"""Paper Figure 3: cross-language neuron-sharing by equivalence class.

Grouped bar chart, one bar per (model, equivalence_class). Bar height
is the mean sharing_fraction across all layers in the data file. A
horizontal dashed line at H6_threshold=0.10 marks the §5.3 significance
floor.

Run:
    python experiments/fig3_cross_language_sharing.py --config-name paper/figure3_cross_language_sharing
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.io import load_cross_language_sharing
from atlas.plotting import apply_style


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    raw = load_cross_language_sharing(data_root=Path(cfg.data_root))
    classes = sorted({c for (_m, c) in raw.keys()})
    models = list(cfg.models)

    # Per-(model, class) mean across all layers in the file.
    means: dict[tuple[str, str], float] = {}
    for m in models:
        for c in classes:
            per_layer = raw.get((m, c), {})
            means[(m, c)] = statistics.fmean(per_layer.values()) if per_layer else 0.0

    fig, ax = plt.subplots(figsize=tuple(cfg.figsize))
    n_classes = len(classes)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_classes)
    for i, m in enumerate(models):
        offsets = x + (i - (n_models - 1) / 2) * width
        heights = [means[(m, c)] for c in classes]
        ax.bar(offsets, heights, width=width, label=m)
    ax.axhline(cfg.h6_threshold, color="red", linestyle="--",
               label=f"H6 threshold ({int(cfg.h6_threshold*100)}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("Mean sharing fraction")
    ax.set_xlabel("equivalence_class")
    ax.set_title("Cross-language sharing by abstract concept")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved figure: {out_png}")

    # Print the §5.3 ratio: mean(DS bars) / mean(QW bars).
    if "DS" in models and "QW" in models:
        ds_overall = statistics.fmean(means[("DS", c)] for c in classes)
        qw_overall = statistics.fmean(means[("QW", c)] for c in classes)
        ratio = ds_overall / qw_overall if qw_overall > 0 else float("nan")
        print(f"DS mean={ds_overall:.4f}, QW mean={qw_overall:.4f}, ratio={ratio:.3f}x")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "figure_path": str(out_png),
                "classes": classes,
                "means": {f"{m}_{c}": round(means[(m, c)], 4)
                          for m in models for c in classes},
            },
            f, indent=2,
        )


if __name__ == "__main__":
    main()
