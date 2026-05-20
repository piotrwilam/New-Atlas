"""Paper Figure 7 (panel a): linear-probe accuracy by layer.

Per-layer mean probe accuracy across all tested concepts in P_QW, with a
±1 std envelope. Chance line at 0.5. Reproduces the §7.3 claim that the
probe achieves 97–100% accuracy at every layer.

Run:
    python experiments/fig7_probe_accuracy.py --config-name paper/figure7_probe_accuracy
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.io import load_probe_results
from atlas.plotting import apply_style


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    per_concept = load_probe_results(
        model=cfg.model, lang=cfg.lang, data_root=Path(cfg.data_root),
    )

    # Aggregate to per-layer mean / std across concepts.
    layers = sorted({L for layers_d in per_concept.values() for L in layers_d})
    mean_per_layer: dict[int, float] = {}
    std_per_layer: dict[int, float] = {}
    for L in layers:
        accs = [d[L]["accuracy"] for d in per_concept.values() if L in d]
        mean_per_layer[L] = statistics.fmean(accs) if accs else 0.0
        std_per_layer[L] = statistics.pstdev(accs) if len(accs) > 1 else 0.0

    fig, ax = plt.subplots(figsize=tuple(cfg.figsize))
    xs = layers
    ys = [mean_per_layer[L] for L in xs]
    lo = [mean_per_layer[L] - std_per_layer[L] for L in xs]
    hi = [mean_per_layer[L] + std_per_layer[L] for L in xs]
    ax.fill_between(xs, lo, hi, alpha=0.2, color="#1f77b4")
    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.6)
    ax.axhline(cfg.chance, color="gray", linestyle="--", label="chance")
    ax.set_ylim(cfg.ymin, cfg.ymax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe 1 Accuracy (object vs checker)")
    ax.set_title(f"{cfg.lang}_{cfg.model}: Probe accuracy by layer")
    ax.legend(loc="center right")
    fig.tight_layout()

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    print(f"min accuracy = {min(ys):.4f} at L{xs[ys.index(min(ys))]}; "
          f"max accuracy = {max(ys):.4f} at L{xs[ys.index(max(ys))]}")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "figure_path": str(out_png),
                "n_concepts": len(per_concept),
                "mean_per_layer": {L: round(mean_per_layer[L], 4) for L in xs},
                "std_per_layer": {L: round(std_per_layer[L], 4) for L in xs},
            },
            f, indent=2,
        )


if __name__ == "__main__":
    main()
