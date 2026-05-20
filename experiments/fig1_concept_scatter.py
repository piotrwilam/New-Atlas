"""Paper Figure 1a/1b: per-concept Qwen vs DeepSeek mean concept-fraction.

One scatter panel per language. Each point is one concept; x =
mean concept-fraction under Qwen, y = under DeepSeek. The y=x line is
drawn for reference. Title shows Spearman rho between the two models'
rankings.

Run:
    python experiments/fig1_concept_scatter.py --config-name paper/figure1_concept_scatter
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.analysis import cross_model_correlation
from atlas.io import load_concept_aggregates
from atlas.plotting import apply_style


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    summary: list[dict] = []

    for panel in cfg.panels:
        lang = panel.lang
        qw = load_concept_aggregates(
            model="QW", lang=lang,
            eps=cfg.extraction.eps, cons=cfg.extraction.cons,
            data_root=Path(cfg.data_root),
        )
        ds = load_concept_aggregates(
            model="DS", lang=lang,
            eps=cfg.extraction.eps, cons=cfg.extraction.cons,
            data_root=Path(cfg.data_root),
        )
        shared = sorted(set(qw) & set(ds))
        assert shared, f"no overlapping concepts between QW and DS for lang={lang}"
        x = [qw[c]["mean_cf"] for c in shared]
        y = [ds[c]["mean_cf"] for c in shared]
        rho, p = cross_model_correlation(x, y, method="spearman")

        fig, ax = plt.subplots(figsize=tuple(cfg.figsize))
        ax.scatter(x, y, alpha=0.65, edgecolors="white", linewidths=0.5)
        ax.plot([0, cfg.axis_max], [0, cfg.axis_max],
                color="red", linestyle="--", alpha=0.7, label="y=x")
        ax.set_xlim(-0.005, cfg.axis_max)
        ax.set_ylim(-0.005, cfg.axis_max)
        ax.set_xlabel("Concept fraction (QW)")
        ax.set_ylabel("Concept fraction (DS)")
        ax.set_title(f"{panel.title_prefix}: Concept fraction QW vs DS (rho={rho:.3f})")
        ax.legend(loc="upper left")
        fig.tight_layout()

        out_png = out_dir / f"{panel.name}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {panel.name}: rho={rho:.4f} p={p:.2e} (n={len(shared)})")

        summary.append({
            "lang": lang, "n_concepts": len(shared),
            "spearman_rho": round(rho, 4), "p_value": float(p),
            "figure_path": str(out_png),
        })

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump({"panels": summary}, f, indent=2)


if __name__ == "__main__":
    main()
