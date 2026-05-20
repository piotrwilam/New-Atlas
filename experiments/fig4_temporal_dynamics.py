"""Paper Figure 4: atomicity-concept temporal dynamics, Qwen vs DeepSeek.

For each of the two models, load per-layer concept-only sizes for the
atomicity super-cluster (Assert, Break, Continue, Import, ImportFrom,
Pass) and render circuit-size-by-layer line plots side by side. The
shape of the two panels makes the §4.4 "How" axis claim visually:
Qwen's two_phase signature vs DeepSeek's build_and_hold.

Run:
    python experiments/fig4_temporal_dynamics.py --config-name paper/figure4_temporal_dynamics
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.io import load_concept_sizes_by_layer
from atlas.plotting import apply_style, plot_temporal_dynamics


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    concepts = list(cfg.concepts)
    panel_specs = list(cfg.panels)

    fig, axes = plt.subplots(1, len(panel_specs), figsize=tuple(cfg.figsize))
    if len(panel_specs) == 1:
        axes = [axes]

    summary: list[dict] = []
    for ax, panel in zip(axes, panel_specs):
        sizes = load_concept_sizes_by_layer(
            model=panel.model,
            lang=cfg.lang,
            eps=cfg.extraction.eps,
            cons=cfg.extraction.cons,
            partition=cfg.partition,
            data_root=Path(cfg.data_root),
        )
        highlight = [tuple(r) for r in panel.highlight_layers]
        plot_temporal_dynamics(
            sizes,
            concepts=concepts,
            title=panel.subtitle,
            highlight_layers=highlight or None,
            ax=ax,
        )
        n_layers = max(len(sizes[c]) for c in concepts)
        summary.append({
            "model": panel.model,
            "lang": cfg.lang,
            "n_layers": n_layers,
            "concepts": concepts,
            "peak_layer_per_concept": {
                c: max(sizes[c], key=lambda L: sizes[c][L]) for c in concepts
            },
            "peak_size_per_concept": {
                c: max(sizes[c].values()) for c in concepts
            },
        })

    fig.suptitle(cfg.suptitle, y=1.02, fontsize=11)
    fig.tight_layout()

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved figure: {out_png}")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {"figure_path": str(out_png), "panels": summary},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
