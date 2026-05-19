"""Paper Figure 13 (Appendix E): Python concept dendrogram, Qwen at L14.

Same load → Jaccard → linkage → plot pipeline as fig5_rust_dendrogram.py;
the cell (lang=P, model=QW) is the only difference. Kept as a separate
entry point per the one-script-per-figure convention in coding_guidelines.md.

Run:
    python experiments/fig13_python_dendrogram.py --config-name paper/figure13_dendrogram
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.analysis import pairwise_jaccard_matrix, ward_linkage_from_jaccard
from atlas.io import load_neuron_lists
from atlas.plotting import apply_style, plot_dendrogram


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    masks = load_neuron_lists(
        model=cfg.model,
        lang=cfg.lang,
        eps=cfg.extraction.eps,
        cons=cfg.extraction.cons,
        layer=cfg.layer,
        partition="concept_only",
        data_root=Path(cfg.data_root),
    )
    masks = {name: ids for name, ids in masks.items() if ids}

    names, j_matrix = pairwise_jaccard_matrix(masks)
    linkage = ward_linkage_from_jaccard(j_matrix)

    title = f"Hierarchical clustering — {cfg.lang}_{cfg.model} L{cfg.layer}"
    fig = plot_dendrogram(
        linkage,
        labels=names,
        title=title,
        color_threshold=cfg.color_threshold,
        figsize=tuple(cfg.figsize),
    )

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png)
    print(f"Saved figure: {out_png}")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "n_concepts": len(names),
                "concept_names": names,
                "figure_path": str(out_png),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
