"""Paper Figure 5a/5b: Rust concept dendrogram at one (model, layer) cell.

Load → compute → plot → save. Nothing else.

Run:
    python experiments/fig5_rust_dendrogram.py --config-name paper/figure5_dendrogram
    # override the (lang, model) cell:
    python experiments/fig5_rust_dendrogram.py --config-name paper/figure5_dendrogram model=DS
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
    # Drop concepts with empty concept-only sets at this layer — they'd
    # collapse to distance 0 everywhere and dominate the dendrogram cut.
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

    # Provenance.
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
