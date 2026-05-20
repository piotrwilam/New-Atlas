"""Paper Figures 9–12: circuit size by layer per (lang, model) cell,
coloured by §6.1 flow type.

One script, four configs (one per cell of the 2×2 matrix). For each
concept at the chosen cell, plot universal-mask size vs layer as a thin
line, coloured by the concept's flow-type assignment (two_phase,
build_and_hold, late_emergence, unclassified). The shape distribution
makes the §6.1 finding visual: Qwen is dominated by `late_emergence`
with a small `two_phase` cluster (the atomicity super-cluster); DeepSeek
has a small `build_and_hold` cluster instead.

Run:
    python experiments/fig_circuit_size_by_flow_type.py --config-name paper/figure9_p_qw
    python experiments/fig_circuit_size_by_flow_type.py --config-name paper/figure10_r_qw
    python experiments/fig_circuit_size_by_flow_type.py --config-name paper/figure11_p_ds
    python experiments/fig_circuit_size_by_flow_type.py --config-name paper/figure12_r_ds
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.io import load_concept_sizes_by_layer, load_flow_type_assignments
from atlas.plotting import apply_style, plot_circuit_size_overlay


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    sizes = load_concept_sizes_by_layer(
        model=cfg.model,
        lang=cfg.lang,
        eps=cfg.extraction.eps,
        cons=cfg.extraction.cons,
        partition=cfg.partition,
        data_root=Path(cfg.data_root),
    )
    flow_types = load_flow_type_assignments(
        model=cfg.model, lang=cfg.lang, data_root=Path(cfg.data_root),
    )

    # The flow-type file is the canonical concept list (106 for Python, 75
    # for Rust). The neuron-list XLSX only contains rows for concepts that
    # passed the testability filter and had non-zero signal; concepts that
    # exist in the concept space but produced no measurable circuit are
    # absent. Reconstruct flat-zero curves for them so the figure includes
    # every classified concept and the legend counts match the paper.
    ft_set = set(flow_types.keys())
    sizes_set = set(sizes.keys())
    layer_range: list[int] = sorted({L for s in sizes.values() for L in s})
    missing = ft_set - sizes_set
    if missing:
        print(f"[info] filling {len(missing)} zero-signal concepts with "
              f"flat-zero curves over layers {layer_range[0]}..{layer_range[-1]}")
        for concept in missing:
            sizes[concept] = {L: 0 for L in layer_range}
    in_sizes_only = sizes_set - ft_set
    if in_sizes_only:
        print(f"[info] {len(in_sizes_only)} concepts have sizes but no flow type — "
              f"sample: {sorted(in_sizes_only)[:5]}")

    title = f"Circuit size curves by flow type — {cfg.lang}_{cfg.model}"
    fig = plot_circuit_size_overlay(
        sizes,
        flow_types,
        title=title,
        figsize=tuple(cfg.figsize),
    )
    fig.tight_layout()

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved figure: {out_png}")

    flow_counts = dict(Counter(flow_types.values()))
    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "lang": cfg.lang,
                "model": cfg.model,
                "n_concepts": len(flow_types),
                "flow_type_counts": flow_counts,
                "figure_path": str(out_png),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
