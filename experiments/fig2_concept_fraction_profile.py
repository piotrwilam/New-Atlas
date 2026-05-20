"""Paper Figure 2: layer-resolved concept-fraction profiles.

Two side-by-side panels (Python | Rust); each panel shows mean
concept-fraction (|concept_only| / |universal|) per layer, per
(concept group, model) — 6 lines per panel.

Run:
    python experiments/fig2_concept_fraction_profile.py --config-name paper/figure2_concept_fraction
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.analysis import concept_fraction
from atlas.io import load_concept_groups, load_concept_sizes_by_layer
from atlas.plotting import apply_style, plot_temporal_dynamics


def _mean_concept_fraction_per_layer(
    concepts: list[str],
    concept_only: dict[str, dict[int, int]],
    universal: dict[str, dict[int, int]],
) -> dict[int, float]:
    """Average per-concept concept-fraction at each layer over `concepts`."""
    layers = sorted({L for c in concepts if c in universal for L in universal[c]})
    out: dict[int, float] = {}
    for L in layers:
        fractions = [
            concept_fraction(concept_only.get(c, {}).get(L, 0), universal[c][L])
            for c in concepts if c in universal and L in universal[c]
        ]
        out[L] = sum(fractions) / len(fractions) if fractions else 0.0
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    fig, axes = plt.subplots(1, len(cfg.panels), figsize=tuple(cfg.figsize))
    if len(cfg.panels) == 1:
        axes = [axes]

    summary: list[dict] = []
    for ax, panel in zip(axes, cfg.panels):
        lang = panel.lang
        groups = list(panel.groups)
        models = list(panel.models)
        # Per-(model, group) per-layer mean concept fraction.
        per_line: dict[str, dict[int, float]] = {}
        peak_per_line: dict[str, tuple[int, float]] = {}
        for group in groups:
            for model in models:
                co = load_concept_sizes_by_layer(
                    model=model, lang=lang,
                    eps=cfg.extraction.eps, cons=cfg.extraction.cons,
                    partition="concept_only",
                    data_root=Path(cfg.data_root),
                )
                un = load_concept_sizes_by_layer(
                    model=model, lang=lang,
                    eps=cfg.extraction.eps, cons=cfg.extraction.cons,
                    partition="universal",
                    data_root=Path(cfg.data_root),
                )
                concept_to_group = load_concept_groups(
                    model=model, lang=lang,
                    eps=cfg.extraction.eps, cons=cfg.extraction.cons,
                    data_root=Path(cfg.data_root),
                )
                members = [c for c, g in concept_to_group.items() if g == group]
                profile = _mean_concept_fraction_per_layer(members, co, un)
                label = f"{group} {model}"
                per_line[label] = profile
                if profile:
                    peak_L = max(profile, key=lambda L: profile[L])
                    peak_per_line[label] = (peak_L, profile[peak_L])

        # Order: model outer, group inner — matches the v1 paper legend
        # (Builtin Q, Modular Q, Non-modular Q, Builtin DS, Modular DS, Non-modular DS).
        line_order = [f"{g} {m}" for m in models for g in groups]
        plot_temporal_dynamics(
            per_line,
            concepts=line_order,
            title=panel.title,
            ylabel="Concept fraction",
            ax=ax,
        )
        ax.set_ylim(bottom=0)
        summary.append({
            "lang": lang,
            "groups": groups,
            "models": models,
            "peak_per_line": {k: {"layer": v[0], "value": round(v[1], 4)}
                              for k, v in peak_per_line.items()},
        })

    fig.tight_layout()

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved figure: {out_png}")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump({"figure_path": str(out_png), "panels": summary}, f, indent=2)


if __name__ == "__main__":
    main()
