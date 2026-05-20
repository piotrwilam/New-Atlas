"""Paper Figure 6: validation of the §6.2 four-cluster claim.

For each of the four hypothesised Rust meta-circuits at (lang=R, model=QW,
L14, eps=0.5, cons=0.8), run a within-group permutation test on mean
pairwise Jaccard against same-size random draws from the universe of Rust
concepts. Save the per-group results as both CSV (matches the schema
used in `original_assets/validation/F6_four_cluster_test.csv`) and JSON,
plus a bar chart with null-band overlay and significance annotations.

Run:
    python experiments/fig6_four_cluster_test.py --config-name paper/figure6_four_cluster
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atlas.analysis import permutation_within_group_p_value
from atlas.io import load_neuron_lists
from atlas.plotting import apply_style, plot_group_coherence


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    # Load ALL concepts at this cell, including those whose concept-only
    # set is empty at L14. The permutation null samples from the full
    # universe — empty masks contribute J=0 to any pair they're in, which
    # is the right behaviour: "is this group more cohesive than a random
    # 9-tuple drawn from the concept pool" requires the pool to include
    # every concept the paper claims exists, not just the non-empty ones.
    masks = load_neuron_lists(
        model=cfg.model,
        lang=cfg.lang,
        eps=cfg.extraction.eps,
        cons=cfg.extraction.cons,
        layer=cfg.layer,
        partition="concept_only",
        data_root=Path(cfg.data_root),
    )

    groups: dict[str, list[str]] = {k: list(v) for k, v in cfg.groups.items()}

    results: list[dict] = []
    for group_name, members in groups.items():
        missing = [m for m in members if m not in masks]
        assert not missing, (
            f"group {group_name!r} has members absent at "
            f"{cfg.lang}_{cfg.model} L{cfg.layer}: {missing}"
        )
        stats = permutation_within_group_p_value(
            members, masks, n_perm=cfg.n_perm, seed=cfg.seed,
        )
        results.append({
            "group": group_name,
            "n": len(members),
            "observed_J": round(stats["observed"], 4),
            "null_mean": round(stats["null_mean"], 4),
            "null_p95": round(stats["null_p95"], 4),
            "p_value": round(stats["p_value"], 4),
            "pass_at_0.05": "yes" if stats["p_value"] < 0.05 else "no",
        })
        print(
            f"{group_name:<24} n={len(members)}  J={stats['observed']:.4f}  "
            f"null_mean={stats['null_mean']:.4f}  p={stats['p_value']:.4f}"
        )

    out_dir = Path(HydraConfig.get().runtime.output_dir)

    csv_path = out_dir / f"{cfg.figure.name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    plot_results = [
        {
            "name": r["group"].split("_", 1)[1].replace("_", " "),
            "observed": r["observed_J"],
            "null_mean": r["null_mean"],
            "null_p95": r["null_p95"],
            "p_value": r["p_value"],
        }
        for r in results
    ]
    fig = plot_group_coherence(
        plot_results,
        title=f"Within-group Jaccard cohesion — {cfg.lang}_{cfg.model} L{cfg.layer}",
    )
    png_path = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(png_path)
    print(f"Saved figure: {png_path}")
    print(f"Saved CSV:    {csv_path}")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "n_groups": len(results),
                "groups": [r["group"] for r in results],
                "results": results,
                "figure_path": str(png_path),
                "csv_path": str(csv_path),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
