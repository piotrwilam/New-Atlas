import logging
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .marginalization import UniversalModuleComputer
from .metrics import compute_jaccard_matrix
from .io_utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Module2Pipeline:
    """
    Pair-by-pair processing with checkpoints.

    Orchestrates the full Module 2 extraction:
    1. Load Parquet, group by (ast_node, builtin_obj)
    2. For each pair: extract -> binarize -> store Pair Representation
    3. Checkpoint every N pairs
    4. After all pairs: compute Universal Modules + Jaccard matrices
    """

    def __init__(self, extractor, builder, parquet_path: str,
                 checkpoint_dir: str, checkpoint_every: int = 200):
        self.extractor = extractor
        self.builder = builder
        self.parquet_path = parquet_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every

    def run(self, resume_from_checkpoint: str = None):
        """
        Execute the full pipeline.

        Args:
            resume_from_checkpoint: path to a checkpoint file to resume from

        Returns:
            (pair_masks, universal_masks, metrics, stats_df)
        """
        df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(df)} prompts, "
                    f"{df.groupby(['ast_node','builtin_obj']).ngroups} pairs")

        grouped = df.groupby(["ast_node", "builtin_obj"])
        all_pairs = list(grouped.groups.keys())

        pair_masks = {}
        start_idx = 0

        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            pair_masks, pairs_processed = load_checkpoint(resume_from_checkpoint)
            start_idx = pairs_processed
            logger.info(f"Resumed from checkpoint: {pairs_processed} pairs done")

        pair_stats = []
        pairs_to_process = all_pairs[start_idx:]

        for idx, (ast_n, blt_o) in enumerate(
            tqdm(pairs_to_process, desc="Extracting pairs")
        ):
            group = grouped.get_group((ast_n, blt_o))
            prompts = group["prompt_text"].tolist()

            masks = self.builder.build(self.extractor, prompts)
            pair_masks[(ast_n, blt_o)] = masks

            for lid, m in masks.items():
                pair_stats.append({
                    "ast_node": ast_n, "builtin_obj": blt_o,
                    "layer": lid, "circuit_size": int(m.sum()),
                    "n_variations": len(prompts),
                })

            actual_idx = idx + start_idx + 1
            if actual_idx % self.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f"module2_ckpt_{actual_idx}.pkl"
                )
                save_checkpoint(ckpt_path, pair_masks, actual_idx)
                logger.info(f"Checkpoint: {ckpt_path} ({actual_idx} pairs)")

        # Compute Universal Modules
        ast_nodes = sorted(set(a for a, _ in pair_masks.keys()))
        builtin_objs = sorted(set(b for _, b in pair_masks.keys()))

        computer = UniversalModuleComputer()
        universal_masks = computer.compute_all(pair_masks, ast_nodes, builtin_objs)

        # Determine n_layers
        n_layers = 0
        if pair_masks:
            first_val = next(iter(pair_masks.values()))
            if first_val:
                n_layers = max(first_val.keys()) + 1

        # Jaccard matrices at representative layer
        rep_layer = min(4, n_layers - 1) if n_layers > 0 else 0
        metrics = {}

        if universal_masks.get("ast"):
            metrics["jaccard_ast_matrix"] = compute_jaccard_matrix(
                universal_masks["ast"], rep_layer
            )
            metrics["ast_names"] = sorted(universal_masks["ast"].keys())

        if universal_masks.get("builtin"):
            metrics["jaccard_builtin_matrix"] = compute_jaccard_matrix(
                universal_masks["builtin"], rep_layer
            )
            metrics["builtin_names"] = sorted(universal_masks["builtin"].keys())

        stats_df = pd.DataFrame(pair_stats)
        logger.info(f"Pipeline complete: {len(pair_masks)} pairs processed")
        return pair_masks, universal_masks, metrics, stats_df
