"""
Stage D — Pipeline Orchestrator & Parquet Export.

Drives Stage A → B → C → D for every (AST node, builtin) pair:
  1. Generate N_GENERATE prompt candidates via ASTPromptGenerator
  2. Filter to N_KEEP via PerplexityFilter
  3. Accumulate rows and write Drive checkpoints every CHECKPOINT_EVERY pairs
  4. Emit final validated_prompts.parquet + stats.json (Atlas 2 §9)
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Output schema column order (Atlas 2 §9)
_COLUMNS = [
    "ast_node", "builtin_obj", "variation_id",
    "prompt_text", "sequence_loss", "token_length", "ast_verified",
]


def run_pipeline(
    concept_pairs: list[tuple[str, str]],
    generator,
    pfilter,
    n_generate: int,
    n_keep: int,
    checkpoint_dir: str,
    checkpoint_every: int = 50,
    run_name: str = None,
    catastrophic_threshold: float = 10.0,
    mode: str = "full",
) -> pd.DataFrame:
    """Run the full Module 1 pipeline and return the resulting DataFrame.

    Parameters
    ----------
    concept_pairs:
        List of (ast_node, builtin_obj) pairs (Stage A output).
    generator:
        ASTPromptGenerator instance (Stage B).
    pfilter:
        PerplexityFilter instance (Stage C).
    n_generate:
        How many raw prompt candidates to generate per pair.
    n_keep:
        How many to keep after perplexity filtering.
    checkpoint_dir:
        Directory for checkpoint Parquet files and stats JSON.
    checkpoint_every:
        Write a checkpoint after every this many pairs processed.
    run_name:
        Human-readable name embedded in output filenames. Defaults to a
        timestamp string.
    catastrophic_threshold:
        Average loss ceiling; cells above this are entirely discarded.
    mode:
        Run mode string stored in stats JSON ("test"/"small"/"full").
    """
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    os.makedirs(checkpoint_dir, exist_ok=True)

    all_rows: list[dict] = []
    stats = {
        "run_name": run_name,
        "mode": mode,
        "n_ast": len({a for a, _ in concept_pairs}),
        "n_builtins": len({b for _, b in concept_pairs}),
        "total_pairs": len(concept_pairs),
        "n_generate": n_generate,
        "n_keep": n_keep,
        "successful_pairs": 0,
        "failed_pairs": 0,
        "empty_gen_pairs": [],
        "catastrophic_pairs": [],
        "total_prompts": 0,
    }

    for idx, (ast_node, builtin_obj) in enumerate(
        tqdm(concept_pairs, desc="Pairs")
    ):
        raw = generator.generate_batch(ast_node, builtin_obj, n=n_generate)
        if not raw:
            stats["failed_pairs"] += 1
            stats["empty_gen_pairs"].append([ast_node, builtin_obj])
            continue

        filtered = pfilter.filter_batch(
            raw,
            top_k=n_keep,
            catastrophic_threshold=catastrophic_threshold,
        )
        if not filtered:
            stats["failed_pairs"] += 1
            stats["catastrophic_pairs"].append([ast_node, builtin_obj])
            continue

        for var_id, p in enumerate(filtered):
            all_rows.append({
                "ast_node": ast_node,
                "builtin_obj": builtin_obj,
                "variation_id": var_id,
                "prompt_text": p["prompt_text"],
                "sequence_loss": p["sequence_loss"],
                "token_length": p["token_length"],
                "ast_verified": p.get("ast_verified", False),
            })
        stats["successful_pairs"] += 1
        stats["total_prompts"] += len(filtered)

        if (idx + 1) % checkpoint_every == 0:
            ckpt = pd.DataFrame(all_rows, columns=_COLUMNS)
            path = os.path.join(
                checkpoint_dir, f"{run_name}_ckpt_{idx + 1}.parquet"
            )
            ckpt.to_parquet(path, index=False)
            logger.info("Checkpoint: %s (%d rows)", path, len(ckpt))

    df = pd.DataFrame(all_rows, columns=_COLUMNS) if all_rows else pd.DataFrame(columns=_COLUMNS)

    final_path = os.path.join(
        checkpoint_dir, f"{run_name}_validated_prompts.parquet"
    )
    df.to_parquet(final_path, index=False)

    stats_path = os.path.join(checkpoint_dir, f"{run_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE: {run_name}")
    print(f"  Pairs: {stats['successful_pairs']}/{stats['total_pairs']} succeeded")
    print(f"  Prompts: {stats['total_prompts']}")
    print(f"  Output: {final_path}")
    print(f"{'=' * 60}")

    return df
