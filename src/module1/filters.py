"""
Stage C — Perplexity Filter.

Scores each prompt with a single forward pass through the CSP model, sorts
ascending by cross-entropy loss, and keeps the top-k lowest-loss prompts.
Cells whose average loss exceeds the catastrophic threshold are discarded
entirely (Atlas 2 §2).
"""

import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class PerplexityFilter:
    """Load a causal LM and filter prompts by cross-entropy loss."""

    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s...", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded.")

    def score_prompt(self, prompt_text: str) -> tuple[float, int]:
        """Return (cross_entropy_loss, token_length) for a single prompt."""
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            token_length = inputs["input_ids"].shape[1]
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        return loss, token_length

    def filter_batch(
        self,
        prompts: list[dict],
        top_k: int = 100,
        catastrophic_threshold: float = 10.0,
    ) -> list[dict]:
        """Score *prompts*, discard catastrophic cells, return sorted top-k.

        Each dict in *prompts* must have a ``"prompt_text"`` key.
        The dicts are mutated in-place to add ``"sequence_loss"`` and
        ``"token_length"`` keys before being returned.
        """
        if not prompts:
            return []

        for p in prompts:
            loss, tok_len = self.score_prompt(p["prompt_text"])
            p["sequence_loss"] = loss
            p["token_length"] = tok_len

        avg_loss = float(np.mean([p["sequence_loss"] for p in prompts]))
        if avg_loss > catastrophic_threshold:
            logger.warning(
                "Avg loss %.2f > %.1f. Discarding cell.", avg_loss, catastrophic_threshold
            )
            return []

        prompts.sort(key=lambda x: x["sequence_loss"])
        return prompts[:top_k]
