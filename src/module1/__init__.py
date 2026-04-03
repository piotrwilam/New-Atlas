"""
Module 1 — Programmatic Prompt Generation & Comprehension Filtering.

Pipeline: Concept Matrix → Variance Engine → Perplexity Filter → Parquet Export
"""

from .concept_matrix import get_ast_nodes, get_builtin_objects, generate_concept_matrix
from .variance_schema import DOMAINS, DOMAIN_KEYS, WRAPPER_TYPES, WRAPPER_WEIGHTS, PADDING_BEFORE, PADDING_AFTER
from .generators import ASTPromptGenerator
from .filters import PerplexityFilter
from .pipeline import run_pipeline

__all__ = [
    "get_ast_nodes",
    "get_builtin_objects",
    "generate_concept_matrix",
    "DOMAINS",
    "DOMAIN_KEYS",
    "WRAPPER_TYPES",
    "WRAPPER_WEIGHTS",
    "PADDING_BEFORE",
    "PADDING_AFTER",
    "ASTPromptGenerator",
    "PerplexityFilter",
    "run_pipeline",
]
