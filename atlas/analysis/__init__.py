"""Analysis: set-similarity, decomposition, hierarchical clustering, meta-circuit stats."""

from atlas.analysis.cross_model import cross_model_correlation
from atlas.analysis.decomposition import concept_fraction, decompose_sets
from atlas.analysis.jaccard import jaccard_sets, pairwise_jaccard_matrix
from atlas.analysis.meta_circuits import (
    mean_pairwise_jaccard,
    permutation_within_group_p_value,
    ward_linkage_from_jaccard,
)
from atlas.analysis.probe_alignment import pairwise_cosine_vs_jaccard

__all__ = [
    "concept_fraction",
    "cross_model_correlation",
    "decompose_sets",
    "jaccard_sets",
    "mean_pairwise_jaccard",
    "pairwise_cosine_vs_jaccard",
    "pairwise_jaccard_matrix",
    "permutation_within_group_p_value",
    "ward_linkage_from_jaccard",
]
