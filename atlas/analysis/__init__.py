"""Analysis: set decomposition, similarity, clustering, flow types, cross-language sharing.

All functions here are pure compute on artifacts already loaded by
`atlas.io`. No I/O, no plotting, no model dependencies.
"""

from atlas.analysis.cross_language import (
    EQUIVALENCE_CLASSES,
    cross_language_sharing_fraction,
    pool_neurons,
)
from atlas.analysis.cross_model import cross_model_correlation
from atlas.analysis.decomposition import concept_fraction, decompose_sets
from atlas.analysis.flow_types import (
    classify_all_flow_types,
    classify_flow_type,
)
from atlas.analysis.jaccard import jaccard_sets, pairwise_jaccard_matrix
from atlas.analysis.meta_circuits import (
    cut_dendrogram_at_k_clusters,
    mean_pairwise_jaccard,
    permutation_within_group_p_value,
    ward_linkage_from_jaccard,
)
from atlas.analysis.probe_alignment import pairwise_cosine_vs_jaccard

__all__ = [
    "EQUIVALENCE_CLASSES",
    "classify_all_flow_types",
    "classify_flow_type",
    "concept_fraction",
    "cross_language_sharing_fraction",
    "cross_model_correlation",
    "cut_dendrogram_at_k_clusters",
    "decompose_sets",
    "jaccard_sets",
    "mean_pairwise_jaccard",
    "pairwise_cosine_vs_jaccard",
    "pairwise_jaccard_matrix",
    "permutation_within_group_p_value",
    "pool_neurons",
    "ward_linkage_from_jaccard",
]
