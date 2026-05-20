import numpy as np


def jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard = |A ∩ B| / |A ∪ B|. Returns 0.0 if both empty."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def jaccard_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Ockham Index: O_I = 1 - Jaccard. 0 = identical, 1 = disjoint."""
    return 1.0 - jaccard_similarity(mask_a, mask_b)


def entanglement_index(pair_mask: np.ndarray, universal_ast_mask: np.ndarray,
                        universal_builtin_mask: np.ndarray) -> float:
    """
    E_I = fraction of pair's neurons NOT explained by the union of
    its Universal AST and Universal Builtin components.

    E_I = 0 -> perfectly compositional (pair = union of parts)
    E_I = 1 -> entirely unique (pair shares nothing with universals)

    Formula: E_I(pair) = |pair_mask \\ (universal_ast ∪ universal_builtin)| / |pair_mask|
    """
    expected_union = np.logical_or(universal_ast_mask, universal_builtin_mask)
    unexplained = np.logical_and(pair_mask, ~expected_union).sum()
    pair_size = pair_mask.sum()
    if pair_size == 0:
        return 0.0
    return float(unexplained / pair_size)


def compute_jaccard_matrix(masks: dict, layer: int) -> np.ndarray:
    """
    Pairwise Jaccard similarity for a set of named masks at one layer.

    Args:
        masks: dict mapping name -> {layer_id: bool_mask}
        layer: which layer to compute for

    Returns:
        (names, square_matrix) where names is the sorted list of keys
    """
    names = sorted(masks.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        mi = masks[names[i]].get(layer)
        if mi is None:
            continue
        for j in range(i, n):
            mj = masks[names[j]].get(layer)
            if mj is None:
                continue
            sim = jaccard_similarity(mi, mj)
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix
