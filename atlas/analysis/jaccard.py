"""Jaccard similarity for set-valued neuron masks.

This module operates on Python `set[int]` values, which is what the
decomposition-stage XLSX loader returns. There is a separate
array-based Jaccard in `src/module2/metrics.py` for boolean-mask inputs
from the binarisation stage; the two are kept distinct because their
inputs are not interchangeable.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def jaccard_sets(a: set[int], b: set[int]) -> float:
    """Jaccard similarity for two integer sets.

    Returns |A ∩ B| / |A ∪ B|. When both sets are empty, returns 0.0
    by convention (matches the array-based jaccard_similarity in
    src/module2/metrics.py).

    Pure function. No mutation.
    """
    assert isinstance(a, set), f"a must be a set, got {type(a).__name__}"
    assert isinstance(b, set), f"b must be a set, got {type(b).__name__}"

    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def pairwise_jaccard_matrix(
    masks: dict[str, set[int]],
) -> tuple[list[str], np.ndarray]:
    """Compute the full pairwise Jaccard similarity matrix.

    Parameters
    ----------
    masks  dict mapping name → set of neuron IDs. All sets are at the
           same (lang, model, layer) cell.

    Returns
    -------
    (names, matrix) where names is the sorted list of keys (deterministic
    order) and matrix is an (n, n) symmetric float array with diagonal = 1
    when the set is non-empty, else 0 (since Jaccard is undefined on the
    empty set; we use the same convention as jaccard_sets).

    Pure. No mutation of inputs.
    """
    assert isinstance(masks, dict), f"masks must be a dict, got {type(masks).__name__}"
    assert all(isinstance(v, set) for v in masks.values()), (
        "all values in masks must be sets"
    )

    names = sorted(masks.keys())
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, j in combinations(range(n), 2):
        sim = jaccard_sets(masks[names[i]], masks[names[j]])
        matrix[i, j] = sim
        matrix[j, i] = sim
    # Diagonal: a set is identical to itself iff non-empty.
    for i, name in enumerate(names):
        matrix[i, i] = 1.0 if masks[name] else 0.0
    return names, matrix
