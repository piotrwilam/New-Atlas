"""Three-way decomposition of the universal mask A against the checker mask B.

Paper §3.6: given the universal mask `A` (neurons firing for a concept's
structural role across many contexts) and the checker mask `B` (neurons
firing for the bare keyword token outside its structural role), three
disjoint partitions follow:

    concept_only = A \\ B   neurons that fire on the concept but not the bare token
    shared       = A ∩ B   neurons that fire on both
    token_only   = B \\ A   neurons that fire on the bare token but not the concept

The **concept fraction** |concept_only| / |A| quantifies how much of a
concept's circuit reflects structural understanding versus surface
token-pattern matching. It is the per-concept, per-layer metric that
underlies §4–§5 of the paper.
"""

from __future__ import annotations


def decompose_sets(
    universal: set[int],
    checker: set[int],
) -> tuple[set[int], set[int], set[int]]:
    """Three-way decomposition of universal mask A against checker mask B.

    Returns the disjoint partition `(concept_only, shared, token_only)`
    where:
        concept_only = universal \\ checker
        shared       = universal ∩ checker
        token_only   = checker \\ universal

    Pure; inputs unchanged. Symmetric in the sense that
    `decompose_sets(B, A)` returns `(token_only, shared, concept_only)`.
    """
    assert isinstance(universal, set), (
        f"universal must be a set, got {type(universal).__name__}"
    )
    assert isinstance(checker, set), (
        f"checker must be a set, got {type(checker).__name__}"
    )
    return universal - checker, universal & checker, checker - universal


def concept_fraction(
    concept_only_size: int,
    universal_size: int,
) -> float:
    """Concept fraction: |A \\ B| / |A|.

    Quantifies the fraction of a concept's universal circuit that is
    *not* explained by mere token-pattern matching. Higher values mean
    more of the circuit is structurally specific.

    Returns 0.0 when `universal_size == 0` (concept has no detectable
    circuit at this cell). This convention keeps downstream means
    well-defined when summing across many (concept, layer) cells.
    """
    assert concept_only_size >= 0, (
        f"concept_only_size must be non-negative, got {concept_only_size}"
    )
    assert universal_size >= 0, (
        f"universal_size must be non-negative, got {universal_size}"
    )
    assert concept_only_size <= universal_size, (
        f"concept_only ({concept_only_size}) cannot exceed universal "
        f"({universal_size})"
    )
    if universal_size == 0:
        return 0.0
    return concept_only_size / universal_size
