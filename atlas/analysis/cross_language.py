"""Cross-language neuron sharing across equivalence classes.

Paper §5.3: do the same abstract concepts (e.g. "Iteration" — Python
`for/while` and Rust `for/while/loop`) share neurons across Python and
Rust within a single model? The methodology pools concept-only neurons
across the language-specific members of an equivalence class and
measures the intersection over the smaller pool.

The default `EQUIVALENCE_CLASSES` mapping reproduces the §5.3 result:
DS mean-of-means sharing fraction across all 9 classes = 1.949× the
Qwen value (the abstract claims "2.3×" — see CLAUDE.md for the
discrepancy). The mapping is exposed as a module constant so callers
can substitute their own groupings (e.g. extending to additional
languages).
"""

from __future__ import annotations


# Default equivalence-class definitions used by §5.3 / Figure 3.
# Names are stripped (concepts come from atlas.io loaders without
# the `ast__` / `rust__` / `builtin__` prefixes).
EQUIVALENCE_CLASSES: dict[str, dict[str, list[str]]] = {
    "Iteration":      {"P": ["For", "While"],
                       "R": ["For", "Loop", "While"]},
    "Binding":        {"P": ["Assign"],
                       "R": ["Let", "LetMut"]},
    "Branching":      {"P": ["If", "IfExp"],
                       "R": ["If", "Match"]},
    "Function def":   {"P": ["FunctionDef", "Lambda"],
                       "R": ["Fn", "Closure"]},
    "Error handling": {"P": ["Try", "Raise"],
                       "R": ["QuestionMark"]},
    "Module import":  {"P": ["Import", "ImportFrom"],
                       "R": ["Use", "Mod"]},
    "Loop control":   {"P": ["Break", "Continue"],
                       "R": ["Break", "Continue"]},
    "Return":         {"P": ["Return"],
                       "R": ["Return"]},
    "Type def":       {"P": ["ClassDef"],
                       "R": ["Struct", "Enum"]},
}


def pool_neurons(
    masks: dict[str, set[int]],
    members: list[str],
) -> set[int]:
    """Union of concept-only neuron sets across a list of member concepts.

    Members absent from `masks` are silently skipped. Useful for pooling
    across the multiple language-specific members of an equivalence
    class (e.g. Rust `Iteration` = `For` ∪ `Loop` ∪ `While`).
    """
    assert isinstance(masks, dict), f"masks must be a dict, got {type(masks).__name__}"
    pool: set[int] = set()
    for m in members:
        s = masks.get(m)
        if s is not None:
            pool |= s
    return pool


def cross_language_sharing_fraction(
    py_pool: set[int],
    rs_pool: set[int],
) -> float:
    """Sharing fraction = |py_pool ∩ rs_pool| / min(|py_pool|, |rs_pool|).

    Convention: 0.0 when either pool is empty (no neurons available to share).
    Normalising by the smaller pool keeps the metric in [0, 1] even when
    the two languages have very different total mask sizes at this layer.
    """
    assert isinstance(py_pool, set), f"py_pool must be a set, got {type(py_pool).__name__}"
    assert isinstance(rs_pool, set), f"rs_pool must be a set, got {type(rs_pool).__name__}"
    min_pool = min(len(py_pool), len(rs_pool))
    if min_pool == 0:
        return 0.0
    return len(py_pool & rs_pool) / min_pool
