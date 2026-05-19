"""Read the per-concept neuron-list XLSX artifacts produced by the
decomposition stage.

Each XLSX has columns: object, layer, n_concept_only, n_both,
n_token_only, concept_only, both, token_only — where the three "_only"
list columns are stored as string-encoded Python lists of int neuron IDs.

Files are split into two layer halves (part1, part2). This module
transparently concatenates both halves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import openpyxl

from atlas.paths import DATA_ROOT

_VALID_MODELS = {"QW", "DS"}
_VALID_LANGS = {"P", "R"}
# Python data uses two prefixes — `ast__` for syntax-tree concepts (Import,
# While, For, …) and `builtin__` for callable builtins (len, range, sorted,
# …). Both are stripped on load so the dendrogram leaf labels match the
# paper figures (which show just `len`, not `builtin__len`).
_PREFIXES: dict[str, tuple[str, ...]] = {
    "P": ("ast__", "builtin__"),
    "R": ("rust__",),
}


def _strip_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    for p in prefixes:
        if name.startswith(p):
            return name[len(p):]
    return name


def load_neuron_lists(
    *,
    model: str,
    lang: str,
    eps: float,
    cons: float,
    layer: int,
    partition: str = "concept_only",
    data_root: Path | None = None,
) -> dict[str, set[int]]:
    """Load concept-only / shared / token-only neuron sets at one (model,
    lang, layer) cell from the decomposition-stage XLSX artifacts.

    Parameters
    ----------
    model      one of {"QW", "DS"} — Qwen or DeepSeek
    lang       one of {"P", "R"}   — Python or Rust
    eps        epsilon threshold used for binarisation (e.g. 0.5)
    cons       consistency threshold (e.g. 0.8)
    layer      integer layer index (0-based)
    partition  which column to return: "concept_only", "both", or "token_only"
    data_root  override the global DATA_ROOT (e.g. for tests)

    Returns
    -------
    dict mapping concept name (with the language prefix stripped — e.g.
    "Import" not "ast__Import") to a set of int neuron IDs at the given
    layer for the given partition.

    Side effects: reads two XLSX files from disk.
    """
    assert model in _VALID_MODELS, f"model must be one of {_VALID_MODELS}, got {model!r}"
    assert lang in _VALID_LANGS, f"lang must be one of {_VALID_LANGS}, got {lang!r}"
    assert partition in {"concept_only", "both", "token_only"}, (
        f"partition must be concept_only|both|token_only, got {partition!r}"
    )
    assert isinstance(layer, int) and layer >= 0, f"layer must be non-negative int, got {layer!r}"

    root = Path(data_root) if data_root is not None else DATA_ROOT
    prefixes = _PREFIXES[lang]
    col_index = {"concept_only": 5, "both": 6, "token_only": 7}[partition]

    out: dict[str, set[int]] = {}
    for part in (1, 2):
        fname = f"{lang}_{model}_4_neuron_list_eps{eps}_cons{cons}_layers_part{part}_both.xlsx"
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected neuron-list file not found: {path}")
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue  # header
            if row[1] != layer:
                continue
            obj = row[0]
            cell = row[col_index]
            try:
                ids = set(ast.literal_eval(cell)) if cell else set()
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"Could not parse neuron list for {obj!r} L{layer} in {path}: {cell!r}"
                ) from e
            out[_strip_prefix(obj, prefixes)] = ids
    return out
