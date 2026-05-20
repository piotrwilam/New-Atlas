"""Test the neuron-list XLSX loader against a tiny synthetic file
written by the test itself. No external dependencies.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from atlas.io.xlsx import load_neuron_lists


def _write_synthetic_xlsx(path: Path, rows: list[tuple]) -> None:
    """Helper: write a single-sheet xlsx with the canonical column order."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([
        "object", "layer",
        "n_concept_only", "n_both", "n_token_only",
        "concept_only", "both", "token_only",
    ])
    for r in rows:
        ws.append(r)
    wb.save(path)


def test_load_rust_qw_concept_only(tmp_path: Path) -> None:
    # Synthetic part1 + part2 covering layers 0..1 (part1) and 14 (part2).
    p1 = tmp_path / "R_QW_4_neuron_list_eps0.5_cons0.8_layers_part1_both.xlsx"
    p2 = tmp_path / "R_QW_4_neuron_list_eps0.5_cons0.8_layers_part2_both.xlsx"
    _write_synthetic_xlsx(p1, [
        ("rust__Foo", 0, 0, 0, 0, "[]", "[]", "[]"),
        ("rust__Foo", 1, 2, 0, 0, "[10, 20]", "[]", "[]"),
        ("rust__Bar", 1, 1, 0, 0, "[30]", "[]", "[]"),
    ])
    _write_synthetic_xlsx(p2, [
        ("rust__Foo", 14, 3, 0, 0, "[1, 2, 3]", "[]", "[]"),
        ("rust__Bar", 14, 2, 0, 0, "[2, 4]", "[]", "[]"),
        ("rust__Baz", 14, 0, 0, 0, "[]", "[]", "[]"),
    ])

    out = load_neuron_lists(
        model="QW", lang="R", eps=0.5, cons=0.8, layer=14, data_root=tmp_path,
    )
    assert set(out) == {"Foo", "Bar", "Baz"}, f"got keys {set(out)}"
    assert out["Foo"] == {1, 2, 3}
    assert out["Bar"] == {2, 4}
    assert out["Baz"] == set()


def test_python_ast_and_builtin_prefixes_both_stripped(tmp_path: Path) -> None:
    """Python data ships with `ast__` (syntax) and `builtin__` (callables)
    prefixes; both must be stripped so leaf labels in the dendrogram match
    the paper (e.g. `len`, not `builtin__len`)."""
    p1 = tmp_path / "P_QW_4_neuron_list_eps0.5_cons0.8_layers_part1_both.xlsx"
    p2 = tmp_path / "P_QW_4_neuron_list_eps0.5_cons0.8_layers_part2_both.xlsx"
    _write_synthetic_xlsx(p1, [
        ("ast__Import",  14, 2, 0, 0, "[1, 2]", "[]", "[]"),
        ("builtin__len", 14, 2, 0, 0, "[3, 4]", "[]", "[]"),
    ])
    _write_synthetic_xlsx(p2, [
        ("ast__While",     14, 1, 0, 0, "[5]", "[]", "[]"),
        ("builtin__range", 14, 1, 0, 0, "[6]", "[]", "[]"),
    ])

    out = load_neuron_lists(
        model="QW", lang="P", eps=0.5, cons=0.8, layer=14, data_root=tmp_path,
    )
    assert set(out) == {"Import", "len", "While", "range"}, f"got keys {set(out)}"
    assert out["Import"] == {1, 2}
    assert out["len"] == {3, 4}
    assert out["range"] == {6}


def test_load_concept_sizes_by_layer_universal(tmp_path: Path) -> None:
    """`universal` partition returns concept_only + both. Used by F4."""
    from atlas.io.xlsx import load_concept_sizes_by_layer
    p1 = tmp_path / "P_QW_4_neuron_list_eps0.5_cons0.8_layers_part1_both.xlsx"
    p2 = tmp_path / "P_QW_4_neuron_list_eps0.5_cons0.8_layers_part2_both.xlsx"
    _write_synthetic_xlsx(p1, [
        ("ast__Import", 0, 3, 10, 7, "[]", "[]", "[]"),
        ("ast__Import", 1, 5, 20, 3, "[]", "[]", "[]"),
    ])
    _write_synthetic_xlsx(p2, [
        ("ast__Import", 14, 8, 100, 2, "[]", "[]", "[]"),
    ])
    sizes = load_concept_sizes_by_layer(
        model="QW", lang="P", eps=0.5, cons=0.8,
        partition="universal", data_root=tmp_path,
    )
    assert set(sizes) == {"Import"}
    assert sizes["Import"] == {0: 13, 1: 25, 14: 108}  # concept_only + both


def test_load_flow_type_assignments(tmp_path: Path) -> None:
    """Loader returns per-concept flow type at one (lang, model) cell,
    filtered from the all-cells file, with the language prefix stripped."""
    from atlas.io.xlsx import load_flow_type_assignments
    path = tmp_path / "7_E6_flow_type_assignments.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["lang", "model", "concept", "flow_type", "max_size", "peak_layer"])
    ws.append(["P", "QW", "ast__Import", "two_phase", 3500, 21])
    ws.append(["P", "QW", "builtin__len", "late_emergence", 3000, 20])
    ws.append(["P", "DS", "ast__Import", "build_and_hold", 4000, 26])  # other cell
    ws.append(["R", "QW", "rust__Use", "two_phase", 3400, 19])         # other cell
    wb.save(path)

    out = load_flow_type_assignments(model="QW", lang="P", data_root=tmp_path)
    assert out == {"Import": "two_phase", "len": "late_emergence"}


def test_load_concept_sizes_by_layer_concept_only(tmp_path: Path) -> None:
    """`concept_only` partition reads the n_concept_only column directly."""
    from atlas.io.xlsx import load_concept_sizes_by_layer
    p1 = tmp_path / "R_QW_4_neuron_list_eps0.5_cons0.8_layers_part1_both.xlsx"
    p2 = tmp_path / "R_QW_4_neuron_list_eps0.5_cons0.8_layers_part2_both.xlsx"
    _write_synthetic_xlsx(p1, [
        ("rust__Foo", 0, 3, 10, 7, "[]", "[]", "[]"),
    ])
    _write_synthetic_xlsx(p2, [
        ("rust__Foo", 14, 8, 100, 2, "[]", "[]", "[]"),
    ])
    sizes = load_concept_sizes_by_layer(
        model="QW", lang="R", eps=0.5, cons=0.8,
        partition="concept_only", data_root=tmp_path,
    )
    assert sizes == {"Foo": {0: 3, 14: 8}}


def test_invalid_model_raises(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="model must be one of"):
        load_neuron_lists(
            model="XX", lang="R", eps=0.5, cons=0.8, layer=14, data_root=tmp_path,
        )


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_neuron_lists(
            model="QW", lang="R", eps=0.5, cons=0.8, layer=14, data_root=tmp_path,
        )
