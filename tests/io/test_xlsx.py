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
