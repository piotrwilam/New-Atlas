"""Tests for atlas.io.probe — CSV + NPZ readers for §7 artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atlas.io.probe import (
    load_jaccard_cosine_correlation,
    load_probe_results,
    load_probe_weights,
)


def test_load_probe_results(tmp_path: Path) -> None:
    p = tmp_path / "P_QW_V2_probe_results.csv"
    p.write_text(
        "concept,layer,accuracy,accuracy_std,n_object,n_checker,top10_neurons\n"
        "Assert,0,0.93,0.1,50,49,\"[1, 2, 3]\"\n"
        "Assert,1,0.98,0.05,50,49,\"[4, 5, 6]\"\n"
        "Break,0,0.91,0.12,40,40,\"[7, 8]\"\n"
    )
    out = load_probe_results(model="QW", lang="P", data_root=tmp_path)
    assert set(out) == {"Assert", "Break"}
    assert out["Assert"][1]["accuracy"] == pytest.approx(0.98)
    assert out["Break"][0]["accuracy"] == pytest.approx(0.91)


def test_load_jaccard_cosine_correlation_skips_blank_rows(tmp_path: Path) -> None:
    p = tmp_path / "P_QW_V2_cosine_jaccard_correlation.csv"
    p.write_text(
        "layer,pearson_r,spearman_rho,p_pearson,p_spearman,n_pairs\n"
        "0,0.1,0.2,0.5,0.4,276\n"
        "1,0.15,0.25,0.4,0.3,276\n"
        "26,,,,,276\n"          # saturated last-layer row
        "27,,,,,276\n"
    )
    out = load_jaccard_cosine_correlation(model="QW", lang="P", data_root=tmp_path)
    assert set(out) == {0, 1}, f"got layers {set(out)}"
    assert out[0]["pearson_r"] == pytest.approx(0.1)


def test_load_probe_weights_single_layer(tmp_path: Path) -> None:
    p = tmp_path / "P_QW_V2_weight_vectors.npz"
    np.savez(
        p,
        Assert_L0=np.ones(8, dtype=np.float64),
        Assert_L1=np.zeros(8, dtype=np.float64),
        Break_L0=np.arange(8, dtype=np.float64),
    )
    layer0 = load_probe_weights(model="QW", lang="P", layer=0, data_root=tmp_path)
    assert set(layer0) == {"Assert", "Break"}
    assert layer0["Assert"].shape == (8,)
    assert np.allclose(layer0["Break"], np.arange(8))


def test_load_probe_weights_all(tmp_path: Path) -> None:
    p = tmp_path / "P_QW_V2_weight_vectors.npz"
    np.savez(
        p,
        Assert_L0=np.ones(4, dtype=np.float64),
        Assert_L1=np.zeros(4, dtype=np.float64),
    )
    out = load_probe_weights(model="QW", lang="P", data_root=tmp_path)
    assert set(out) == {("Assert", 0), ("Assert", 1)}
