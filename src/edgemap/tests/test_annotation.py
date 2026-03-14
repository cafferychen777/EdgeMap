import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse as sp

import edgemap.annotation as ann
from edgemap.annotation import build_annotation_ldscores, build_per_pair_ldscores


def test_build_annotation_ldscores_aligns_genes_and_computes_diagnostics(monkeypatch):
    # Dense matrix path exercises CSR conversion branch in _get_snp_gene_weights.
    wm = ad.AnnData(
        X=np.array(
            [
                [1.0, 0.0, 2.0],  # rs1
                [0.0, 1.0, 1.0],  # rs2
            ],
            dtype=np.float64,
        )
    )
    wm.obs_names = ["rs1", "rs2"]
    wm.var_names = ["G1", "G2", "G3"]

    ann._weight_cache.clear()
    monkeypatch.setattr("edgemap.annotation.resolve_resource_dir", lambda _: Path("/fake_resource"))
    monkeypatch.setattr("edgemap.annotation.ad.read_h5ad", lambda _: wm)

    node_scores = pd.Series({"G1": 2.0, "G2": 1.0})  # G3 absent -> filled 0
    edge_scores = pd.Series({"G2": 3.0, "G3": 2.0})  # G1 absent -> filled 0

    annot_ld, diag = build_annotation_ldscores(node_scores, edge_scores, "/fake_resource")

    assert list(annot_ld.columns) == ["SNP", "ell_node", "ell_edge"]
    assert list(annot_ld["SNP"]) == ["rs1", "rs2"]
    assert np.all(annot_ld["ell_node"] >= 0)
    assert np.all(annot_ld["ell_edge"] >= 0)

    assert diag["n_node_genes"] == 2
    assert diag["n_edge_genes"] == 2
    assert diag["n_both_genes"] == 1
    assert -1.0 <= diag["gene_corr"] <= 1.0
    assert -1.0 <= diag["snp_corr"] <= 1.0


def test_build_annotation_ldscores_accepts_sparse_weight_matrix(monkeypatch):
    # Sparse matrix path exercises W = W.tocsr() branch.
    wm = ad.AnnData(
        X=sp.csc_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )
    )
    wm.obs_names = ["rs1", "rs2"]
    wm.var_names = ["G1", "G2"]

    ann._weight_cache.clear()
    monkeypatch.setattr("edgemap.annotation.resolve_resource_dir", lambda _: Path("/fake_resource_sparse"))
    monkeypatch.setattr("edgemap.annotation.ad.read_h5ad", lambda _: wm)

    node_scores = pd.Series({"G1": 2.0})
    edge_scores = pd.Series({"G2": 3.0})
    annot_ld, _ = build_annotation_ldscores(node_scores, edge_scores, "/fake_resource_sparse")

    assert list(annot_ld["SNP"]) == ["rs1", "rs2"]
    assert annot_ld["ell_node"].iloc[0] > 0
    assert annot_ld["ell_edge"].iloc[1] > 0


def test_build_per_pair_ldscores_returns_empty_when_no_positive_scores(monkeypatch):
    W = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    ann._weight_cache.clear()
    ann._weight_cache["__test__"] = (W, ["rs1", "rs2"], ["G1", "G2"])

    try:
        # Bypass filesystem resolution and directly hit the cache key.
        monkeypatch.setattr("edgemap.annotation.resolve_resource_dir", lambda _: "__test__")
        pair_ld, snps = build_per_pair_ldscores(
            pair_names=["G1-G2"],
            pair_genes={"G1-G2": (["G1"], ["G2"])},
            pair_scores={"G1-G2": 0.0},
            resource_dir="__test__",
        )
    finally:
        ann._weight_cache.clear()

    assert pair_ld == {}
    assert snps == ["rs1", "rs2"]


def test_annotation_weights_are_cached_per_resource(monkeypatch):
    wm = ad.AnnData(
        X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    )
    wm.obs_names = ["rs1", "rs2"]
    wm.var_names = ["G1", "G2"]

    ann._weight_cache.clear()
    calls = {"n": 0}

    def fake_read(_):
        calls["n"] += 1
        return wm

    monkeypatch.setattr("edgemap.annotation.resolve_resource_dir", lambda _: Path("/cache_test"))
    monkeypatch.setattr("edgemap.annotation.ad.read_h5ad", fake_read)

    node_scores = pd.Series({"G1": 1.0})
    edge_scores = pd.Series({"G2": 1.0})
    build_annotation_ldscores(node_scores, edge_scores, "/cache_test")
    build_annotation_ldscores(node_scores, edge_scores, "/cache_test")

    assert calls["n"] == 1
