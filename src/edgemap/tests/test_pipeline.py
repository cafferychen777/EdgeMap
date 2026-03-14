import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from edgemap.config import PipelineConfig
from edgemap.pipeline import run


def _toy_adata() -> ad.AnnData:
    X = sparse.csr_matrix(
        np.array(
            [
                [10.0, 0.0],
                [0.0, 5.0],
                [3.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=["G0", "G1"]))
    adata.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return adata


def _patch_pipeline_common(monkeypatch, adata: ad.AnnData, edge_p: float) -> None:
    """Patch deterministic pipeline dependencies; vary only edge significance."""
    monkeypatch.setattr("edgemap.pipeline.resolve_resource_dir", lambda _: Path("/fake_resource"))
    monkeypatch.setattr("edgemap.pipeline.load_st", lambda *_: adata.copy())
    monkeypatch.setattr(
        "edgemap.pipeline.build_spatial_graph",
        lambda *args, **kwargs: (
            sparse.csr_matrix(np.eye(3, dtype=np.float64)),
            np.array([[0, 1], [1, 0], [2, 0]], dtype=int),
            np.array([[True, True], [True, True], [True, True]], dtype=bool),
        ),
    )
    monkeypatch.setattr("edgemap.pipeline.load_lr_pairs", lambda *_: [(["G0"], ["G1"], "G0-G1")])
    monkeypatch.setattr(
        "edgemap.pipeline.compute_communication",
        lambda *args, **kwargs: (
            np.array([[2.0], [1.0], [3.0]], dtype=np.float64),
            ["G0-G1"],
            {"G0-G1": (["G0"], ["G1"])},
        ),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.compute_edge_scores",
        lambda *args, **kwargs: (
            np.array([2.0, 2.0], dtype=np.float64),
            {"G0-G1": {"mean_comm": 2.0, "n_active_cells": 3, "pair_score": 2.2}},
        ),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.compute_node_scores",
        lambda *args, **kwargs: np.array([1.5, 1.2], dtype=np.float32),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.build_annotation_ldscores",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "SNP": ["rs1", "rs2"],
                    "ell_node": [0.2, 0.1],
                    "ell_edge": [0.3, 0.2],
                }
            ),
            {
                "n_node_genes": 2,
                "n_edge_genes": 2,
                "n_both_genes": 2,
                "gene_corr": 0.1,
                "snp_corr": 0.1,
            },
        ),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.load_sumstats",
        lambda *args, **kwargs: pd.DataFrame({"SNP": ["rs1", "rs2"], "Z": [2.0, 1.5], "N": [1000, 1000]}),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.load_baseline",
        lambda *args, **kwargs: (
            pd.DataFrame({"SNP": ["rs1", "rs2"], "base1": [0.2, 0.3]}),
            1_000_000.0,
        ),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.load_regression_weights",
        lambda *args, **kwargs: pd.DataFrame({"SNP": ["rs1", "rs2"], "L2": [1.0, 1.0]}),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.run_sldsc",
        lambda *args, **kwargs: {
            "ell_node": {"tau": 0.1, "se": 0.05, "z": 2.0, "p_onesided": 0.02},
            "ell_edge": {"tau": 0.2, "se": 0.05, "z": 4.0 if edge_p < 0.05 else 0.2, "p_onesided": edge_p},
            "intercept": 1.01,
            "n_snps": 2,
            "N_bar": 1000.0,
            "M_total": 1_000_000.0,
        },
    )


def test_pipeline_results_json_includes_per_pair_fields(monkeypatch, tmp_path):
    """results.json must include per-pair summary fields when per-pair test runs."""
    adata = _toy_adata()
    _patch_pipeline_common(monkeypatch, adata, edge_p=1e-4)
    monkeypatch.setattr(
        "edgemap.pipeline.build_per_pair_ldscores",
        lambda *args, **kwargs: ({"G0-G1": np.array([0.4, 0.2])}, ["rs1", "rs2"]),
    )
    monkeypatch.setattr(
        "edgemap.pipeline.run_per_pair_ldsc",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "pair": ["G0-G1"],
                "tau": [0.2],
                "se": [0.05],
                "z": [4.0],
                "p_onesided": [3e-5],
                "p_bonferroni": [3e-5],
            }
        ),
    )

    out_dir = tmp_path / "out"
    cfg = PipelineConfig(
        st_h5ad="fake_st.h5ad",
        gwas_sumstats="fake_sumstats.tsv",
        gwas_label="fake_trait",
        output_dir=str(out_dir),
        resource_dir="/fake_resource",
    )
    output = run(cfg)

    assert output["edge_significant"] is True
    assert output["n_pairs_tested"] == 1
    assert output["n_pairs_significant"] == 1

    results_path = out_dir / "results.json"
    assert results_path.exists()
    with open(results_path) as f:
        saved = json.load(f)

    assert saved["n_pairs_tested"] == 1
    assert saved["n_pairs_significant"] == 1
    assert (out_dir / "per_pair_sldsc.csv").exists()
    assert (out_dir / "lr_pair_stats.json").exists()


def test_pipeline_adata_direct_pass_writes_back(monkeypatch, tmp_path):
    """Passing adata directly should write node/edge scores back to adata.var and adata.uns."""
    adata = _toy_adata()
    _patch_pipeline_common(monkeypatch, adata, edge_p=0.42)

    # Patch preprocess_st to just return the copy as-is (skip normalization checks)
    monkeypatch.setattr("edgemap.pipeline.preprocess_st", lambda a, cfg: a)

    out_dir = tmp_path / "out_adata"
    cfg = PipelineConfig(
        gwas_sumstats="fake_sumstats.tsv",
        gwas_label="fake_trait",
        output_dir=str(out_dir),
        resource_dir="/fake_resource",
    )
    output = run(cfg, adata=adata)

    # Results written back to the original adata
    assert "node_score" in adata.var.columns
    assert "edge_score" in adata.var.columns
    assert "edgemap" in adata.uns
    assert adata.uns["edgemap"]["gwas_label"] == "fake_trait"
    assert output["st_data"] == "AnnData (in-memory)"


def test_pipeline_non_significant_edge_skips_per_pair(monkeypatch, tmp_path):
    """Non-significant edge tau should not trigger per-pair regression."""
    adata = _toy_adata()
    _patch_pipeline_common(monkeypatch, adata, edge_p=0.42)

    # If this is called, the branch logic is wrong.
    monkeypatch.setattr(
        "edgemap.pipeline.run_per_pair_ldsc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("per-pair should be skipped")),
    )

    out_dir = tmp_path / "out_nonsig"
    cfg = PipelineConfig(
        st_h5ad="fake_st.h5ad",
        gwas_sumstats="fake_sumstats.tsv",
        gwas_label="fake_trait",
        output_dir=str(out_dir),
        resource_dir="/fake_resource",
    )
    output = run(cfg)

    assert output["edge_significant"] is False
    assert "n_pairs_tested" not in output
    assert not (out_dir / "per_pair_sldsc.csv").exists()
