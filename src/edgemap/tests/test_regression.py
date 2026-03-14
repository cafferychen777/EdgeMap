import io
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from edgemap.config import RegressionConfig
from edgemap.regression import (
    _block_jackknife,
    load_baseline,
    load_regression_weights,
    load_sumstats,
    run_per_pair_ldsc,
    run_sldsc,
)


def _toy_regression_inputs():
    sumstats = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "Z": [2.0, 1.8, 1.5, 2.2, 1.2],
            "N": [1000, 1000, 1000, 1000, 1000],
        }
    )
    baseline = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "base1": [0.4, 0.2, 0.3, 0.5, 0.1],
            "base2": [0.1, 0.3, 0.2, 0.2, 0.4],
        }
    )
    annot_ld = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "ell_node": [0.2, 0.1, 0.3, 0.4, 0.1],
            "ell_edge": [0.3, 0.2, 0.1, 0.4, 0.2],
        }
    )
    w_ld = pd.DataFrame({"SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"], "L2": [1.0] * 5})
    return sumstats, baseline, annot_ld, w_ld


def test_load_sumstats_validates_and_filters(tmp_path):
    path = tmp_path / "sumstats.tsv"
    df = pd.DataFrame(
        {
            "SNP": ["rs1", "rs1", "rs2", "rs3", "rs4"],
            "Z": [2.0, 2.0, np.nan, np.inf, 100.0],
            "N": [1000, 1000, 1000, 1000, 1000],
        }
    )
    df.to_csv(path, sep="\t", index=False)

    cfg = RegressionConfig(chisq_max_factor=0.001, chisq_max_floor=5.0)
    out = load_sumstats(str(path), cfg)

    # rs4 is removed by chisq threshold (100^2 > 5), rs2/rs3 invalid, rs1 deduplicated.
    assert list(out["SNP"]) == ["rs1"]
    assert list(out.columns) == ["SNP", "Z", "N"]


def test_load_sumstats_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.tsv"
    pd.DataFrame({"SNP": ["rs1"], "Z": [1.2]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_sumstats(str(path), RegressionConfig())


def test_block_jackknife_handles_singular_matrix():
    # Collinear columns force XtX singular; function should fall back to lstsq.
    Xw = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    yw = np.array([1.0, 2.0, 3.0, 4.0])

    beta, se = _block_jackknife(Xw, yw, n_blocks=3)
    assert beta.shape == (2,)
    assert se.shape == (2,)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(se))


def test_run_sldsc_returns_expected_structure():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=3)
    out = run_sldsc(sumstats, baseline, annot_ld, w_ld, M_total=1_000_000.0, cfg=cfg)

    assert out["n_snps"] == 5
    assert "intercept" in out
    for key in ("base1", "base2", "ell_node", "ell_edge"):
        assert key in out
        assert set(out[key]) == {"tau", "se", "z", "p_twosided", "p_onesided"}
        assert 0.0 <= out[key]["p_onesided"] <= 1.0
        assert 0.0 <= out[key]["p_twosided"] <= 1.0


def test_run_per_pair_ldsc_skips_zero_and_applies_bonferroni():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=3)

    pair_ld = {
        "pair_active": np.array([0.2, 0.1, 0.4, 0.2, 0.3]),
        "pair_zero": np.zeros(5),
    }
    out = run_per_pair_ldsc(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld_node=annot_ld[["SNP", "ell_node"]],
        pair_ld_scores=pair_ld,
        snp_names=["rs1", "rs2", "rs3", "rs4", "rs5"],
        w_ld=w_ld,
        M_total=1_000_000.0,
        cfg=cfg,
    )

    assert list(out["pair"]) == ["pair_active"]
    assert "p_bonferroni" in out.columns
    # Bonferroni multiplier uses total tested dict length (including zero pair)
    assert np.isclose(out.iloc[0]["p_bonferroni"], min(out.iloc[0]["p_onesided"] * 2.0, 1.0))


def test_load_baseline_casts_float16_and_accumulates_M(monkeypatch):
    monkeypatch.setattr("edgemap.regression.resolve_resource_dir", lambda _: Path("/fake"))

    def fake_read_feather(_):
        return pd.DataFrame(
            {
                "SNP": ["rs1", "rs2"],
                "base_f16": np.array([0.5, 1.5], dtype=np.float16),
                "base_f64": np.array([1.0, 2.0], dtype=np.float64),
            }
        )

    monkeypatch.setattr("edgemap.regression.pd.read_feather", fake_read_feather)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO("1 2 3"))

    baseline, m_total = load_baseline("/fake")

    assert len(baseline) == 44  # 22 chrom * 2 rows each
    assert baseline["base_f16"].dtype == np.float64
    assert m_total == 22 * (1 + 2 + 3)


def test_load_regression_weights_concatenates_all_chromosomes(monkeypatch):
    monkeypatch.setattr("edgemap.regression.resolve_resource_dir", lambda _: Path("/fake"))

    def fake_read_csv(path, **kwargs):
        chrom = int(str(path).split("weights.")[1].split(".")[0])
        return pd.DataFrame({"SNP": [f"rs{chrom}"], "L2": [float(chrom)]})

    monkeypatch.setattr("edgemap.regression.pd.read_csv", fake_read_csv)
    weights = load_regression_weights("/fake")

    assert len(weights) == 22
    assert set(weights.columns) == {"SNP", "L2"}
    assert weights["SNP"].iloc[0] == "rs1"
    assert weights["SNP"].iloc[-1] == "rs22"


def test_run_per_pair_ldsc_aligns_to_snp_index_and_sorts_by_z(monkeypatch):
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=3)

    # Use extra SNP that is absent in merged base data; should be ignored via snp_idx mapping.
    snp_names = ["rsX", "rs1", "rs2", "rs3", "rs4", "rs5"]
    pair_ld = {
        "pair_low": np.array([9.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        "pair_high": np.array([9.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
    }

    def fake_block_jackknife(Xw, yw, n_blocks):
        # Pair coefficient is the penultimate column in run_per_pair_ldsc.
        tau = float(Xw[:, -2].mean())
        beta = np.zeros(Xw.shape[1], dtype=float)
        se = np.ones(Xw.shape[1], dtype=float)
        beta[-2] = tau
        se[-2] = 1.0
        return beta, se

    monkeypatch.setattr("edgemap.regression._block_jackknife", fake_block_jackknife)

    out = run_per_pair_ldsc(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld_node=annot_ld[["SNP", "ell_node"]],
        pair_ld_scores=pair_ld,
        snp_names=snp_names,
        w_ld=w_ld,
        M_total=1_000_000.0,
        cfg=cfg,
    )

    assert list(out["pair"]) == ["pair_high", "pair_low"]
    assert out.iloc[0]["z"] > out.iloc[1]["z"]


def test_run_sldsc_handles_zero_se_without_inf(monkeypatch):
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=3)

    def fake_block_jackknife(Xw, yw, n_blocks):
        p = Xw.shape[1]
        beta = np.zeros(p, dtype=float)
        se = np.zeros(p, dtype=float)  # triggers se<=1e-15 protection branch
        beta[-1] = 1.0  # intercept
        return beta, se

    monkeypatch.setattr("edgemap.regression._block_jackknife", fake_block_jackknife)

    out = run_sldsc(sumstats, baseline, annot_ld, w_ld, M_total=1_000_000.0, cfg=cfg)
    for key in ("base1", "base2", "ell_node", "ell_edge"):
        assert out[key]["z"] == 0.0
        assert out[key]["p_onesided"] == 0.5
        assert out[key]["p_twosided"] == 1.0
