import io
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from edgemap.config import RegressionConfig
from edgemap.regression import (
    _baseline_cache,
    _regression_weight_cache,
    _block_jackknife,
    _sldsc_weights,
    load_baseline,
    load_regression_weights,
    load_sumstats,
    run_per_pair_ldsc,
    run_per_pair_ldsc_custom,
    run_sldsc,
    run_sldsc_custom,
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


def test_load_sumstats_filters_invalid_sample_size(tmp_path):
    path = tmp_path / "bad_n.tsv"
    pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3"],
            "Z": [2.0, 1.5, 1.2],
            "N": [1000, np.inf, -5],
        }
    ).to_csv(path, sep="\t", index=False)

    out = load_sumstats(str(path), RegressionConfig())
    assert list(out["SNP"]) == ["rs1"]


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


def test_block_jackknife_rejects_nonpositive_blocks():
    Xw = np.array([[1.0, 2.0], [3.0, 4.0]])
    yw = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="n_blocks must be > 0"):
        _block_jackknife(Xw, yw, n_blocks=0)


def test_regression_config_rejects_nonpositive_blocks():
    with pytest.raises(ValueError, match="n_blocks must be > 0"):
        RegressionConfig(n_blocks=0)


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
    assert set(out.columns) == {"pair", "tau", "se", "z"}
    assert np.isfinite(out.iloc[0]["z"])


def test_run_sldsc_custom_accepts_extra_annotations():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    annot_extra = annot_ld.copy()
    annot_extra["ell_fibro"] = [0.1, 0.0, 0.2, 0.1, 0.0]

    out = run_sldsc_custom(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld=annot_extra,
        w_ld=w_ld,
        M_total=1_000_000.0,
        cfg=RegressionConfig(n_blocks=3),
        annot_cols=["ell_node", "ell_edge", "ell_fibro"],
    )

    assert "ell_fibro" in out
    assert set(out["ell_fibro"]) == {"tau", "se", "z", "p_twosided", "p_onesided"}



def test_run_per_pair_ldsc_custom_accepts_multiple_controls():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    annot_controls = annot_ld[["SNP", "ell_node"]].copy()
    annot_controls["ell_fibro"] = [0.1, 0.0, 0.2, 0.1, 0.0]
    pair_ld = {"pair_active": np.array([0.2, 0.1, 0.4, 0.2, 0.3])}

    out = run_per_pair_ldsc_custom(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld_controls=annot_controls,
        pair_ld_scores=pair_ld,
        snp_names=["rs1", "rs2", "rs3", "rs4", "rs5"],
        w_ld=w_ld,
        M_total=1_000_000.0,
        cfg=RegressionConfig(n_blocks=3),
        control_cols=["ell_node", "ell_fibro"],
    )

    assert list(out["pair"]) == ["pair_active"]
    assert np.isfinite(out.iloc[0]["tau"])


def test_load_baseline_casts_float16_and_accumulates_M(monkeypatch):
    _baseline_cache.clear()
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
    _regression_weight_cache.clear()
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


def test_load_baseline_uses_cache(monkeypatch):
    _baseline_cache.clear()
    monkeypatch.setattr("edgemap.regression.resolve_resource_dir", lambda _: Path("/fake"))

    calls = {"feather": 0, "open": 0}

    def fake_read_feather(_):
        calls["feather"] += 1
        return pd.DataFrame({"SNP": ["rs1"], "base1": [1.0]})

    def fake_open(*args, **kwargs):
        calls["open"] += 1
        return io.StringIO("1")

    monkeypatch.setattr("edgemap.regression.pd.read_feather", fake_read_feather)
    monkeypatch.setattr("builtins.open", fake_open)

    first = load_baseline("/fake", copy=False)
    second = load_baseline("/fake", copy=False)

    assert calls == {"feather": 22, "open": 22}
    assert first[0] is second[0]
    assert first[1] == second[1]


def test_load_regression_weights_uses_cache(monkeypatch):
    _regression_weight_cache.clear()
    monkeypatch.setattr("edgemap.regression.resolve_resource_dir", lambda _: Path("/fake"))

    calls = {"csv": 0}

    def fake_read_csv(path, **kwargs):
        calls["csv"] += 1
        chrom = int(str(path).split("weights.")[1].split(".")[0])
        return pd.DataFrame({"SNP": [f"rs{chrom}"], "L2": [float(chrom)]})

    monkeypatch.setattr("edgemap.regression.pd.read_csv", fake_read_csv)

    first = load_regression_weights("/fake", copy=False)
    second = load_regression_weights("/fake", copy=False)

    assert calls == {"csv": 22}
    assert first is second


def test_load_baseline_returns_defensive_copy_by_default(monkeypatch):
    _baseline_cache.clear()
    monkeypatch.setattr("edgemap.regression.resolve_resource_dir", lambda _: Path("/fake"))
    monkeypatch.setattr(
        "edgemap.regression.pd.read_feather",
        lambda _: pd.DataFrame({"SNP": ["rs1"], "base1": [1.0]}),
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO("1"))

    first, _ = load_baseline("/fake")
    second, _ = load_baseline("/fake")
    first.loc[0, "base1"] = 99.0

    assert first is not second
    assert second.loc[0, "base1"] == 1.0


def test_load_regression_weights_returns_defensive_copy_by_default(monkeypatch):
    _regression_weight_cache.clear()
    monkeypatch.setattr("edgemap.regression.resolve_resource_dir", lambda _: Path("/fake"))
    monkeypatch.setattr(
        "edgemap.regression.pd.read_csv",
        lambda path, **kwargs: pd.DataFrame({"SNP": [str(path)], "L2": [1.0]}),
    )

    first = load_regression_weights("/fake")
    second = load_regression_weights("/fake")
    first.loc[0, "L2"] = 99.0

    assert first is not second
    assert second.loc[0, "L2"] == 1.0


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


def test_run_sldsc_raises_on_empty_merge():
    """No overlapping SNPs → ValueError."""
    sumstats = pd.DataFrame({"SNP": ["rs1"], "Z": [2.0], "N": [1000]})
    baseline = pd.DataFrame({"SNP": ["rsX"], "base1": [0.5]})
    annot_ld = pd.DataFrame({"SNP": ["rsX"], "ell_node": [0.1], "ell_edge": [0.2]})
    w_ld = pd.DataFrame({"SNP": ["rsX"], "L2": [1.0]})

    with pytest.raises(ValueError, match="No SNPs remain"):
        run_sldsc(sumstats, baseline, annot_ld, w_ld, 1e6, RegressionConfig())


def test_block_jackknife_clamps_blocks_to_n():
    """n_blocks > n should not crash (clamped internally)."""
    Xw = np.array([[1.0, 2.0], [3.0, 4.0]])
    yw = np.array([1.0, 2.0])
    beta, se = _block_jackknife(Xw, yw, n_blocks=100)
    assert beta.shape == (2,)
    assert np.all(np.isfinite(beta))


def test_sldsc_weights_matches_inline():
    """Shared _sldsc_weights matches the original inline computation."""
    rng = np.random.RandomState(42)
    y = rng.rand(50) * 5 + 1
    baseline_ld = rng.rand(50, 3)
    w_ld_vals = np.maximum(rng.rand(50), 0.1)
    N_bar, M_total = 50000.0, 1e6

    w = _sldsc_weights(y, baseline_ld, w_ld_vals, N_bar, M_total)

    # Inline reference
    x_tot = baseline_ld.sum(axis=1)
    h2_init = np.clip((y.mean() - 1) * M_total / (N_bar * x_tot.mean()), 0.01, 1.0)
    Ey = 1.0 + np.clip(h2_init * N_bar / M_total * x_tot, 0, 1e4)
    expected = 1.0 / (2.0 * Ey**2 * w_ld_vals)

    np.testing.assert_allclose(w, expected, rtol=1e-15)
