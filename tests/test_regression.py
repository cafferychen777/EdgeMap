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
    _order_genomically,
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
    rng = np.random.default_rng(2026)
    n_snps = 40
    snps = [f"rs{i}" for i in range(n_snps)]
    base1 = rng.uniform(0.1, 1.0, n_snps)
    base2 = rng.uniform(0.1, 1.0, n_snps)
    node = rng.uniform(0.0, 0.8, n_snps)
    edge = rng.uniform(0.0, 0.8, n_snps)
    sample_size = rng.integers(900, 1100, n_snps).astype(float)
    chisq = (
        1.0
        + sample_size * (2e-4 * base1 + 1e-4 * base2 + 3e-4 * node + 4e-4 * edge)
        + rng.normal(0.0, 0.03, n_snps)
    )
    sumstats = pd.DataFrame(
        {
            "SNP": snps,
            "Z": np.sqrt(chisq),
            "N": sample_size,
        }
    )
    baseline = pd.DataFrame(
        {
            "SNP": snps,
            "base1": base1,
            "base2": base2,
        }
    )
    annot_ld = pd.DataFrame(
        {
            "SNP": snps,
            "ell_node": node,
            "ell_edge": edge,
        }
    )
    w_ld = pd.DataFrame({"SNP": snps, "L2": rng.uniform(1.0, 2.0, n_snps)})
    return sumstats, baseline, annot_ld, w_ld


def test_load_sumstats_validates_and_filters(tmp_path):
    path = tmp_path / "sumstats.tsv"
    df = pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4"],
            "Z": [2.0, np.nan, np.inf, 100.0],
            "N": [1000, 1000, 1000, 1000],
        }
    )
    df.to_csv(path, sep="\t", index=False)

    cfg = RegressionConfig(chisq_max_factor=0.001, chisq_max_floor=5.0)
    out = load_sumstats(str(path), cfg)

    # rs4 is removed by chisq threshold (100^2 > 5); rs2/rs3 are invalid.
    assert list(out["SNP"]) == ["rs1"]
    assert list(out.columns) == ["SNP", "Z", "N"]


def test_load_sumstats_rejects_duplicate_snps(tmp_path):
    path = tmp_path / "duplicate.tsv"
    pd.DataFrame(
        {"SNP": ["rs1", "rs1"], "Z": [1.0, 1.1], "N": [1000, 1000]}
    ).to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="duplicate SNP"):
        load_sumstats(str(path), RegressionConfig())


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


def test_block_jackknife_rejects_singular_matrix():
    # Collinear annotations do not define separate conditional coefficients.
    Xw = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )
    yw = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="ill-conditioned|rank-deficient"):
        _block_jackknife(Xw, yw, n_blocks=3)


def test_block_jackknife_rejects_nonpositive_blocks():
    Xw = np.array([[1.0, 2.0], [3.0, 4.0]])
    yw = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="n_blocks must be >= 2"):
        _block_jackknife(Xw, yw, n_blocks=0)


def test_regression_config_rejects_fewer_than_two_blocks():
    for n_blocks in (0, 1):
        with pytest.raises(ValueError, match="n_blocks must be >= 2"):
            RegressionConfig(n_blocks=n_blocks)


def test_run_sldsc_returns_expected_structure():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=3)
    out = run_sldsc(sumstats, baseline, annot_ld, w_ld, M_total=1_000_000.0, cfg=cfg)

    assert out["n_snps"] == 40
    assert "intercept" in out
    for key in ("base1", "base2", "ell_node", "ell_edge"):
        assert key in out
        assert set(out[key]) == {"tau", "se", "z", "p_twosided", "p_onesided"}
        assert 0.0 <= out[key]["p_onesided"] <= 1.0
        assert 0.0 <= out[key]["p_twosided"] <= 1.0


def test_run_per_pair_ldsc_skips_zero_and_returns_ranking_columns():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=3)

    pair_ld = {
        "pair_active": np.linspace(0.1, 0.9, len(sumstats)),
        "pair_zero": np.zeros(len(sumstats)),
    }
    out = run_per_pair_ldsc(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld_node=annot_ld[["SNP", "ell_node"]],
        pair_ld_scores=pair_ld,
        snp_names=baseline["SNP"].tolist(),
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
    annot_extra["ell_fibro"] = np.linspace(0.0, 0.7, len(annot_extra)) ** 2

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
    annot_controls["ell_fibro"] = np.linspace(0.0, 0.7, len(annot_controls)) ** 2
    pair_ld = {"pair_active": np.linspace(0.1, 0.9, len(sumstats))}

    out = run_per_pair_ldsc_custom(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld_controls=annot_controls,
        pair_ld_scores=pair_ld,
        snp_names=baseline["SNP"].tolist(),
        w_ld=w_ld,
        M_total=1_000_000.0,
        cfg=RegressionConfig(n_blocks=3),
        control_cols=["ell_node", "ell_fibro"],
    )

    assert list(out["pair"]) == ["pair_active"]
    assert np.isfinite(out.iloc[0]["tau"])


def test_per_pair_z_is_invariant_to_positive_scalar_rescaling():
    rng = np.random.default_rng(123)
    n_snps = 80
    snp_names = [f"rs{i}" for i in range(n_snps)]
    baseline_values = rng.uniform(0.1, 1.0, n_snps)
    node_values = rng.uniform(0.1, 1.0, n_snps)
    membership_values = rng.uniform(0.0, 1.0, n_snps)
    sample_size = np.full(n_snps, 1000.0)

    expected_chisq = (
        1.0
        + sample_size
        * (
            0.0002 * baseline_values
            + 0.0003 * node_values
            + 0.0005 * membership_values
        )
        + rng.normal(0.0, 0.02, n_snps)
    )
    sumstats = pd.DataFrame(
        {
            "SNP": snp_names,
            "Z": np.sqrt(np.maximum(expected_chisq, 0.01)),
            "N": sample_size,
        }
    )
    baseline = pd.DataFrame({"SNP": snp_names, "base": baseline_values})
    controls = pd.DataFrame({"SNP": snp_names, "ell_node": node_values})
    weights = pd.DataFrame({"SNP": snp_names, "L2": np.ones(n_snps)})
    scale = 7.3

    out = run_per_pair_ldsc_custom(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld_controls=controls,
        pair_ld_scores={
            "membership-scale": membership_values,
            "rescaled": scale * membership_values,
        },
        snp_names=snp_names,
        w_ld=weights,
        M_total=1_000_000.0,
        cfg=RegressionConfig(n_blocks=10),
        control_cols=["ell_node"],
    ).set_index("pair")

    assert out.loc["rescaled", "tau"] == pytest.approx(
        out.loc["membership-scale", "tau"] / scale,
        rel=1e-10,
    )
    assert out.loc["rescaled", "se"] == pytest.approx(
        out.loc["membership-scale", "se"] / scale,
        rel=1e-9,
    )
    assert out.loc["rescaled", "z"] == pytest.approx(
        out.loc["membership-scale", "z"],
        rel=1e-9,
    )


def test_per_pair_rejects_exact_own_membership_collinearity():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    membership = np.linspace(0.1, 0.9, len(sumstats))

    with pytest.raises(ValueError, match="positive scalar multiples"):
        run_per_pair_ldsc_custom(
            sumstats=sumstats,
            baseline=baseline,
            annot_ld_controls=annot_ld[["SNP", "ell_node"]],
            pair_ld_scores={"L-R": 2.5 * membership},
            pair_membership_ld_scores={"L-R": membership},
            snp_names=baseline["SNP"].tolist(),
            w_ld=w_ld,
            M_total=1_000_000.0,
            cfg=RegressionConfig(n_blocks=3),
            control_cols=["ell_node"],
        )


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
    snp_names = ["rsX"] + baseline["SNP"].tolist()
    n_aligned = len(snp_names)
    pair_ld = {
        "pair_low": np.array([9.0] + [0.1] * (n_aligned - 1)),
        "pair_high": np.array([9.0] + [5.0] * (n_aligned - 1)),
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


def test_run_sldsc_rejects_zero_standard_error(monkeypatch):
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=3)

    def fake_block_jackknife(Xw, yw, n_blocks):
        p = Xw.shape[1]
        beta = np.zeros(p, dtype=float)
        se = np.zeros(p, dtype=float)  # triggers se<=1e-15 protection branch
        beta[-1] = 1.0  # intercept
        return beta, se

    monkeypatch.setattr("edgemap.regression._block_jackknife", fake_block_jackknife)

    with pytest.raises(ValueError, match="Non-positive jackknife standard error"):
        run_sldsc(sumstats, baseline, annot_ld, w_ld, M_total=1_000_000.0, cfg=cfg)


def test_run_sldsc_raises_on_empty_merge():
    """No overlapping SNPs → ValueError."""
    sumstats = pd.DataFrame({"SNP": ["rs1"], "Z": [2.0], "N": [1000]})
    baseline = pd.DataFrame({"SNP": ["rsX"], "base1": [0.5]})
    annot_ld = pd.DataFrame({"SNP": ["rsX"], "ell_node": [0.1], "ell_edge": [0.2]})
    w_ld = pd.DataFrame({"SNP": ["rsX"], "L2": [1.0]})

    with pytest.raises(ValueError, match="No SNPs remain"):
        run_sldsc(sumstats, baseline, annot_ld, w_ld, 1e6, RegressionConfig())


def test_block_jackknife_rejects_more_blocks_than_snps():
    """n_blocks cannot exceed the available SNP count."""
    x = np.arange(8, dtype=float)
    Xw = np.column_stack([np.ones(8), x])
    yw = 1.0 + 2.0 * x + np.array([0.1, -0.1] * 4)
    with pytest.raises(ValueError, match="cannot exceed"):
        _block_jackknife(Xw, yw, n_blocks=100)


def test_sldsc_weights_use_per_snp_sample_size():
    """Expected chi-square and weights use each SNP's own sample size."""
    rng = np.random.RandomState(42)
    y = rng.rand(50) * 5 + 1
    baseline_ld = rng.rand(50, 3)
    w_ld_vals = np.maximum(rng.rand(50), 0.1)
    N = np.linspace(25_000.0, 75_000.0, len(y))
    M_total = 1e6

    w = _sldsc_weights(y, baseline_ld, w_ld_vals, N, M_total)

    # Inline reference
    x_tot = baseline_ld.sum(axis=1)
    h2_init = np.clip((y.mean() - 1) * M_total / np.mean(N * x_tot), 0.0, 1.0)
    Ey = 1.0 + h2_init * N / M_total * np.maximum(x_tot, 1.0)
    expected = 1.0 / (2.0 * Ey**2 * w_ld_vals)

    np.testing.assert_allclose(w, expected, rtol=1e-15)


def test_order_genomically_restores_baseline_order():
    """The frame is sorted into the baseline's row order, whatever it arrives in."""
    _, baseline, _, _ = _toy_regression_inputs()
    permutation = np.random.default_rng(7).permutation(len(baseline))
    scrambled = pd.DataFrame(
        {
            "SNP": baseline["SNP"].iloc[permutation].values,
            "value": permutation,
        }
    )
    ordered = _order_genomically(scrambled, baseline)
    assert list(ordered["SNP"]) == list(baseline["SNP"])
    assert list(ordered["value"]) == list(range(len(baseline)))
    assert list(ordered.index) == list(range(len(baseline)))


def test_sldsc_is_invariant_to_the_order_the_frame_arrives_in():
    """Row order must not reach the jackknife.

    _block_jackknife blocks on runs of consecutive rows, so before the ordering
    fix a frame built from scrambled summary statistics produced a different
    standard error from the same data in genomic order. Both tau and se must
    now agree.
    """
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=5)
    ordered = run_sldsc_custom(sumstats, baseline, annot_ld, w_ld, 1000.0, cfg)

    perm = np.random.default_rng(8).permutation(len(sumstats))
    shuffled = run_sldsc_custom(
        sumstats.iloc[perm].reset_index(drop=True),
        baseline, annot_ld, w_ld, 1000.0, cfg,
    )

    for key in ("ell_node", "ell_edge"):
        assert ordered[key]["tau"] == pytest.approx(shuffled[key]["tau"], rel=1e-12)
        assert ordered[key]["se"] == pytest.approx(shuffled[key]["se"], rel=1e-12)


def test_per_pair_is_invariant_to_the_order_the_frame_arrives_in():
    sumstats, baseline, annot_ld, w_ld = _toy_regression_inputs()
    cfg = RegressionConfig(n_blocks=5)
    snp_names = list(baseline["SNP"])
    pair_ld = {"A-B": np.linspace(0.1, 0.9, len(sumstats))}

    def run(ss):
        return run_per_pair_ldsc_custom(
            sumstats=ss, baseline=baseline,
            annot_ld_controls=annot_ld, pair_ld_scores=pair_ld,
            snp_names=snp_names, w_ld=w_ld, M_total=1000.0, cfg=cfg,
            control_cols=["ell_node"],
        )

    a = run(sumstats)
    perm = np.random.default_rng(9).permutation(len(sumstats))
    b = run(sumstats.iloc[perm].reset_index(drop=True))
    assert a.loc[0, "tau"] == pytest.approx(b.loc[0, "tau"], rel=1e-12)
    assert a.loc[0, "se"] == pytest.approx(b.loc[0, "se"], rel=1e-12)
