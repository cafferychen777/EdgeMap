"""
Stratified LD Score Regression with block jackknife.

Step 7 of the pipeline.

Model:
    E[χ²_s] = 1 + N · Σ_q τ_q · ℓ^(q)_s        (baseline annotations)
                + N · τ_node · ℓ_node(s)           (expression specificity)
                + N · τ_edge · ℓ_edge(s)           (communication specificity)

Weights: w_s = 1 / (2 · E[χ²_s]² · w_ld_s)
  - First factor: heteroscedasticity correction (χ² variance ∝ E[χ²]²)
  - Second factor: LD-induced correlation between nearby SNPs

Standard errors: delete-one-block jackknife over ~200 LD blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

from .config import RegressionConfig, resolve_resource_dir


def load_baseline(
    resource_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, float]:
    """Load pre-computed baseline LD scores and total M_5_50."""
    rdir = resolve_resource_dir(resource_dir)
    baseline_dir = rdir / "quick_mode" / "baseline"

    frames = []
    M_total = 0.0
    for chrom in range(1, 23):
        bl = pd.read_feather(baseline_dir / f"baseline.{chrom}.l2.ldscore.feather")
        for col in bl.select_dtypes(include=["float16"]).columns:
            bl[col] = bl[col].astype(np.float64)
        frames.append(bl)
        with open(baseline_dir / f"baseline.{chrom}.l2.M_5_50") as f:
            M_total += sum(float(x) for x in f.read().strip().split())
    return pd.concat(frames, ignore_index=True), M_total


def load_regression_weights(
    resource_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load LD-based regression weights (w_ld)."""
    rdir = resolve_resource_dir(resource_dir)
    weight_dir = rdir / "LDSC_resource" / "weights_hm3_no_hla"

    frames = []
    for chrom in range(1, 23):
        w = pd.read_csv(
            weight_dir / f"weights.{chrom}.l2.ldscore.gz",
            sep="\t", compression="gzip",
        )
        frames.append(w)
    return pd.concat(frames, ignore_index=True)


def load_sumstats(path: str, cfg: RegressionConfig) -> pd.DataFrame:
    """Load munged GWAS summary statistics (SNP/Z/N format).

    Filters extreme chi-squared values to avoid outlier-driven estimates.
    """
    ss = pd.read_csv(path, sep="\t")
    required = {"SNP", "Z", "N"}
    missing = required - set(ss.columns)
    if missing:
        raise ValueError(f"Sumstats missing columns: {missing}. Found: {list(ss.columns)}")

    ss = ss.dropna(subset=["Z", "N"])
    ss = ss[np.isfinite(ss["Z"])]
    chisq_max = max(cfg.chisq_max_factor * ss["N"].max(), cfg.chisq_max_floor)
    ss = ss[ss["Z"] ** 2 < chisq_max]
    return ss[["SNP", "Z", "N"]].drop_duplicates("SNP")


def _sldsc_weights(
    y: np.ndarray,
    baseline_ld: np.ndarray,
    w_ld_vals: np.ndarray,
    N_bar: float,
    M_total: float,
) -> np.ndarray:
    """S-LDSC regression weights: heteroscedasticity × LD correction.

    w_s = 1 / (2 · E[χ²_s]² · w_ld_s)
    """
    x_tot = baseline_ld.sum(axis=1)
    h2_init = np.clip((y.mean() - 1) * M_total / (N_bar * x_tot.mean()), 0.01, 1.0)
    Ey = 1.0 + np.clip(h2_init * N_bar / M_total * x_tot, 0, 1e4)
    return 1.0 / (2.0 * Ey**2 * w_ld_vals)


def _block_jackknife(
    Xw: np.ndarray, yw: np.ndarray, n_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast block jackknife for weighted least squares.

    Precomputes X'X and X'y, then obtains each delete-one-block estimate
    by subtracting the block's contribution — O(p³) per block instead of
    O(n·p²) for a full re-regression.

    Uses np.array_split for even block distribution when n is not
    divisible by n_blocks.
    """
    n, p = Xw.shape
    n_blocks = min(n_blocks, n)

    # Full regression via normal equations
    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw
    try:
        beta_hat = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta_hat = np.linalg.lstsq(Xw, yw, rcond=None)[0]

    # Block-level contributions
    blocks = np.array_split(np.arange(n), n_blocks)
    beta_delete = np.zeros((n_blocks, p))
    for b, idx in enumerate(blocks):
        Xb = Xw[idx]
        yb = yw[idx]
        XtX_del = XtX - Xb.T @ Xb
        Xty_del = Xty - Xb.T @ yb
        try:
            beta_delete[b] = np.linalg.solve(XtX_del, Xty_del)
        except np.linalg.LinAlgError:
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            beta_delete[b] = np.linalg.lstsq(Xw[mask], yw[mask], rcond=None)[0]

    # Pseudovalues → estimate and SE
    pseudo = n_blocks * beta_hat[None, :] - (n_blocks - 1) * beta_delete
    se = np.sqrt(np.var(pseudo, axis=0) / n_blocks)

    return beta_hat, se


def run_sldsc(
    sumstats: pd.DataFrame,
    baseline: pd.DataFrame,
    annot_ld: pd.DataFrame,
    w_ld: pd.DataFrame,
    M_total: float,
    cfg: RegressionConfig,
) -> dict:
    """Run joint S-LDSC regression with node and edge annotations.

    Merges all data on common SNPs, constructs design matrix,
    computes heteroscedasticity + LD correction weights, runs WLS
    with block jackknife standard errors.
    """
    df = (
        sumstats
        .merge(baseline, on="SNP", how="inner")
        .merge(annot_ld, on="SNP", how="inner")
        .merge(w_ld[["SNP", "L2"]], on="SNP", how="inner")
        .rename(columns={"L2": "w_ld"})
    )
    n_snps = len(df)
    if n_snps == 0:
        raise ValueError("No SNPs remain after merging sumstats, baseline, annotations, and weights.")

    y = (df["Z"] ** 2).values
    N = df["N"].values
    Nbar = N.mean()

    # Annotation columns: baseline + node + edge
    baseline_cols = [c for c in baseline.columns if c != "SNP"]
    annot_names = baseline_cols + ["ell_node", "ell_edge"]

    # Design matrix: [N·ℓ_1, ..., N·ℓ_k, 1]
    ell_arrays = [df[c].values.astype(np.float64) for c in annot_names]
    X = np.column_stack([N * ell for ell in ell_arrays] + [np.ones(n_snps)])

    # Regression weights: heteroscedasticity × LD overlap correction
    w_ld_vals = np.maximum(df["w_ld"].values, 1.0)
    w = _sldsc_weights(y, df[baseline_cols].values, w_ld_vals, Nbar, M_total)

    # Weighted LS with block jackknife
    sqrtw = np.sqrt(w)
    Xw = X * sqrtw[:, None]
    yw = y * sqrtw

    beta_hat, jk_se = _block_jackknife(Xw, yw, cfg.n_blocks)

    # Package results
    results = {"n_snps": n_snps, "N_bar": float(Nbar), "M_total": float(M_total)}
    for i, name in enumerate(annot_names):
        tau = float(beta_hat[i])
        se = float(jk_se[i])
        z = tau / se if se > 1e-15 else 0.0
        results[name] = {
            "tau": tau, "se": se, "z": z,
            "p_twosided": float(2 * norm.sf(abs(z))),
            "p_onesided": float(norm.sf(z)),
        }
    results["intercept"] = float(beta_hat[-1])

    return results


def run_per_pair_ldsc(
    sumstats: pd.DataFrame,
    baseline: pd.DataFrame,
    annot_ld_node: pd.DataFrame,
    pair_ld_scores: dict[str, np.ndarray],
    snp_names: list[str],
    w_ld: pd.DataFrame,
    M_total: float,
    cfg: RegressionConfig,
) -> pd.DataFrame:
    """Per-LR-pair conditional S-LDSC: test each pair individually.

    For each pair p, the model is:
        E[χ²] = 1 + N·Σ τ_q·ℓ^(q) + N·τ_node·ℓ_node + N·τ_p·ℓ_p

    Conditions on baseline + node annotation, testing one pair at a time.
    This identifies which specific LR pairs drive the aggregate edge signal.

    Args:
        sumstats: GWAS summary stats (SNP, Z, N)
        baseline: baseline LD score DataFrame
        annot_ld_node: DataFrame with columns [SNP, ell_node]
        pair_ld_scores: dict mapping pair_name -> (n_snps,) LD score array
        snp_names: SNP names aligned with pair_ld_scores arrays
        w_ld: regression weights DataFrame
        M_total: total M for h² scaling
        cfg: regression config

    Returns:
        DataFrame with columns [pair, tau, se, z, p_onesided, p_bonferroni]
        sorted by z descending.
    """
    # Merge base data once
    df_base = (
        sumstats
        .merge(baseline, on="SNP", how="inner")
        .merge(annot_ld_node[["SNP", "ell_node"]], on="SNP", how="inner")
        .merge(w_ld[["SNP", "L2"]], on="SNP", how="inner")
        .rename(columns={"L2": "w_ld"})
    )

    # Build SNP index for fast lookup
    snp_to_idx = {s: i for i, s in enumerate(snp_names)}
    df_snp_indices = df_base["SNP"].map(snp_to_idx)
    valid = df_snp_indices.notna()
    df_base = df_base[valid].copy()
    snp_idx = df_snp_indices[valid].astype(int).values

    n_snps = len(df_base)
    if n_snps == 0:
        raise ValueError("No SNPs remain after merging sumstats, baseline, node annotations, and weights.")

    y = (df_base["Z"] ** 2).values
    N = df_base["N"].values
    Nbar = N.mean()

    baseline_cols = [c for c in baseline.columns if c != "SNP"]
    ell_node = df_base["ell_node"].values.astype(np.float64)
    baseline_ld = df_base[baseline_cols].values.astype(np.float64)
    w_ld_vals = np.maximum(df_base["w_ld"].values, 1.0)

    # Precompute weights and base design matrix (shared across all pairs)
    w = _sldsc_weights(y, baseline_ld, w_ld_vals, Nbar, M_total)
    sqrtw = np.sqrt(w)

    # Base design: [N*baseline, N*node, 1]  -- pair column appended per iteration
    X_base = np.column_stack(
        [N * baseline_ld[:, i] for i in range(baseline_ld.shape[1])]
        + [N * ell_node, np.ones(n_snps)]
    )

    n_pairs = len(pair_ld_scores)
    results = []

    for pname, pair_ell_full in pair_ld_scores.items():
        pair_ell = pair_ell_full[snp_idx]

        # Skip pairs with no annotation signal
        if pair_ell.max() <= 0:
            continue

        # Insert pair column before intercept
        X = np.column_stack([
            X_base[:, :-1],     # baseline + node
            N * pair_ell,       # pair-specific edge
            X_base[:, -1:],     # intercept
        ])

        Xw = X * sqrtw[:, None]
        yw = y * sqrtw

        beta_hat, jk_se = _block_jackknife(Xw, yw, cfg.n_blocks)

        # Pair coefficient is at index -2 (before intercept)
        tau = float(beta_hat[-2])
        se = float(jk_se[-2])
        z = tau / se if se > 1e-15 else 0.0

        results.append({
            "pair": pname,
            "tau": tau, "se": se, "z": z,
            "p_onesided": float(norm.sf(z)),
        })

    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results["p_bonferroni"] = np.minimum(
            df_results["p_onesided"] * n_pairs, 1.0
        )
        df_results = df_results.sort_values("z", ascending=False).reset_index(drop=True)

    return df_results
