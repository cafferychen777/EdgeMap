"""
Simulation framework for validating EdgeMap type-I error and power.

Generates synthetic GWAS chi-squared statistics under known genetic
architectures (null, node-only, edge-only, mixed) using real annotation
LD scores from the pipeline. This isolates the statistical test from
biological signal, enabling calibration and power assessment.

Design:
    E[χ²_s] = 1 + N * Σ_q τ_q * ℓ^(q)_s + N * τ_node * ℓ_node + N * τ_edge * ℓ_edge
    χ²_s ~ (1 + λ_s) * χ²_1, where λ_s = E[χ²_s] - 1

We sample from this distribution and feed the synthetic chi-squared
values into the same S-LDSC regression, checking whether the method
recovers the true architecture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm

from .regression import _block_jackknife, _sldsc_weights


@dataclass
class SimConfig:
    """Configuration for a single simulation scenario."""
    tau_node: float = 0.0     # true node coefficient (per-SNP, per-unit-annotation)
    tau_edge: float = 0.0     # true edge coefficient
    n_reps: int = 500         # number of replicates
    seed: int = 42


def simulate_chisq(
    baseline_ld: np.ndarray,
    ell_node: np.ndarray,
    ell_edge: np.ndarray,
    N_bar: float,
    M_total: float,
    tau_node: float,
    tau_edge: float,
    rng: np.random.Generator,
    baseline_tau: np.ndarray | None = None,
) -> np.ndarray:
    """Generate synthetic chi-squared statistics under a specified architecture.

    The noncentrality parameter at each SNP is:
        λ_s = N * (Σ_q τ_q * ℓ^(q)_s + τ_node * ℓ_node_s + τ_edge * ℓ_edge_s)

    Chi-squared values are drawn from a scaled chi-squared(1) distribution:
        χ²_s ~ (1 + λ_s) * χ²_1

    Args:
        baseline_ld: (n_snps, n_baseline) baseline annotation LD scores
        ell_node: (n_snps,) node annotation LD scores
        ell_edge: (n_snps,) edge annotation LD scores
        N_bar: mean GWAS sample size
        M_total: total number of common SNPs (for h² scaling)
        tau_node: true per-SNP node effect
        tau_edge: true per-SNP edge effect
        rng: numpy random generator
        baseline_tau: (n_baseline,) baseline coefficients; if None, estimated
                      from a reasonable default (uniform h² across baseline)

    Returns:
        chisq: (n_snps,) synthetic chi-squared values
    """
    n_snps, n_bl = baseline_ld.shape

    # Default baseline: uniform heritability spread across baseline annotations
    if baseline_tau is None:
        # Target h² = 0.3, split uniformly across baseline annotations
        h2_target = 0.3
        total_ell = baseline_ld.sum(axis=0).mean()
        baseline_tau = np.full(n_bl, h2_target / (n_bl * total_ell * N_bar / M_total))

    # Noncentrality: λ_s = N * (Σ τ_q ℓ_q + τ_node ℓ_node + τ_edge ℓ_edge)
    lam = N_bar * (baseline_ld @ baseline_tau + tau_node * ell_node + tau_edge * ell_edge)
    lam = np.maximum(lam, 0.0)

    # χ² ~ (1 + λ) * χ²(1)
    z = rng.standard_normal(n_snps)
    chisq = (1.0 + lam) * z**2

    return chisq


def run_simulation(
    baseline: pd.DataFrame,
    annot_ld: pd.DataFrame,
    w_ld: pd.DataFrame,
    M_total: float,
    N_bar: float,
    cfg: SimConfig,
    n_blocks: int = 200,
) -> pd.DataFrame:
    """Run a full simulation scenario with multiple replicates.

    Uses the same S-LDSC regression machinery as the real pipeline,
    but with synthetic chi-squared values.

    Returns:
        DataFrame with columns [rep, tau_node, se_node, z_node, p_node,
                                tau_edge, se_edge, z_edge, p_edge]
    """
    rng = np.random.default_rng(cfg.seed)

    # Prepare regression inputs (same merge logic as run_sldsc)
    baseline_cols = [c for c in baseline.columns if c != "SNP"]
    df = (
        baseline
        .merge(annot_ld, on="SNP", how="inner")
        .merge(w_ld[["SNP", "L2"]], on="SNP", how="inner")
        .rename(columns={"L2": "w_ld"})
    )

    baseline_ld = df[baseline_cols].values.astype(np.float64)
    ell_node = df["ell_node"].values.astype(np.float64)
    ell_edge = df["ell_edge"].values.astype(np.float64)
    w_ld_vals = np.maximum(df["w_ld"].values, 1.0)
    n_snps = len(df)

    # Design matrix columns (same as run_sldsc)
    annot_names = baseline_cols + ["ell_node", "ell_edge"]
    node_idx = len(baseline_cols)      # index of ell_node in beta
    edge_idx = len(baseline_cols) + 1  # index of ell_edge in beta

    results = []
    for rep in range(cfg.n_reps):
        # Generate synthetic chi-squared
        chisq = simulate_chisq(
            baseline_ld, ell_node, ell_edge,
            N_bar, M_total,
            cfg.tau_node, cfg.tau_edge,
            rng,
        )

        # Construct design matrix: [N * ℓ_1, ..., N * ℓ_k, 1]
        ell_arrays = [baseline_ld[:, i] for i in range(baseline_ld.shape[1])]
        ell_arrays += [ell_node, ell_edge]
        X = np.column_stack([N_bar * arr for arr in ell_arrays] + [np.ones(n_snps)])

        # Weights (same formula as run_sldsc)
        w = _sldsc_weights(chisq, baseline_ld, w_ld_vals, N_bar, M_total)

        sqrtw = np.sqrt(w)
        Xw = X * sqrtw[:, None]
        yw = chisq * sqrtw

        beta_hat, jk_se = _block_jackknife(Xw, yw, n_blocks)

        # Extract node and edge results
        for label, idx in [("node", node_idx), ("edge", edge_idx)]:
            tau = float(beta_hat[idx])
            se = float(jk_se[idx])
            z = tau / se if se > 1e-15 else 0.0
            results.append({
                "rep": rep,
                "annotation": label,
                "tau": tau, "se": se, "z": z,
                "p_onesided": float(norm.sf(z)),
                "p_twosided": float(2 * norm.sf(abs(z))),
            })

    return pd.DataFrame(results)


# ── Predefined scenarios ──────────────────────────────────────────────

# Scale factor: typical tau ~ 5e-9 from SBP results
_TAU_SCALE = 5e-9

SCENARIOS = {
    "null": SimConfig(tau_node=0.0, tau_edge=0.0),
    "node_only": SimConfig(tau_node=_TAU_SCALE, tau_edge=0.0),
    "edge_only": SimConfig(tau_node=0.0, tau_edge=_TAU_SCALE),
    "mixed": SimConfig(tau_node=_TAU_SCALE, tau_edge=_TAU_SCALE),
}
