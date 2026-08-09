"""
Stratified LD Score Regression with block jackknife.

Step 7 of the pipeline.

Model:
    E[χ²_s] = 1 + N · Σ_q τ_q · ℓ^(q)_s        (baseline annotations)
                + N · τ_node · ℓ_node(s)           (expression specificity)
                + N · τ_edge · ℓ_edge(s)           (spatial LR-gene annotation)

Weights: w_s = 1 / (2 · E[χ²_s]² · w_ld_s)
  - First factor: heteroscedasticity correction (χ² variance ∝ E[χ²]²)
  - Second factor: LD-induced correlation between nearby SNPs

Standard errors: delete-one-block jackknife over ~200 LD blocks.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import RegressionConfig, resolve_resource_dir


_baseline_cache: dict[str, tuple[pd.DataFrame, float]] = {}
_regression_weight_cache: dict[str, pd.DataFrame] = {}
_MAX_CONDITION_NUMBER = 100_000.0
_IRLS_UPDATES = 2


def _get_cached_baseline(
    resource_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, float]:
    """Load and cache pre-computed baseline LD scores."""
    rdir = resolve_resource_dir(resource_dir)
    key = str(rdir)
    if key in _baseline_cache:
        return _baseline_cache[key]

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
    result = (pd.concat(frames, ignore_index=True), M_total)
    _baseline_cache[key] = result
    return result


def load_baseline(
    resource_dir: str | Path | None = None,
    *,
    copy: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Load pre-computed baseline LD scores and total M_5_50.

    By default returns a defensive copy so external callers cannot mutate the
    module-level cache. Internal performance-sensitive callers can request
    ``copy=False`` to reuse the cached DataFrame directly.
    """
    baseline, m_total = _get_cached_baseline(resource_dir)
    return (baseline.copy(deep=True), m_total) if copy else (baseline, m_total)


def _get_cached_regression_weights(
    resource_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load and cache LD-based regression weights (w_ld)."""
    rdir = resolve_resource_dir(resource_dir)
    key = str(rdir)
    if key in _regression_weight_cache:
        return _regression_weight_cache[key]

    weight_dir = rdir / "LDSC_resource" / "weights_hm3_no_hla"

    frames = []
    for chrom in range(1, 23):
        w = pd.read_csv(
            weight_dir / f"weights.{chrom}.l2.ldscore.gz",
            sep="\t", compression="gzip",
        )
        frames.append(w)
    result = pd.concat(frames, ignore_index=True)
    _regression_weight_cache[key] = result
    return result


def load_regression_weights(
    resource_dir: str | Path | None = None,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Load LD-based regression weights (w_ld).

    By default returns a defensive copy so external callers cannot mutate the
    module-level cache. Internal performance-sensitive callers can request
    ``copy=False`` to reuse the cached DataFrame directly.
    """
    weights = _get_cached_regression_weights(resource_dir)
    return weights.copy(deep=True) if copy else weights


def load_sumstats(path: str, cfg: RegressionConfig) -> pd.DataFrame:
    """Load munged GWAS summary statistics (SNP/Z/N format).

    Filters extreme chi-squared values to avoid outlier-driven estimates.
    """
    ss = pd.read_csv(path, sep="\t")
    required = {"SNP", "Z", "N"}
    missing = required - set(ss.columns)
    if missing:
        raise ValueError(f"Sumstats missing columns: {missing}. Found: {list(ss.columns)}")

    ss["Z"] = pd.to_numeric(ss["Z"], errors="coerce")
    ss["N"] = pd.to_numeric(ss["N"], errors="coerce")
    ss = ss.dropna(subset=["SNP", "Z", "N"])
    ss = ss[np.isfinite(ss["Z"]) & np.isfinite(ss["N"]) & (ss["N"] > 0)]
    if len(ss) == 0:
        raise ValueError("No valid SNPs remain after filtering invalid Z/N values.")
    chisq_max = max(cfg.chisq_max_factor * ss["N"].max(), cfg.chisq_max_floor)
    ss = ss[ss["Z"] ** 2 < chisq_max]
    if len(ss) == 0:
        raise ValueError("No valid SNPs remain after chi-squared filtering.")
    if ss["SNP"].duplicated().any():
        n_duplicates = int(ss["SNP"].duplicated(keep=False).sum())
        raise ValueError(
            f"Sumstats contains {n_duplicates} rows with duplicate SNP "
            "identifiers; provide one unambiguous row per SNP"
        )
    return ss[["SNP", "Z", "N"]].reset_index(drop=True)


def _sldsc_weights(
    y: np.ndarray,
    baseline_ld: np.ndarray,
    w_ld_vals: np.ndarray,
    N: np.ndarray | float,
    M_total: float,
    *,
    h2: float | None = None,
    intercept: float = 1.0,
) -> np.ndarray:
    """S-LDSC regression weights: heteroscedasticity × LD correction.

    w_s = 1 / (2 · E[χ²_s]² · w_ld_s)
    """
    y = np.asarray(y, dtype=np.float64)
    baseline_ld = np.asarray(baseline_ld, dtype=np.float64)
    w_ld_vals = np.asarray(w_ld_vals, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("S-LDSC response must be a finite, non-negative vector")
    if baseline_ld.ndim != 2 or baseline_ld.shape[0] != len(y):
        raise ValueError("baseline_ld must be a 2D matrix aligned with y")
    if not np.all(np.isfinite(baseline_ld)):
        raise ValueError("Baseline LD scores must be finite")
    if w_ld_vals.shape != y.shape:
        raise ValueError("Regression LD weights must align with y")
    try:
        N = np.broadcast_to(np.asarray(N, dtype=np.float64), y.shape)
    except ValueError as exc:
        raise ValueError("Per-SNP sample sizes must align with y") from exc
    if M_total <= 0 or not np.isfinite(M_total):
        raise ValueError("M_total must be finite and > 0")
    if np.any(N <= 0) or not np.all(np.isfinite(N)):
        raise ValueError("Per-SNP sample sizes must be finite and > 0")
    if np.any(w_ld_vals <= 0) or not np.all(np.isfinite(w_ld_vals)):
        raise ValueError("Regression LD weights must be finite and > 0")

    x_tot = baseline_ld.sum(axis=1)
    if h2 is None:
        denominator = float(np.mean(N * x_tot))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Total baseline LD score must have a positive finite mean")
        h2 = M_total * (float(y.mean()) - 1.0) / denominator
    if not np.isfinite(h2):
        raise ValueError("S-LDSC weighting heritability must be finite")
    h2 = float(np.clip(h2, 0.0, 1.0))
    intercept = float(intercept)
    if not np.isfinite(intercept):
        raise ValueError("S-LDSC weighting intercept must be finite")
    expected_chisq = intercept + h2 * N / M_total * np.maximum(x_tot, 1.0)
    if not np.all(np.isfinite(expected_chisq)):
        raise ValueError("Expected chi-squared values are non-finite")
    expected_magnitude = np.maximum(
        np.abs(expected_chisq), np.finfo(np.float64).eps,
    )
    return 1.0 / (2.0 * expected_magnitude**2 * w_ld_vals)


def _solve_identifiable(
    X: np.ndarray,
    y: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Solve a full-rank, well-conditioned least-squares problem."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"Invalid regression shapes in {context}")
    if X.shape[0] < X.shape[1]:
        raise ValueError(
            f"Regression design is underdetermined in {context}: "
            f"{X.shape[0]} rows for {X.shape[1]} coefficients"
        )
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Regression inputs contain non-finite values in {context}")

    column_scale = np.linalg.norm(X, axis=0)
    if np.any(column_scale <= 0) or not np.all(np.isfinite(column_scale)):
        raise ValueError(f"Regression design is rank-deficient in {context}")
    X_normalized = X / column_scale

    beta_normalized, _, rank, singular_values = np.linalg.lstsq(
        X_normalized, y, rcond=None,
    )
    if rank != X.shape[1] or len(singular_values) == 0 or singular_values[-1] <= 0:
        raise ValueError(f"Regression design is rank-deficient in {context}")
    condition_number = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition_number) or condition_number > _MAX_CONDITION_NUMBER:
        raise ValueError(
            f"Regression design is ill-conditioned in {context} "
            f"(condition number {condition_number:.3g} > {_MAX_CONDITION_NUMBER:.3g}); "
            "the requested conditional coefficients are not stably identifiable"
        )
    return beta_normalized / column_scale


def _iterative_sldsc_weights(
    y: np.ndarray,
    baseline_ld: np.ndarray,
    w_ld_vals: np.ndarray,
    N: np.ndarray,
    M_total: float,
) -> np.ndarray:
    """Estimate standard S-LDSC weights with two aggregate IRLS updates."""
    Nbar = float(np.mean(N))
    if not np.isfinite(Nbar) or Nbar <= 0:
        raise ValueError("Mean sample size must be finite and > 0")
    x_tot = np.asarray(baseline_ld, dtype=np.float64).sum(axis=1)
    aggregate_design = np.column_stack([N / Nbar * x_tot, np.ones(len(y))])
    weights = _sldsc_weights(y, baseline_ld, w_ld_vals, N, M_total)

    for _ in range(_IRLS_UPDATES):
        sqrtw = np.sqrt(weights)
        beta = _solve_identifiable(
            aggregate_design * sqrtw[:, None],
            y * sqrtw,
            context="aggregate S-LDSC weight model",
        )
        h2 = M_total * float(beta[0]) / Nbar
        weights = _sldsc_weights(
            y,
            baseline_ld,
            w_ld_vals,
            N,
            M_total,
            h2=h2,
            intercept=float(beta[1]),
        )
    return weights


def _is_positive_scalar_multiple(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> bool:
    """Return whether two finite vectors differ only by a positive scalar.

    The tight tolerance accommodates floating-point sparse matrix products
    without treating merely correlated annotations as the same predictor.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        return False
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return False

    right_norm_sq = float(right @ right)
    if right_norm_sq <= 0.0:
        return False
    scale = float((left @ right) / right_norm_sq)
    if not np.isfinite(scale) or scale <= 0.0:
        return False

    comparison_atol = atol * max(1.0, float(np.max(np.abs(left))))
    return bool(np.allclose(left, scale * right, rtol=rtol, atol=comparison_atol))


def _order_genomically(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Sort the regression rows into genomic order.

    _block_jackknife forms its blocks from runs of consecutive rows, so a block
    is a contiguous genomic region only when the rows arrive sorted by position.
    A frame merged onto the summary statistics inherits the munged file's row
    order instead, and that order is not always genomic. The ``Jackknife
    Ordering`` sheet in Supplementary Table 5 documents the affected trait
    frames in the current analysis; when the order is not genomic, a block is a
    random sample drawn from the whole genome rather than one region. The point
    estimate is unaffected either way, but the jackknife standard error and
    z-score can change materially.

    The baseline annotation is written per chromosome in position order, so its
    row index is the reference. SNPs absent from the baseline keep their
    relative order and are placed last; in practice the frame is an inner join
    on the baseline, so there are none.
    """
    order = pd.Series(np.arange(len(baseline)), index=baseline["SNP"])
    pos = df["SNP"].map(order)
    return (df.assign(_genomic_pos=pos)
              .sort_values("_genomic_pos", kind="stable", na_position="last")
              .drop(columns="_genomic_pos")
              .reset_index(drop=True))


def _merge_regression_inputs(
    sumstats: pd.DataFrame,
    baseline: pd.DataFrame,
    annotations: pd.DataFrame,
    annotation_cols: list[str],
    w_ld: pd.DataFrame,
) -> pd.DataFrame:
    """Perform a validated one-to-one SNP merge in genomic order."""
    frames = {
        "sumstats": sumstats,
        "baseline": baseline,
        "annotations": annotations,
        "regression weights": w_ld,
    }
    for label, frame in frames.items():
        if not frame.columns.is_unique:
            raise ValueError(f"{label} contains duplicate column names")
        if "SNP" not in frame.columns:
            raise ValueError(f"{label} is missing the SNP column")
        if frame["SNP"].duplicated().any():
            raise ValueError(f"{label} contains duplicate SNP identifiers")

    baseline_cols = [c for c in baseline.columns if c != "SNP"]
    if not baseline_cols:
        raise ValueError("baseline must provide at least one LD-score column")
    missing_annotations = [c for c in annotation_cols if c not in annotations]
    if missing_annotations:
        raise ValueError(
            f"annotations is missing requested columns: {missing_annotations}"
        )
    if "L2" not in w_ld:
        raise ValueError("regression weights is missing the L2 column")
    overlap = set(baseline_cols) & set(annotation_cols)
    if overlap:
        raise ValueError(
            f"Baseline and custom annotation names overlap: {sorted(overlap)}"
        )
    if len(annotation_cols) != len(set(annotation_cols)):
        raise ValueError("annotation column names must be unique")

    df = (
        sumstats
        .merge(baseline, on="SNP", how="inner", validate="one_to_one")
        .merge(
            annotations[["SNP"] + annotation_cols],
            on="SNP",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            w_ld[["SNP", "L2"]],
            on="SNP",
            how="inner",
            validate="one_to_one",
        )
        .rename(columns={"L2": "w_ld"})
    )
    return _order_genomically(df, baseline)


def _block_jackknife(
    Xw: np.ndarray, yw: np.ndarray, n_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast block jackknife for weighted least squares.

    Blocks are runs of consecutive rows, so the caller must hand over a frame
    in genomic order; see _order_genomically.

    Precomputes X'X and X'y, then obtains each delete-one-block estimate
    by subtracting the block's contribution — O(p³) per block instead of
    O(n·p²) for a full re-regression.

    Uses LDSC-compatible floor-spaced separators when n is not divisible by
    n_blocks.
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    yw = np.asarray(yw, dtype=np.float64)
    if Xw.ndim != 2 or yw.ndim != 1 or Xw.shape[0] != yw.shape[0]:
        raise ValueError("Invalid weighted-regression shapes for block jackknife")
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    n, p = Xw.shape
    if n_blocks > n:
        raise ValueError(
            f"n_blocks ({n_blocks}) cannot exceed the number of SNPs ({n})"
        )

    # Unit-norm columns keep the fast Gram-matrix subtraction stable and make
    # the condition threshold describe predictor geometry rather than units.
    column_scale = np.linalg.norm(Xw, axis=0)
    if np.any(column_scale <= 0) or not np.all(np.isfinite(column_scale)):
        raise ValueError("Regression design is rank-deficient in full S-LDSC model")
    X_unit = Xw / column_scale
    beta_hat_unit = _solve_identifiable(
        X_unit, yw, context="full S-LDSC model",
    )
    XtX = X_unit.T @ X_unit
    Xty = X_unit.T @ yw

    # Block-level contributions
    # Match LDSC's approximately equal contiguous genomic blocks exactly.
    separators = np.floor(np.linspace(0, n, n_blocks + 1)).astype(int)
    blocks = [np.arange(separators[b], separators[b + 1]) for b in range(n_blocks)]
    beta_delete = np.zeros((n_blocks, p))
    for b, idx in enumerate(blocks):
        Xb = X_unit[idx]
        yb = yw[idx]
        XtX_del = XtX - Xb.T @ Xb
        Xty_del = Xty - Xb.T @ yb
        if n - len(idx) < p:
            raise ValueError(
                "A delete-block regression is underdetermined; use fewer "
                "jackknife blocks or more SNPs"
            )
        eigenvalues, eigenvectors = np.linalg.eigh(XtX_del)
        if eigenvalues[0] <= 0:
            raise ValueError(
                f"Regression design becomes rank-deficient after deleting block {b}"
            )
        delete_condition = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
        if delete_condition > _MAX_CONDITION_NUMBER:
            raise ValueError(
                f"Regression design becomes ill-conditioned after deleting block {b} "
                f"(condition number {delete_condition:.3g})"
            )
        beta_delete[b] = eigenvectors @ (
            (eigenvectors.T @ Xty_del) / eigenvalues
        )

    # Pseudovalues → estimate and SE
    pseudo = n_blocks * beta_hat_unit[None, :] - (n_blocks - 1) * beta_delete
    beta_hat = beta_hat_unit / column_scale
    se = np.sqrt(np.var(pseudo, axis=0, ddof=1) / n_blocks) / column_scale

    return beta_hat, se


def run_sldsc_custom(
    sumstats: pd.DataFrame,
    baseline: pd.DataFrame,
    annot_ld: pd.DataFrame,
    w_ld: pd.DataFrame,
    M_total: float,
    cfg: RegressionConfig,
    annot_cols: list[str] | None = None,
) -> dict:
    """Run joint S-LDSC regression with an arbitrary set of annotation columns."""
    if annot_cols is None:
        annot_cols = [c for c in annot_ld.columns if c != "SNP"]
    if not annot_cols:
        raise ValueError("annot_ld must provide at least one annotation column")

    df = _merge_regression_inputs(
        sumstats,
        baseline,
        annot_ld,
        annot_cols,
        w_ld,
    )
    n_snps = len(df)
    if n_snps == 0:
        raise ValueError("No SNPs remain after merging sumstats, baseline, annotations, and weights.")

    y = (df["Z"] ** 2).values
    N = df["N"].values
    Nbar = N.mean()

    baseline_cols = [c for c in baseline.columns if c != "SNP"]
    all_annot = baseline_cols + annot_cols
    ell_arrays = [df[c].values.astype(np.float64) for c in all_annot]
    X = np.column_stack([N / Nbar * ell for ell in ell_arrays] + [np.ones(n_snps)])

    w_ld_vals = np.maximum(df["w_ld"].values, 1.0)
    w = _iterative_sldsc_weights(
        y, df[baseline_cols].values, w_ld_vals, N, M_total,
    )

    sqrtw = np.sqrt(w)
    Xw = X * sqrtw[:, None]
    yw = y * sqrtw

    beta_hat, jk_se = _block_jackknife(Xw, yw, cfg.n_blocks)

    results = {"n_snps": n_snps, "N_bar": float(Nbar), "M_total": float(M_total)}
    for i, name in enumerate(all_annot):
        tau = float(beta_hat[i] / Nbar)
        se = float(jk_se[i] / Nbar)
        if not np.isfinite(se) or se <= 0:
            raise ValueError(f"Non-positive jackknife standard error for {name!r}")
        z = tau / se
        results[name] = {
            "tau": tau, "se": se, "z": z,
            "p_twosided": float(2 * norm.sf(abs(z))),
            "p_onesided": float(norm.sf(z)),
        }
    results["intercept"] = float(beta_hat[-1])

    return results


def run_sldsc(
    sumstats: pd.DataFrame,
    baseline: pd.DataFrame,
    annot_ld: pd.DataFrame,
    w_ld: pd.DataFrame,
    M_total: float,
    cfg: RegressionConfig,
) -> dict:
    """Run joint S-LDSC with node and aggregate spatial LR-gene annotations."""
    return run_sldsc_custom(
        sumstats, baseline, annot_ld, w_ld, M_total, cfg,
        annot_cols=["ell_node", "ell_edge"],
    )


def run_per_pair_ldsc_custom(
    sumstats: pd.DataFrame,
    baseline: pd.DataFrame,
    annot_ld_controls: pd.DataFrame,
    pair_ld_scores: dict[str, np.ndarray],
    snp_names: list[str],
    w_ld: pd.DataFrame,
    M_total: float,
    cfg: RegressionConfig,
    control_cols: list[str] | None = None,
    pair_membership_ld_scores: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Rank LR-context constituent-gene annotations with arbitrary controls.

    ``pair_membership_ld_scores`` remains available for API compatibility and
    for non-collinear alternative annotations. If an own-membership vector is a
    positive scalar multiple of its score-scaled context vector, the two
    predictors cannot identify separate effects and this function raises a
    ``ValueError`` rather than silently relying on a pseudoinverse.
    """
    if control_cols is None:
        control_cols = [c for c in annot_ld_controls.columns if c != "SNP"]
    if not control_cols:
        raise ValueError("annot_ld_controls must contain at least one control annotation")

    df_base = _merge_regression_inputs(
        sumstats,
        baseline,
        annot_ld_controls,
        control_cols,
        w_ld,
    )

    if len(snp_names) != len(set(snp_names)):
        raise ValueError("snp_names contains duplicate SNP identifiers")
    snp_to_idx = {s: i for i, s in enumerate(snp_names)}
    df_snp_indices = df_base["SNP"].map(snp_to_idx)
    valid = df_snp_indices.notna()
    df_base = df_base[valid].copy()
    snp_idx = df_snp_indices[valid].astype(int).values

    n_snps = len(df_base)
    if n_snps == 0:
        raise ValueError("No SNPs remain after merging sumstats, baseline, control annotations, and weights.")

    y = (df_base["Z"] ** 2).values
    N = df_base["N"].values
    Nbar = N.mean()

    baseline_cols = [c for c in baseline.columns if c != "SNP"]
    baseline_ld = df_base[baseline_cols].values.astype(np.float64)
    control_arrays = [df_base[c].values.astype(np.float64) for c in control_cols]
    w_ld_vals = np.maximum(df_base["w_ld"].values, 1.0)

    w = _iterative_sldsc_weights(y, baseline_ld, w_ld_vals, N, M_total)
    sqrtw = np.sqrt(w)
    yw = y * sqrtw

    X_base = np.column_stack(
        [N / Nbar * baseline_ld[:, i] for i in range(baseline_ld.shape[1])]
        + [N / Nbar * arr for arr in control_arrays]
        + [np.ones(n_snps)]
    )

    results = []
    skipped_unidentifiable: dict[str, str] = {}
    for pname, pair_ell_full in pair_ld_scores.items():
        pair_ell_full = np.asarray(pair_ell_full, dtype=np.float64)
        if pair_ell_full.ndim != 1 or len(pair_ell_full) != len(snp_names):
            raise ValueError(
                f"LR-context LD scores for {pname!r} must be a vector aligned "
                "with snp_names"
            )
        if not np.all(np.isfinite(pair_ell_full)):
            raise ValueError(f"LR-context LD scores for {pname!r} are non-finite")
        pair_ell = pair_ell_full[snp_idx]
        if pair_ell.max() <= 0:
            continue

        # Build: baseline + controls [+ non-collinear alternative] + context + intercept.
        extra_cols = []
        if pair_membership_ld_scores is not None and pname in pair_membership_ld_scores:
            membership_full = np.asarray(
                pair_membership_ld_scores[pname], dtype=np.float64
            )
            if membership_full.ndim != 1 or len(membership_full) != len(snp_names):
                raise ValueError(
                    f"Membership LD scores for {pname!r} must be a vector "
                    "aligned with snp_names"
                )
            if not np.all(np.isfinite(membership_full)):
                raise ValueError(
                    f"Membership LD scores for {pname!r} are non-finite"
                )
            membership_ell = membership_full[snp_idx]
            if _is_positive_scalar_multiple(pair_ell, membership_ell):
                raise ValueError(
                    "Cannot include both the score-scaled LR-context annotation "
                    f"and its own constituent-gene membership annotation for {pname!r}: "
                    "they are positive scalar multiples, so separate pair identity "
                    "or communication effects are not identifiable."
                )
            extra_cols.append(N / Nbar * membership_ell)

        X = np.column_stack(
            [X_base[:, :-1]]
            + extra_cols
            + [N / Nbar * pair_ell, X_base[:, -1:]]
        )
        Xw = X * sqrtw[:, None]
        try:
            beta_hat, jk_se = _block_jackknife(Xw, yw, cfg.n_blocks)
            tau = float(beta_hat[-2] / Nbar)
            se = float(jk_se[-2] / Nbar)
            if not np.isfinite(se) or se <= 0:
                raise ValueError("non-positive jackknife standard error")
        except ValueError as exc:
            # A sparse context can lose all identifying variation when one LD
            # block is deleted. That context has no valid jackknife ranking,
            # but it should not invalidate other independently fitted contexts.
            skipped_unidentifiable[pname] = str(exc)
            continue
        z = tau / se
        results.append({"pair": pname, "tau": tau, "se": se, "z": z})

    df_results = pd.DataFrame(results, columns=["pair", "tau", "se", "z"])
    if len(df_results) > 0:
        df_results = df_results.sort_values("z", ascending=False).reset_index(drop=True)
    df_results.attrs["skipped_unidentifiable"] = skipped_unidentifiable
    if skipped_unidentifiable:
        warnings.warn(
            f"Skipped {len(skipped_unidentifiable)} LR contexts whose conditional "
            "coefficients were not identifiable in every jackknife replicate.",
            RuntimeWarning,
        )
    return df_results


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
    """Rank LR-context constituent-gene annotations: baseline + node + context."""
    return run_per_pair_ldsc_custom(
        sumstats=sumstats,
        baseline=baseline,
        annot_ld_controls=annot_ld_node,
        pair_ld_scores=pair_ld_scores,
        snp_names=snp_names,
        w_ld=w_ld,
        M_total=M_total,
        cfg=cfg,
        control_cols=["ell_node"],
    )
