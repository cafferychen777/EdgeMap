"""Regression primitives for frozen pair-response genetic tests."""

from __future__ import annotations

import numpy as np


def sldsc_regression_weights(
    chi_square: np.ndarray,
    baseline_ld: np.ndarray,
    regression_ld: np.ndarray,
    mean_sample_size: float,
    total_snps: float,
) -> np.ndarray:
    """Approximate the standard S-LDSC heteroskedastic regression weights."""

    response = np.asarray(chi_square, dtype=np.float64)
    baseline = np.asarray(baseline_ld, dtype=np.float64)
    weight_ld = np.asarray(regression_ld, dtype=np.float64)
    total_ld = baseline.sum(axis=1)
    initial_h2 = np.clip(
        (response.mean() - 1.0)
        * total_snps
        / (mean_sample_size * total_ld.mean()),
        0.01,
        1.0,
    )
    expected = 1.0 + np.clip(
        initial_h2 * mean_sample_size / total_snps * total_ld,
        0.0,
        1e4,
    )
    weights = 1.0 / (
        2.0 * np.square(expected) * np.maximum(weight_ld, 1.0)
    )
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid S-LDSC regression weights")
    return weights


def fwl_block_jackknife_many(
    base_design: np.ndarray,
    annotations: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit many one-annotation FWL models against one fixed base design."""

    base = np.asarray(base_design, dtype=np.float64)
    primary = np.asarray(annotations, dtype=np.float64)
    outcome = np.asarray(response, dtype=np.float64)
    regression_weights = np.asarray(weights, dtype=np.float64)
    n_rows = len(outcome)
    if base.ndim != 2 or primary.ndim != 2:
        raise ValueError("Design matrices must be two-dimensional")
    if (
        len(base) != n_rows
        or len(primary) != n_rows
        or len(regression_weights) != n_rows
    ):
        raise ValueError("Regression inputs have different row counts")
    if n_blocks < 3 or n_blocks > n_rows:
        raise ValueError("n_blocks must be between three and the row count")
    if not all(
        np.isfinite(value).all()
        for value in [base, primary, outcome, regression_weights]
    ):
        raise ValueError("Regression inputs contain non-finite values")
    if np.any(regression_weights <= 0):
        raise ValueError("Regression weights must be positive")

    blocks = np.array_split(np.arange(n_rows), n_blocks)
    n_base = base.shape[1]
    n_annotations = primary.shape[1]
    square_root_weights = np.sqrt(regression_weights)
    btb_blocks = np.zeros((n_blocks, n_base, n_base))
    btp_blocks = np.zeros((n_blocks, n_base, n_annotations))
    bty_blocks = np.zeros((n_blocks, n_base))
    pty_blocks = np.zeros((n_blocks, n_annotations))
    ptp_blocks = np.zeros((n_blocks, n_annotations))
    for block_index, row_indices in enumerate(blocks):
        current_weight = square_root_weights[row_indices]
        weighted_base = base[row_indices] * current_weight[:, None]
        weighted_primary = primary[row_indices] * current_weight[:, None]
        weighted_outcome = outcome[row_indices] * current_weight
        btb_blocks[block_index] = weighted_base.T @ weighted_base
        btp_blocks[block_index] = weighted_base.T @ weighted_primary
        bty_blocks[block_index] = weighted_base.T @ weighted_outcome
        pty_blocks[block_index] = weighted_primary.T @ weighted_outcome
        ptp_blocks[block_index] = np.square(weighted_primary).sum(axis=0)

    btb = btb_blocks.sum(axis=0)
    btp = btp_blocks.sum(axis=0)
    bty = bty_blocks.sum(axis=0)
    pty = pty_blocks.sum(axis=0)
    ptp = ptp_blocks.sum(axis=0)
    base_coefficient = np.linalg.solve(btb, bty)
    projected_primary = np.linalg.solve(btb, btp)
    numerator = pty - btp.T @ base_coefficient
    denominator = ptp - (btp * projected_primary).sum(axis=0)
    if np.any(denominator <= 1e-20):
        raise ValueError("Primary annotation is collinear with the base model")
    full_estimate = numerator / denominator

    btb_delete = btb[None, :, :] - btb_blocks
    btp_delete = btp[None, :, :] - btp_blocks
    bty_delete = bty[None, :] - bty_blocks
    pty_delete = pty[None, :] - pty_blocks
    ptp_delete = ptp[None, :] - ptp_blocks
    base_delete = np.linalg.solve(
        btb_delete,
        bty_delete[:, :, None],
    )[:, :, 0]
    projected_delete = np.linalg.solve(btb_delete, btp_delete)
    numerator_delete = pty_delete - np.einsum(
        "bpk,bp->bk",
        btp_delete,
        base_delete,
    )
    denominator_delete = ptp_delete - np.einsum(
        "bpk,bpk->bk",
        btp_delete,
        projected_delete,
    )
    if np.any(denominator_delete <= 1e-20):
        raise ValueError("A delete block makes the primary annotation collinear")
    delete_estimate = numerator_delete / denominator_delete
    pseudo_values = (
        n_blocks * full_estimate[None, :]
        - (n_blocks - 1) * delete_estimate
    )
    standard_error = np.sqrt(
        np.var(pseudo_values, axis=0, ddof=1) / n_blocks
    )
    if np.any(standard_error <= 0) or not np.isfinite(
        standard_error
    ).all():
        raise ValueError("Invalid block-jackknife standard error")
    return full_estimate, standard_error


def empirical_permutation_summary(
    observed: float,
    null_values: np.ndarray,
) -> dict[str, float]:
    """Summarize a one-sided and two-sided empirical permutation test."""

    null = np.asarray(null_values, dtype=np.float64)
    if null.ndim != 1 or len(null) < 1 or not np.isfinite(null).all():
        raise ValueError("null_values must be a finite non-empty vector")
    if not np.isfinite(observed):
        raise ValueError("observed must be finite")
    return {
        "observed_z": float(observed),
        "null_mean_z": float(null.mean()),
        "null_sd_z": float(null.std(ddof=1)),
        "empirical_p_greater": float(
            (1.0 + np.sum(null >= observed)) / (1.0 + len(null))
        ),
        "empirical_p_two_sided": float(
            (1.0 + np.sum(np.abs(null) >= abs(observed)))
            / (1.0 + len(null))
        ),
    }
