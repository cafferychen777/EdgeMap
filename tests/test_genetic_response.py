"""Tests for frozen pair-response genetic regression primitives."""

import numpy as np

from edgemap.genetic_response import (
    empirical_permutation_summary,
    fwl_block_jackknife_many,
    sldsc_regression_weights,
)


def test_fwl_block_jackknife_recovers_primary_effect():
    rng = np.random.default_rng(101)
    n_rows = 1200
    nuisance = rng.normal(size=(n_rows, 3))
    primary = rng.normal(size=(n_rows, 4))
    base = np.column_stack([np.ones(n_rows), nuisance])
    response = (
        0.8 * nuisance[:, 0]
        + 0.4 * primary[:, 0]
        + rng.normal(scale=0.7, size=n_rows)
    )
    estimate, standard_error = fwl_block_jackknife_many(
        base,
        primary,
        response,
        np.ones(n_rows),
        n_blocks=30,
    )
    assert estimate[0] > 0.3
    assert estimate[0] / standard_error[0] > 5.0
    assert np.max(np.abs(estimate[1:])) < 0.08


def test_sldsc_weights_and_empirical_summary_are_finite():
    rng = np.random.default_rng(103)
    baseline = rng.uniform(0.1, 3.0, size=(100, 5))
    response = rng.chisquare(df=1, size=100)
    weights = sldsc_regression_weights(
        response,
        baseline,
        np.ones(100),
        mean_sample_size=100_000.0,
        total_snps=1_000_000.0,
    )
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)

    summary = empirical_permutation_summary(
        2.0,
        np.asarray([-1.0, 0.0, 1.0, 3.0]),
    )
    assert summary["empirical_p_greater"] == 0.4
    assert summary["empirical_p_two_sided"] == 0.4
