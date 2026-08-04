"""Tests for pair-response statistical primitives."""

import numpy as np

from edgemap.response import (
    bh_adjust,
    block_jackknife_association,
    build_crossfit_designs,
    conservative_response_weights,
    coordinate_basis,
    crossfit_residualize,
    fixed_program_empirical_test,
    holm_adjust,
    matched_weight_strata,
    max_t_adjust,
    permute_weights_within_strata,
    random_effects_meta,
    residualize_pairwise_main_effects,
    spatial_block_permutations,
    spatial_blocks,
    spatial_crossfit_folds,
)


def make_grid(side: int = 12) -> np.ndarray:
    axis = np.arange(side, dtype=float)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([x.ravel(), y.ravel()])


def test_spatial_blocks_are_balanced_and_permutations_are_bijective():
    coords = make_grid()
    labels = spatial_blocks(coords, 8)
    counts = np.bincount(labels)
    assert len(counts) == 8
    assert counts.max() - counts.min() <= 1

    permutations = spatial_block_permutations(coords, 8, 4, seed=17)
    identity = np.arange(len(coords))
    for permutation in permutations:
        assert np.array_equal(np.sort(permutation), identity)
        assert not np.array_equal(permutation, identity)

    folds = spatial_crossfit_folds(coords, n_blocks=24, n_folds=6)
    blocks = spatial_blocks(coords, 24)
    fold_counts = np.bincount(folds)
    maximum_block_size = np.bincount(blocks).max()
    assert fold_counts.max() - fold_counts.min() <= maximum_block_size
    for block in np.unique(blocks):
        assert len(np.unique(folds[blocks == block])) == 1


def test_crossfit_residualization_removes_train_learned_nuisance():
    rng = np.random.default_rng(19)
    coords = make_grid()
    folds = spatial_blocks(coords, 6)
    condition = rng.normal(size=(len(coords), 20))
    base = np.column_stack([coordinate_basis(coords), condition[:, 0]])
    outcome = (
        3.0 * condition[:, [0]]
        - 2.0 * base[:, [0]]
        + rng.normal(scale=0.2, size=(len(coords), 1))
    )
    designs = build_crossfit_designs(
        condition,
        base,
        folds,
        n_components=5,
        seed=23,
    )
    residual = crossfit_residualize(outcome, designs, rank_normalize=False)
    assert residual.shape == outcome.shape
    assert np.isfinite(residual).all()
    assert abs(np.corrcoef(residual[:, 0], condition[:, 0])[0, 1]) < 0.15


def test_block_jackknife_and_random_effects_recover_direction():
    rng = np.random.default_rng(29)
    effects = []
    errors = []
    for section in range(5):
        exposure = rng.normal(size=(240, 2))
        response = np.column_stack(
            [
                0.35 * exposure[:, 0] + rng.normal(size=240),
                rng.normal(size=240),
            ]
        )
        blocks = np.repeat(np.arange(12), 20)
        beta, standard_error = block_jackknife_association(
            exposure,
            response,
            blocks,
        )
        effects.append(beta)
        errors.append(standard_error)
    meta = random_effects_meta(np.stack(effects), np.stack(errors))
    assert meta.effect[0, 0] > 0
    assert meta.p_value[0, 0] < 0.05
    assert meta.prediction_high.shape == (2, 2)


def test_pairwise_residualization_removes_endpoint_main_effects():
    rng = np.random.default_rng(27)
    n_rows = 1000
    n_pairs = 25
    receptor = rng.normal(size=(n_rows, n_pairs))
    ligand = 0.25 * receptor + rng.normal(size=(n_rows, n_pairs))
    interaction = rng.normal(size=(n_rows, n_pairs))
    raw = interaction + 2.0 * receptor - 3.0 * ligand
    residual = residualize_pairwise_main_effects(
        raw,
        receptor,
        ligand,
        np.arange(800),
    )
    train = np.arange(800)
    receptor_covariance = np.mean(
        residual[train] * receptor[train],
        axis=0,
    )
    ligand_covariance = np.mean(
        residual[train] * ligand[train],
        axis=0,
    )
    recovery = np.asarray(
        [
            np.corrcoef(residual[:, column], interaction[:, column])[0, 1]
            for column in range(n_pairs)
        ]
    )
    assert np.max(np.abs(receptor_covariance)) < 1e-10
    assert np.max(np.abs(ligand_covariance)) < 1e-10
    assert np.median(recovery) > 0.99


def test_conservative_weights_require_all_specifications_and_loo_stability():
    beta = np.asarray(
        [
            [
                [0.4, 0.4, 0.4, 0.4],
                [0.5, 0.5, -0.4, 0.5],
                [0.3, 0.6, 0.5, 0.3],
                [0.4, 0.5, -0.6, 0.4],
                [0.6, 0.4, 0.5, 0.6],
            ],
            [
                [0.3, 0.3, 0.3, -0.3],
                [0.4, 0.4, -0.3, -0.4],
                [0.5, 0.5, 0.4, -0.5],
                [0.3, 0.6, -0.5, -0.3],
                [0.4, 0.5, 0.4, -0.4],
            ],
        ]
    )
    weights, diagnostics = conservative_response_weights(
        beta,
        minimum_sign_sections=4,
    )
    assert weights[0] > 0
    assert weights[1] > 0
    assert weights[2] == 0
    assert weights[3] == 0
    assert diagnostics["eligible_by_specification"].shape == (2, 4)
    assert not diagnostics["cross_spec_sign_agreement"][3]


def test_matched_permutations_preserve_each_stratum_distribution():
    rng = np.random.default_rng(31)
    covariates = rng.normal(size=(80, 4))
    weights = np.linspace(0.0, 2.0, len(covariates))
    strata = matched_weight_strata(
        covariates,
        target_size=10,
        minimum_size=4,
        seed=37,
    )
    permutations = permute_weights_within_strata(
        weights,
        strata,
        n_permutations=5,
        seed=41,
    )
    for label in np.unique(strata):
        indices = np.flatnonzero(strata == label)
        expected = np.sort(weights[indices])
        for permutation in permutations:
            assert np.allclose(np.sort(permutation[indices]), expected)


def test_fixed_program_empirical_test_replicates_signed_target_pattern():
    rng = np.random.default_rng(43)
    beta_by_specification = {}
    signed_pattern = np.asarray([1.0, -1.0, 1.0])
    for specification in ["pc8", "pc12", "pc20"]:
        beta = rng.normal(scale=0.08, size=(4, 12, 6))
        beta[:, 0, :3] += 0.8 * signed_pattern
        beta_by_specification[specification] = beta
    result = fixed_program_empirical_test(
        beta_by_specification,
        target_indices=np.asarray([0, 1, 2]),
        signed_weights=signed_pattern,
        pair_index=0,
        exposure_controls=np.arange(1, 12),
        section_groups={
            "platform_a": np.asarray([0, 1]),
            "platform_b": np.asarray([2, 3]),
        },
        n_null=1000,
        seed=47,
    )
    assert result.statistic > 3
    assert result.empirical_p_value < 0.01
    assert all(
        np.all(scores > 0)
        for scores in result.leave_one_section_out_scores.values()
    )


def test_multiplicity_adjustments_are_monotone_and_bounded():
    p_values = np.asarray([0.001, 0.04, 0.2])
    adjusted = bh_adjust(p_values)
    assert np.all((adjusted >= p_values) & (adjusted <= 1.0))
    holm = holm_adjust(p_values)
    assert np.allclose(holm, [0.003, 0.08, 0.2])

    observed = np.asarray([2.0, 1.0])
    null = np.asarray([[0.5, 0.2], [1.5, 0.9], [2.5, 1.2]])
    max_t = max_t_adjust(observed, null)
    assert np.all((max_t > 0) & (max_t <= 1.0))
