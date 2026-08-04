"""Statistical primitives for identifiable pair-response programs.

The functions in this module are intentionally independent of GWAS outcomes.
They support the frozen per-pair workflow by providing spatial block
cross-fitting, section-level uncertainty estimates, random-effects summaries,
conservative target weights, and matched within-target permutations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, rankdata, t
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class FoldDesign:
    """One spatial cross-fitting fold with a train-fitted nuisance design."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    matrix: np.ndarray
    condition_number: float
    explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class RandomEffectsResult:
    """Random-effects meta-analysis arrays with small-sample inference."""

    effect: np.ndarray
    standard_error: np.ndarray
    p_value: np.ndarray
    tau_squared: np.ndarray
    q_statistic: np.ndarray
    i_squared: np.ndarray
    prediction_low: np.ndarray
    prediction_high: np.ndarray


@dataclass(frozen=True)
class FixedProgramTestResult:
    """Empirical replication test for one frozen signed target program."""

    statistic: float
    null_statistics: np.ndarray
    empirical_p_value: float
    unit_observed_scores: dict[str, float]
    leave_one_section_out_scores: dict[str, np.ndarray]


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def standardize_columns(
    values: np.ndarray,
    reference_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns using either all rows or a training subset."""

    array = _as_2d(values, "values")
    reference = (
        array
        if reference_indices is None
        else array[np.asarray(reference_indices, dtype=np.int64)]
    )
    center = reference.mean(axis=0)
    scale = reference.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return (array - center) / scale, center, scale


def rank_inverse_normal(values: np.ndarray) -> np.ndarray:
    """Apply a tie-aware inverse-normal transform column by column."""

    array = _as_2d(values, "values")
    n_rows = len(array)
    transformed = np.empty_like(array)
    for column in range(array.shape[1]):
        ranks = rankdata(array[:, column], method="average")
        probabilities = (ranks - 0.375) / (n_rows + 0.25)
        transformed[:, column] = norm.ppf(
            np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        )
    return standardize_columns(transformed)[0]


def spatial_blocks(coords: np.ndarray, n_blocks: int) -> np.ndarray:
    """Construct deterministic, balanced, contiguous recursive spatial blocks."""

    coordinates = _as_2d(coords, "coords")
    if coordinates.shape[1] < 2:
        raise ValueError("coords must contain at least two spatial dimensions")
    if n_blocks < 2 or n_blocks > len(coordinates):
        raise ValueError("n_blocks must be between 2 and the number of rows")

    blocks = [np.arange(len(coordinates), dtype=np.int64)]
    while len(blocks) < n_blocks:
        splittable = [
            index for index, members in enumerate(blocks) if len(members) > 1
        ]
        if not splittable:
            raise RuntimeError("Unable to construct the requested spatial blocks")

        def split_priority(index: int) -> tuple[int, float]:
            members = blocks[index]
            span = np.ptp(coordinates[members, :2], axis=0)
            return len(members), float(span.max())

        block_index = max(splittable, key=split_priority)
        members = blocks.pop(block_index)
        span = np.ptp(coordinates[members, :2], axis=0)
        axis = int(np.argmax(span))
        order = np.argsort(
            coordinates[members, axis],
            kind="mergesort",
        )
        midpoint = len(order) // 2
        left = members[order[:midpoint]]
        right = members[order[midpoint:]]
        blocks.extend([left, right])

    labels = np.empty(len(coordinates), dtype=np.int64)
    ordered_blocks = sorted(
        blocks,
        key=lambda members: tuple(
            coordinates[members, :2].mean(axis=0).tolist()
        ),
    )
    for label, members in enumerate(ordered_blocks):
        labels[members] = label
    return labels


def spatial_crossfit_folds(
    coords: np.ndarray,
    n_blocks: int,
    n_folds: int,
) -> np.ndarray:
    """Assign separated contiguous blocks to balanced spatial folds."""

    if n_folds < 2 or n_blocks < n_folds:
        raise ValueError("Spatial cross-fitting requires blocks >= folds >= 2")
    if n_blocks % n_folds != 0:
        raise ValueError("n_blocks must be divisible by n_folds")
    block_labels = spatial_blocks(coords, n_blocks)
    return block_labels % n_folds


def spatial_block_permutations(
    coords: np.ndarray,
    n_blocks: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Return bijections that move intact, equally sized spatial blocks."""

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    coordinates = _as_2d(coords, "coords")
    labels = spatial_blocks(coordinates, n_blocks)
    members = [
        np.flatnonzero(labels == block)
        for block in range(n_blocks)
    ]
    for block, indices in enumerate(members):
        centered = coordinates[indices, :2] - coordinates[
            indices, :2
        ].mean(axis=0)
        angle = np.arctan2(centered[:, 1], centered[:, 0])
        radius = np.square(centered).sum(axis=1)
        local_order = np.lexsort((radius, angle))
        members[block] = indices[local_order]

    size_groups: dict[int, list[int]] = {}
    for block, indices in enumerate(members):
        size_groups.setdefault(len(indices), []).append(block)
    if max(map(len, size_groups.values())) < 2:
        raise ValueError("No equal-sized spatial blocks are available to permute")

    rng = np.random.default_rng(seed)
    permutations = np.empty(
        (n_permutations, len(coordinates)),
        dtype=np.int64,
    )
    identity = np.arange(len(coordinates), dtype=np.int64)
    for permutation_index in range(n_permutations):
        mapping = identity.copy()
        moved = 0
        for group in size_groups.values():
            if len(group) < 2:
                continue
            shift = int(rng.integers(1, len(group)))
            shuffled = np.asarray(group, dtype=np.int64)[
                rng.permutation(len(group))
            ]
            destinations = np.roll(shuffled, shift)
            for source, destination in zip(
                shuffled,
                destinations,
                strict=True,
            ):
                mapping[members[source]] = members[destination]
                moved += len(members[source])
        if moved == 0 or np.array_equal(mapping, identity):
            raise RuntimeError("Spatial block permutation did not move any rows")
        if len(np.unique(mapping)) != len(mapping):
            raise RuntimeError("Spatial block permutation is not bijective")
        permutations[permutation_index] = mapping
    return permutations


def coordinate_basis(coords: np.ndarray) -> np.ndarray:
    """Create a fixed, outcome-free low-frequency spatial nuisance basis."""

    coordinates = standardize_columns(_as_2d(coords, "coords")[:, :2])[0]
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    basis = np.column_stack(
        [
            x,
            y,
            x * y,
            x**2,
            y**2,
            np.sin(np.pi * x / 2.0),
            np.cos(np.pi * x / 2.0),
            np.sin(np.pi * y / 2.0),
            np.cos(np.pi * y / 2.0),
            np.sqrt(x**2 + y**2),
        ]
    )
    return standardize_columns(basis)[0]


def build_crossfit_designs(
    condition_expression: np.ndarray,
    base_covariates: np.ndarray,
    fold_labels: np.ndarray,
    n_components: int,
    seed: int,
) -> list[FoldDesign]:
    """Fit nuisance PCA and scaling separately outside each spatial fold."""

    condition = _as_2d(condition_expression, "condition_expression")
    base = _as_2d(base_covariates, "base_covariates")
    folds = np.asarray(fold_labels, dtype=np.int64)
    if len(condition) != len(base) or len(folds) != len(condition):
        raise ValueError("Cross-fitting inputs do not have the same row count")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("At least two spatial folds are required")

    designs: list[FoldDesign] = []
    for fold_position, fold in enumerate(unique_folds):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if n_components >= min(len(train), condition.shape[1]):
            raise ValueError("n_components is too large for a training fold")

        standardized_condition = standardize_columns(condition, train)[0]
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=seed + fold_position,
        )
        pca.fit(standardized_condition[train])
        factors = pca.transform(standardized_condition)
        nuisance = np.column_stack([base, factors])
        nuisance = standardize_columns(nuisance, train)[0]
        matrix = np.column_stack([np.ones(len(nuisance)), nuisance])
        designs.append(
            FoldDesign(
                train_indices=train,
                test_indices=test,
                matrix=matrix,
                condition_number=float(np.linalg.cond(matrix[train])),
                explained_variance_ratio=pca.explained_variance_ratio_.copy(),
            )
        )
    return designs


def crossfit_residualize(
    values: np.ndarray,
    designs: list[FoldDesign],
    rank_normalize: bool = True,
) -> np.ndarray:
    """Return out-of-fold residuals from train-only nuisance regressions."""

    array = _as_2d(values, "values")
    residual = np.full_like(array, np.nan)
    seen = np.zeros(len(array), dtype=np.int64)
    for design in designs:
        train = design.train_indices
        test = design.test_indices
        coefficients, *_ = np.linalg.lstsq(
            design.matrix[train],
            array[train],
            rcond=None,
        )
        residual[test] = array[test] - design.matrix[test] @ coefficients
        seen[test] += 1
    if not np.all(seen == 1):
        raise RuntimeError("Cross-fitting folds do not partition the rows")
    if rank_normalize:
        return rank_inverse_normal(residual)
    return standardize_columns(residual)[0]


def residualize_pairwise_main_effects(
    exposures: np.ndarray,
    receptor_values: np.ndarray,
    local_ligand_values: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Residualize each LR product from its ligand and receptor main effects."""

    values = _as_2d(exposures, "exposures")
    receptor = _as_2d(receptor_values, "receptor_values")
    ligand = _as_2d(local_ligand_values, "local_ligand_values")
    train = np.asarray(train_indices, dtype=np.int64)
    if receptor.shape != values.shape or ligand.shape != values.shape:
        raise ValueError("Pairwise exposure and main-effect arrays must align")
    if (
        train.ndim != 1
        or len(train) < 3
        or np.any(train < 0)
        or np.any(train >= len(values))
    ):
        raise ValueError("train_indices are invalid")

    receptor_train = receptor[train]
    ligand_train = ligand[train]
    exposure_train = values[train]
    s_rr = np.square(receptor_train).sum(axis=0)
    s_ll = np.square(ligand_train).sum(axis=0)
    s_rl = (receptor_train * ligand_train).sum(axis=0)
    s_re = (receptor_train * exposure_train).sum(axis=0)
    s_le = (ligand_train * exposure_train).sum(axis=0)
    determinant = s_rr * s_ll - np.square(s_rl)
    scale = np.maximum(s_rr * s_ll, 1.0)
    stable = np.abs(determinant) > 1e-10 * scale
    coefficient_r = np.zeros(values.shape[1], dtype=np.float64)
    coefficient_l = np.zeros(values.shape[1], dtype=np.float64)
    coefficient_r[stable] = (
        s_re[stable] * s_ll[stable] - s_le[stable] * s_rl[stable]
    ) / determinant[stable]
    coefficient_l[stable] = (
        s_le[stable] * s_rr[stable] - s_re[stable] * s_rl[stable]
    ) / determinant[stable]
    for column in np.flatnonzero(~stable):
        design = np.column_stack(
            [receptor_train[:, column], ligand_train[:, column]]
        )
        coefficients, *_ = np.linalg.lstsq(
            design,
            exposure_train[:, column],
            rcond=None,
        )
        coefficient_r[column], coefficient_l[column] = coefficients
    return (
        values
        - receptor * coefficient_r[None, :]
        - ligand * coefficient_l[None, :]
    )


def block_jackknife_association(
    exposure: np.ndarray,
    response: np.ndarray,
    block_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate section associations and spatial block-jackknife standard errors."""

    exposure_array = _as_2d(exposure, "exposure")
    response_array = _as_2d(response, "response")
    blocks = np.asarray(block_labels, dtype=np.int64)
    if len(exposure_array) != len(response_array) or len(blocks) != len(
        exposure_array
    ):
        raise ValueError("Association inputs do not have the same row count")
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) < 3:
        raise ValueError("At least three jackknife blocks are required")

    total = exposure_array.T @ response_array
    estimate = total / len(exposure_array)
    delete_sum = np.zeros_like(estimate)
    delete_square_sum = np.zeros_like(estimate)
    for block in unique_blocks:
        indices = np.flatnonzero(blocks == block)
        remaining = len(exposure_array) - len(indices)
        if remaining < 1:
            raise ValueError("A jackknife block contains all observations")
        deleted = (
            total - exposure_array[indices].T @ response_array[indices]
        ) / remaining
        delete_sum += deleted
        delete_square_sum += np.square(deleted)
    mean_deleted = delete_sum / len(unique_blocks)
    centered_sum = np.maximum(
        delete_square_sum
        - len(unique_blocks) * np.square(mean_deleted),
        0.0,
    )
    variance = (
        (len(unique_blocks) - 1) / len(unique_blocks) * centered_sum
    )
    return estimate, np.sqrt(variance)


def random_effects_meta(
    effects: np.ndarray,
    standard_errors: np.ndarray,
) -> RandomEffectsResult:
    """Run DerSimonian-Laird random effects with modified Hartung-Knapp tests."""

    effect_array = np.asarray(effects, dtype=np.float64)
    error_array = np.asarray(standard_errors, dtype=np.float64)
    if effect_array.shape != error_array.shape or effect_array.ndim < 1:
        raise ValueError("Effect and standard-error arrays must align")
    if effect_array.shape[0] < 3:
        raise ValueError("At least three sections are required")
    if np.any(~np.isfinite(effect_array)) or np.any(~np.isfinite(error_array)):
        raise ValueError("Meta-analysis inputs contain non-finite values")
    error_array = np.maximum(error_array, 1e-8)

    n_sections = effect_array.shape[0]
    fixed_weights = 1.0 / np.square(error_array)
    fixed_weight_sum = fixed_weights.sum(axis=0)
    fixed_effect = (
        (fixed_weights * effect_array).sum(axis=0) / fixed_weight_sum
    )
    q_statistic = (
        fixed_weights * np.square(effect_array - fixed_effect)
    ).sum(axis=0)
    degrees_freedom = n_sections - 1
    denominator = fixed_weight_sum - (
        np.square(fixed_weights).sum(axis=0) / fixed_weight_sum
    )
    tau_squared = np.maximum(
        (q_statistic - degrees_freedom) / np.maximum(denominator, 1e-30),
        0.0,
    )
    random_weights = 1.0 / (
        np.square(error_array) + tau_squared[None, ...]
    )
    random_weight_sum = random_weights.sum(axis=0)
    pooled = (
        (random_weights * effect_array).sum(axis=0) / random_weight_sum
    )
    residual_q = (
        random_weights * np.square(effect_array - pooled)
    ).sum(axis=0)
    hk_scale = np.maximum(residual_q / degrees_freedom, 1.0)
    pooled_se = np.sqrt(hk_scale / random_weight_sum)
    statistic = pooled / np.maximum(pooled_se, 1e-30)
    p_value = 2.0 * t.sf(np.abs(statistic), degrees_freedom)
    i_squared = np.where(
        q_statistic > 0,
        np.maximum((q_statistic - degrees_freedom) / q_statistic, 0.0),
        0.0,
    )
    critical = float(t.ppf(0.975, degrees_freedom))
    prediction_se = np.sqrt(tau_squared + np.square(pooled_se))
    return RandomEffectsResult(
        effect=pooled,
        standard_error=pooled_se,
        p_value=p_value,
        tau_squared=tau_squared,
        q_statistic=q_statistic,
        i_squared=i_squared,
        prediction_low=pooled - critical * prediction_se,
        prediction_high=pooled + critical * prediction_se,
    )


def conservative_response_weights(
    beta_by_specification: np.ndarray,
    minimum_sign_sections: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Construct weights that remain positive across all frozen specifications."""

    beta = np.asarray(beta_by_specification, dtype=np.float64)
    if beta.ndim != 3:
        raise ValueError(
            "beta_by_specification must have specification, section, target axes"
        )
    n_specifications, n_sections, _ = beta.shape
    if minimum_sign_sections < 2 or minimum_sign_sections > n_sections:
        raise ValueError("minimum_sign_sections is outside the valid range")

    sign_counts = np.maximum(
        (beta > 0).sum(axis=1),
        (beta < 0).sum(axis=1),
    )
    cross_products = []
    for specification in range(n_specifications):
        current = beta[specification]
        products = [
            current[left] * current[right]
            for left in range(n_sections)
            for right in range(left + 1, n_sections)
        ]
        cross_products.append(np.mean(np.stack(products), axis=0))
    energy = np.sqrt(np.maximum(np.stack(cross_products), 0.0))

    loo_stable = np.ones((n_specifications, beta.shape[2]), dtype=bool)
    full_sign = np.sign(beta.mean(axis=1))
    for held_out in range(n_sections):
        keep = np.arange(n_sections) != held_out
        loo_sign = np.sign(beta[:, keep, :].mean(axis=1))
        loo_stable &= (loo_sign == full_sign) & (loo_sign != 0)

    eligible_by_specification = (
        (sign_counts >= minimum_sign_sections)
        & loo_stable
        & (energy > 0)
    )
    cross_spec_sign_agreement = (
        (full_sign == full_sign[[0], :]).all(axis=0)
        & (full_sign != 0).all(axis=0)
    )
    consensus_sign = full_sign[0].copy()
    consensus_sign[~cross_spec_sign_agreement] = 0.0
    robust = np.min(energy, axis=0)
    robust[
        ~(
            eligible_by_specification.all(axis=0)
            & cross_spec_sign_agreement
        )
    ] = 0.0
    positive = robust > 0
    if np.any(positive):
        robust[positive] /= robust[positive].mean()
    diagnostics = {
        "energy_by_specification": energy,
        "sign_counts": sign_counts,
        "loo_stable": loo_stable,
        "eligible_by_specification": eligible_by_specification,
        "cross_spec_sign_agreement": cross_spec_sign_agreement,
        "consensus_sign": consensus_sign,
    }
    return robust, diagnostics


def fixed_program_empirical_test(
    beta_by_specification: dict[str, np.ndarray],
    target_indices: np.ndarray,
    signed_weights: np.ndarray,
    pair_index: int,
    exposure_controls: np.ndarray,
    section_groups: dict[str, np.ndarray],
    n_null: int,
    seed: int,
) -> FixedProgramTestResult:
    """Test a frozen signed target program against alternative exposures.

    A control exposure is sampled independently for every held-out section and
    the same sampled mappings are reused across nuisance specifications and
    platform groups. The final statistic is the minimum standardized score
    across those units.
    """

    if not beta_by_specification:
        raise ValueError("At least one beta specification is required")
    targets = np.asarray(target_indices, dtype=np.int64)
    weights = np.asarray(signed_weights, dtype=np.float64)
    controls = np.asarray(exposure_controls, dtype=np.int64)
    if (
        targets.ndim != 1
        or weights.ndim != 1
        or len(targets) != len(weights)
        or len(targets) == 0
    ):
        raise ValueError("Target indices and signed weights must align")
    if not np.all(np.isfinite(weights)) or np.abs(weights).sum() <= 0:
        raise ValueError("Signed weights must be finite and nonzero")
    if controls.ndim != 1 or len(controls) < 2:
        raise ValueError("At least two exposure controls are required")
    if n_null < 100:
        raise ValueError("n_null must be at least 100")

    arrays = {
        name: np.asarray(beta, dtype=np.float64)
        for name, beta in beta_by_specification.items()
    }
    reference_shape = next(iter(arrays.values())).shape
    if len(reference_shape) != 3:
        raise ValueError("Beta arrays require section, pair, and gene axes")
    for name, beta in arrays.items():
        if beta.shape != reference_shape:
            raise ValueError(f"Beta shape differs for specification {name}")
        if not np.all(np.isfinite(beta)):
            raise ValueError(f"Non-finite beta values in specification {name}")
    n_sections, n_pairs, n_genes = reference_shape
    if not 0 <= pair_index < n_pairs:
        raise ValueError("pair_index is outside the beta pair axis")
    if np.any(targets < 0) or np.any(targets >= n_genes):
        raise ValueError("Target indices are outside the beta gene axis")
    if np.any(controls < 0) or np.any(controls >= n_pairs):
        raise ValueError("Exposure controls are outside the beta pair axis")
    if pair_index in controls:
        raise ValueError("The tested pair cannot be an exposure control")

    normalized_weights = weights / np.abs(weights).sum()
    normalized_groups: dict[str, np.ndarray] = {}
    covered_sections: set[int] = set()
    for group_name, raw_sections in section_groups.items():
        sections = np.asarray(raw_sections, dtype=np.int64)
        if len(sections) < 1 or len(np.unique(sections)) != len(sections):
            raise ValueError(f"Invalid section group {group_name}")
        if np.any(sections < 0) or np.any(sections >= n_sections):
            raise ValueError(f"Section group {group_name} is out of range")
        overlap = covered_sections.intersection(map(int, sections))
        if overlap:
            raise ValueError("Section groups must not overlap")
        covered_sections.update(map(int, sections))
        normalized_groups[group_name] = sections
    if covered_sections != set(range(n_sections)):
        raise ValueError("Section groups must partition all sections")

    rng = np.random.default_rng(seed)
    mappings = rng.choice(
        controls,
        size=(n_null, n_sections),
        replace=True,
    )
    observed_z: list[float] = []
    null_z: list[np.ndarray] = []
    unit_observed: dict[str, float] = {}
    loo_scores: dict[str, np.ndarray] = {}
    for specification, beta in arrays.items():
        exposure_scores = np.einsum(
            "spg,g->sp",
            beta[:, :, targets],
            normalized_weights,
        )
        current_loo = np.empty(n_sections, dtype=np.float64)
        for held_out in range(n_sections):
            keep = np.arange(n_sections) != held_out
            current_loo[held_out] = exposure_scores[
                keep,
                pair_index,
            ].mean()
        loo_scores[specification] = current_loo
        for group_name, sections in normalized_groups.items():
            observed = float(exposure_scores[sections, pair_index].mean())
            null = exposure_scores[
                sections[None, :],
                mappings[:, sections],
            ].mean(axis=1)
            center = float(null.mean())
            scale = float(null.std(ddof=1))
            if scale < 1e-12:
                scale = 1.0
            observed_z.append((observed - center) / scale)
            null_z.append((null - center) / scale)
            unit_observed[f"{specification}|{group_name}"] = observed

    statistic = float(np.min(observed_z))
    null_statistics = np.min(np.stack(null_z), axis=0)
    empirical_p = float(
        (1.0 + (null_statistics >= statistic).sum())
        / (n_null + 1.0)
    )
    return FixedProgramTestResult(
        statistic=statistic,
        null_statistics=null_statistics,
        empirical_p_value=empirical_p,
        unit_observed_scores=unit_observed,
        leave_one_section_out_scores=loo_scores,
    )


def matched_weight_strata(
    covariates: np.ndarray,
    target_size: int,
    minimum_size: int,
    seed: int,
) -> np.ndarray:
    """Cluster targets into covariate-matched strata for weight permutations."""

    values = standardize_columns(_as_2d(covariates, "covariates"))[0]
    n_targets = len(values)
    if minimum_size < 2:
        raise ValueError("minimum_size must be at least two")
    if target_size < minimum_size:
        raise ValueError("target_size must be at least minimum_size")
    if n_targets < 2 * minimum_size:
        raise ValueError("Too few targets for matched weight strata")
    n_clusters = max(2, n_targets // target_size)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(values)

    while True:
        counts = np.bincount(labels)
        small = np.flatnonzero((counts > 0) & (counts < minimum_size))
        if len(small) == 0:
            break
        active = np.flatnonzero(counts >= minimum_size)
        if len(active) == 0:
            raise RuntimeError("Unable to construct non-degenerate strata")
        for cluster in small:
            indices = np.flatnonzero(labels == cluster)
            center = values[indices].mean(axis=0)
            active_centers = np.stack(
                [values[labels == candidate].mean(axis=0) for candidate in active]
            )
            destination = active[
                int(np.argmin(np.square(active_centers - center).sum(axis=1)))
            ]
            labels[indices] = destination
    unique = np.unique(labels)
    relabel = {old: new for new, old in enumerate(unique)}
    result = np.asarray([relabel[value] for value in labels], dtype=np.int64)
    if np.min(np.bincount(result)) < minimum_size:
        raise RuntimeError("Matched strata contain a degenerate group")
    return result


def permute_weights_within_strata(
    weights: np.ndarray,
    strata: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Permute fixed response weights only within matched target strata."""

    values = np.asarray(weights, dtype=np.float64)
    labels = np.asarray(strata, dtype=np.int64)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("weights and strata must be aligned vectors")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Each matched stratum must contain at least two targets")
    rng = np.random.default_rng(seed)
    result = np.empty((n_permutations, len(values)), dtype=np.float64)
    for permutation in range(n_permutations):
        result[permutation] = values
        for group in groups:
            result[permutation, group] = values[rng.permutation(group)]
    return result


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a vector in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a vector in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(
        ranked * np.arange(len(ranked), 0, -1)
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def max_t_adjust(
    observed_statistics: np.ndarray,
    null_statistics: np.ndarray,
) -> np.ndarray:
    """One-sided maxT family-wise adjusted permutation p-values."""

    observed = np.asarray(observed_statistics, dtype=np.float64)
    null = np.asarray(null_statistics, dtype=np.float64)
    if observed.ndim != 1 or null.ndim != 2 or null.shape[1] != len(observed):
        raise ValueError("Observed and null statistics do not align")
    maximum_null = null.max(axis=1)
    return np.asarray(
        [
            (1.0 + np.sum(maximum_null >= statistic)) / (len(null) + 1.0)
            for statistic in observed
        ],
        dtype=np.float64,
    )
