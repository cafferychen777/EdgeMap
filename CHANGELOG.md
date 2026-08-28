# Changelog

This changelog tracks public package, CLI, API, and release-facing metadata changes. It does not track manuscript drafting history or local analysis artifacts.

## [Unreleased]

### Changed
- Moved the source test suite from `src/edgemap/tests` to the top-level
  `tests/` directory as a repository/CI asset, and added CI coverage for
  linting, source tests, and the installed wheel release surface.
- Pinned the build backend below the first release that emits
  `Metadata-Version: 2.5`, keeping source and wheel distributions compatible
  with the current release validation toolchain.

## [0.3.0] - 2026-08-09

### Changed
- Spatial LR activity now uses a union-symmetrized Gaussian KNN graph and a row-normalized neighboring-ligand mean, so uniform expression remains a uniform-activity null across graph degrees; cells with no valid neighbor are excluded from LR specificity rather than encoded as biological zeros.
- Node-score ranks now preserve expression ties and are invariant to joint cell-order permutations.
- S-LDSC weighting now uses per-SNP sample sizes with two aggregate IRLS updates, while the regression design is scaled by mean sample size for numerical stability.
- Expression scale can be declared explicitly with `input_scale="raw_counts"` or `input_scale="log1p"`; the previous `preprocessed` flag remains a compatibility alias.
- Result JSON now records the package version and complete scientific configuration; the LR subunit prevalence threshold is configurable as `min_lr_cell_pct` / `--min-lr-cell-pct`.

### Fixed
- Rank-deficient, underdetermined, or severely ill-conditioned conditional regression designs now fail explicitly instead of returning arbitrary pseudoinverse coefficients.
- Block-jackknife variance now uses the sample variance of pseudovalues (`ddof=1`) and rejects fewer than two blocks or non-positive standard errors.
- Reused output directories no longer retain a stale `per_pair_sldsc.csv` from an incompatible prior run; managed outputs are written through atomic replacement.
- LR expression prevalence thresholds now use a ceiling, and configuration, coordinates, expression scale, SNP uniqueness, and annotation names receive explicit validation.
- Source distributions now retain the bundled LR database while excluding repository-local temporary directories and `node_modules` license artifacts.

## [0.2.0] - 2026-06-04

### Added
- `build_multi_annotation_ldscores()`: generalized annotation builder that accepts an arbitrary number of gene-level score vectors with pairwise diagnostic correlations.
- `node_agg_percentile` config parameter for sensitivity analysis of node score aggregation (default 100.0 = max, matching v0.1 behavior).
- Quickstart example (`quickstart/`) with a synthetic data generator and aggregate-workflow smoke test.

### Changed
- `build_annotation_ldscores()` now delegates to `build_multi_annotation_ldscores()` internally; the public API is unchanged.
- Regression module refactored for clarity.

## [0.1.0] - 2026-03-30

### Added
- Initial public package release metadata, including machine-readable citation support.
- Public changelog and GitHub release-note configuration.

### Changed
- README and package metadata were tightened to present a clearer package-first public surface.
- Installation metadata now matches runtime requirements more closely.

### Fixed
- Test coverage was extended to cover remaining high-value control paths in config validation, score aggregation, CSC reuse, and top-level pipeline input validation.

### Performance
- Regression-resource reuse and sparse-matrix handling are covered by targeted tests to protect current performance behavior.
