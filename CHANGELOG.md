# Changelog

This changelog tracks public package, CLI, API, and release-facing metadata changes. It does not track manuscript drafting history or local analysis artifacts.

## [Unreleased]

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
