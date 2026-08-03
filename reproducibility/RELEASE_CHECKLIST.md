# Publication release checklist

Do not describe the repository as a complete manuscript-reproduction archive
or create the immutable publication release until all items pass from a clean
clone. Audit documents and reviewed package fixes may be published earlier when
their incomplete scope is stated explicitly.

## Identity and provenance

- [ ] Record the final Git commit and immutable release tag in the manuscript.
- [ ] Verify that the release contains commit `8ebcd46` or an equivalent reviewed
      genomic-order implementation and its order-invariance tests.
- [ ] Record Python and package versions plus the operating system used for the
      frozen analysis.
- [ ] Give every external input an accession/DOI, version or retrieval date,
      expected file name, and checksum when redistribution terms permit.
- [ ] Review redistribution terms for each GWAS, spatial dataset, LIANA-derived
      file, gsMap resource, reference panel, and external annotation.

## Scientific and implementation review

- [ ] Confirm that every paper estimate was produced by the tagged core package.
- [ ] Re-run the aggregate and per-pair order-invariance regression tests.
- [ ] Confirm that formal per-pair claims use the empirical calibration, not the
      uncalibrated normal approximation.
- [ ] Verify seeds, replicate counts, sidedness, multiple-testing families, and
      inclusion rules against the Methods and frozen outputs.
- [ ] Remove or quarantine smoke, dry-run, superseded, and exploratory outputs.
- [ ] Resolve every duplicate analysis family to one maintained entry point.

## Portability and separation

- [ ] Analysis programs accept input/output roots as explicit arguments or a
      documented config; no personal absolute path is required.
- [ ] Analysis programs export CSV/JSON and do not create publication figures.
- [ ] Visualization programs read only frozen CSV/JSON and static public assets;
      they do not import analysis programs or rerun heavy computation.
- [ ] Every manifest path resolves inside a clean clone or an immutable DOI.
- [ ] All 77 final SI sheets appear exactly once in the publication manifest;
      all nine final workbooks regenerate from one tracked builder.
- [x] Extended Data contains eight figures and two tables (ten combined items).
- [ ] Verify that Extended Data Figure 1c is regenerated from the frozen
      top-15 rank estimand documented in `reproducibility/ED_FIGURE_AUDIT.md`.
- [ ] The SI builder fails closed when a frozen CSV/JSON input is absent and
      verifies input checksums before writing workbooks.
- [ ] Supplementary Table 7's observed controlled pair results and 50,000-draw
      null chunks have been restored, audited, frozen and checksummed.
- [ ] Every final figure regenerates byte-for-byte or passes a documented visual
      equivalence check from frozen inputs alone.

## Security, licensing, and repository hygiene

- [ ] Run a dedicated secret scanner over tracked files and Git history.
- [ ] Scan tracked text for `/Users/`, `/home/`, `/scratch/`, private hosts,
      personal configuration, tokens, and credentials.
- [ ] Confirm licenses and required notices for bundled code and data.
- [ ] Reject files above the repository size limit unless they use an approved
      archival data mechanism.
- [ ] Check all Markdown and manuscript links from a clean clone.
- [ ] Build the sdist and wheel; inspect their file lists for accidental data,
      tests, manuscript files, caches, or personal artifacts.

## Verification commands

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
rg -n '/Users/|/home/|/scratch/' $(git ls-files)
rg -n -i 'api[_-]?key|access[_-]?token|password|private[_-]?key' $(git ls-files)
git ls-files -z | xargs -0 du -k | sort -n
git diff --check
```

The two regular-expression checks are triage only. Human review and a secret
scanner are still required.
