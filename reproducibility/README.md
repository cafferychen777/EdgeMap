# EdgeMap publication reproducibility contract

This directory defines what an external researcher needs to verify the paper
without conflating package installation, numerical reproduction, and access to
third-party data. It is also a release gate: a manuscript artifact is not
described as reproducible from the repository until every required item in
`PUBLICATION_MANIFEST.csv` has a public, versioned path and has passed the
checks in `RELEASE_CHECKLIST.md`.

## Minimal sufficient public record

The publication record has six layers.

1. **Core package.** The installable `edgemap` package, its bundled LR database,
   tests, configuration schema, and the genomic-order block-jackknife fix must
   be in the exact tagged release used for the paper.
2. **Analysis programs.** One maintained entry point per analysis family must
   read explicit command-line paths and export machine-readable CSV or JSON.
   Cluster submission wrappers are optional and must not be the only entry
   point. Exploratory, superseded, smoke-test, and path-bound scripts are not
   publication programs.
3. **Visualization programs.** Plotting must be separate from analysis. A
   visualization program may read only frozen derived CSV/JSON files and public
   static assets; it must regenerate its figure without rerunning an analysis.
4. **Frozen derived results.** The exact, compact CSV/JSON inputs consumed by
   each visualization and table builder must be archived with checksums. These
   are necessary even when raw data are public, because they make every plotted
   value independently auditable.
5. **External data and resources.** Raw spatial data, GWAS summary statistics,
   gsMap resources, reference panels, and third-party annotations remain at
   their authoritative repositories when redistribution is unnecessary or not
   permitted. The archive must give accession or DOI, version or retrieval
   date, expected file name, checksum when licensing permits, and the command
   that converts each source into an analysis input.
6. **Paper assets.** The manuscript source, bibliography, final figure files,
   and final table workbooks belong in the archival publication snapshot. Build
   products, editor synchronization metadata, filled forms, and local working
   files do not belong in the software package.

This separation is deliberate. The package answers whether the method can be
run; the analysis archive answers whether the reported estimates can be
recomputed; the frozen results and visualization programs answer whether every
figure and table can be regenerated and audited.

## Current release state

The reviewed source history contains commit `8ebcd46`, which sorts regression
rows into baseline genomic order before constructing consecutive jackknife
blocks and adds order-invariance tests for both aggregate and per-pair
regression. A publication release remains blocked until this implementation,
the final publication manifests, and every artifact marked as required are in
one immutable public tag whose commit identifier is recorded in the
manuscript.

The current `scripts/` and `results/` working trees are not a safe publication
archive. They contain many exploratory and superseded programs, hard-coded
local or cluster paths, scheduler logs and wrappers, duplicated analysis
variants, and multi-gigabyte intermediate outputs. They are intentionally not
admitted wholesale by `.gitignore`. `EXCLUSION_REGISTER.md` records the
exclusion rationale; `PUBLICATION_MANIFEST.csv` records the clean replacement
or remaining release blocker for every paper artifact.

## Quickstart scope

`quickstart/run_quickstart.py` is a smoke test of the aggregate pipeline path
on generated spatial and GWAS inputs. It requires the separately installed
gsMap resource archive (approximately 621 MiB to download as checked on
3 August 2026) and does not perform the paper's 50,000-replicate empirical
per-pair calibration. With the maintained default seed, the aggregate screen
is negative and the conditional per-pair branch is not entered. It therefore
demonstrates installation and file schemas, not numerical reproduction of the
manuscript.

## Supplementary Information scope

The submission package now contains nine final Supplementary Information
workbooks with 77 sheets. `PUBLICATION_MANIFEST.csv` has one
`supplementary_sheet` row for every sheet, including guides and notes. The
publication archive should include all nine final workbooks, the compact frozen
CSV/JSON objects used to build their data sheets, and the maintained analysis
and builder entry points after they pass review.

The 24 legacy workbooks must not be published alongside the final package. The
`Legacy Mapping` sheet in Supplementary Table 1 is the authoritative mapping
from legacy content to the nine final workbooks; retaining the old workbooks
would create redundant and potentially divergent public copies.

Raw or large third-party GWAS, spatial-transcriptomics, gsMap, reference-panel,
and Kuppe deconvolution inputs are not part of this compact archive unless a
source-specific redistribution review explicitly permits them. Their manifest
rows instead require stable accessions, versions or retrieval dates, and
checksums where terms allow.

The nine workbooks currently exist as paper assets, but the clean-clone builder
that assembles the consolidated nine-workbook package is not tracked. Several
legacy sheet builders remain useful provenance, but they do not regenerate the
final filenames and sheet layout. Supplementary Information reproduction is
therefore still blocked until a single final builder is added and every
manifested frozen input is present. In particular, the observed and 50,000-draw
null directories required by the membership-controlled per-pair builder are
absent from the current local frozen-result tree and must be restored and
checksummed before release.

## Release invariant

The paper, response letter, README, GitHub tag, and archival DOI must describe
the same scope. In particular, the phrase "scripts to reproduce all analyses
and figures are included" is permitted only after every manifest row is marked
`ready`, all referenced paths resolve from a clean clone, and the frozen result
checksums have been verified.
