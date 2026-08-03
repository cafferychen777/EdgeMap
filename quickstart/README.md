# EdgeMap Quickstart

Minimal smoke test of the aggregate EdgeMap workflow. The script generates
synthetic spatial transcriptomics and GWAS inputs, builds the node and edge
annotations, and runs aggregate S-LDSC.

This example requires the external gsMap resource bundle. It validates the
installation, aggregate workflow, and output schemas; it does not reproduce the
manuscript analyses or run the separate 50,000-replicate empirical per-pair
calibration. With the maintained default seed, the aggregate screen is
negative and the conditional per-pair branch is not triggered.

## Prerequisites

```bash
pip install edgemap
```

EdgeMap also requires the **external gsMap resource directory**. The upstream
archive is approximately 621 MiB to download (650,877,553 bytes as checked on
3 August 2026); extracted size depends on the release and filesystem. It
contains pre-computed baseline LD scores, regression weights, and a SNP-gene
weight matrix for HapMap3 SNPs. It is not bundled with EdgeMap or downloaded by
the quickstart. Install it first as described in the repository README, then
point EdgeMap to the extracted directory via one of:

- Environment variable: `export EDGEMAP_RESOURCE_DIR=/path/to/gsMap_resource`
- Command-line flag: `--resource-dir /path/to/gsMap_resource`
- Auto-detection: place `data/gsMap_resource/` relative to the package root

## Run

```bash
python quickstart/run_quickstart.py
```

Or with explicit resource path:

```bash
python quickstart/run_quickstart.py --resource-dir /path/to/gsMap_resource
```

The maintained fixed-seed example takes approximately 4 seconds for the
reported EdgeMap computation on our test machine after the resource directory
is installed; dependency acquisition, Python startup, and file generation add
machine-dependent overhead. All output is written to `quickstart/output/`.

## What the script does

1. **Generates synthetic Visium data** (500 spots, ~200 genes) with three
   spatial domains and complementary ligand/receptor expression patterns
   that create spatially concentrated cell-cell communication at domain
   boundaries.

2. **Generates synthetic GWAS summary statistics** (~1.2M HapMap3 SNPs)
   with random Z-scores mimicking a moderately polygenic trait.

3. **Runs the aggregate EdgeMap path**:
   - Loads ST data and builds a spatial KNN graph
   - Computes LR communication intensity across spatial neighborhoods
   - Computes node scores (expression specificity) and edge scores
     (communication specificity) for each gene
   - Maps gene-level scores to SNP-level annotation LD scores via the
     gsMap SNP-gene weight matrix
   - Runs joint stratified LD score regression (S-LDSC) against the
     synthetic GWAS, testing whether communication carries trait
     heritability beyond what expression alone explains

## Output files

| File | Description |
|------|-------------|
| `results.json` | Pipeline summary: parameters, regression coefficients (tau, z, p), and annotation diagnostics |
| `lr_pair_stats.json` | Per-LR-pair statistics: mean communication intensity, number of active cells, and specificity score |
| `per_pair_sldsc.csv` | Per-pair conditional S-LDSC results (only produced when the aggregate edge tau is significant at p < 0.05) |
| `demo_visium.h5ad` | The synthetic spatial transcriptomics dataset |
| `demo_gwas.tsv` | The synthetic GWAS summary statistics |

## How to interpret `results.json`

The key output is the S-LDSC regression under `"regression"`:

```
"ell_node": {                          # Node annotation (expression specificity)
    "tau": -1.57e-07,                  #   per-SNP heritability contribution
    "se": 9.77e-08,                    #   jackknife standard error
    "z": -1.609,                       #   z-score = tau / se
    "p_onesided": 0.946                #   one-sided p-value (H1: tau > 0)
},
"ell_edge": {                          # Edge annotation (communication)
    "tau": 1.17e-07,
    "se": 7.27e-08,
    "z": 1.605,
    "p_onesided": 0.054
},
"intercept": 1.157                     # Expect ~1.0; >>1 suggests confounding
```

**tau** is the per-SNP heritability coefficient: how much each additional
unit of the annotation's LD score contributes to expected chi-squared
statistics. A significantly positive tau_edge means that SNPs near genes
involved in spatially concentrated cell-cell communication explain more
trait heritability than expected from baseline genomic features and
expression specificity alone.

**Interpretation for real data**: On real Visium + GWAS data, a significant
edge tau (p < 0.05) indicates that intercellular communication in the
profiled tissue carries unique trait heritability. When the edge is
significant, EdgeMap additionally runs per-LR-pair conditional regressions.
Those conditional z-scores rank candidate pairs but are not formal per-pair
tests without empirical calibration.

**Note**: The synthetic demo data uses random GWAS Z-scores, so the
results are not biologically meaningful. With the fixed seed, the aggregate
edge result does not cross the one-sided 0.05 screen
(`edge_significant=false`), so the conditional per-pair branch is not run.
This behavior is intentional: the example does not manufacture an association
merely to trigger a downstream branch.

## Adapting to your own data

Replace the two input files:

```python
cfg = PipelineConfig(
    st_h5ad="your_tissue.h5ad",         # AnnData with .obsm["spatial"] and raw counts
    gwas_sumstats="your_trait.tsv",      # Munged GWAS: tab-separated SNP/Z/N
    gwas_label="your_trait_name",
    output_dir="results/your_analysis",
)
results = edgemap.run(cfg)
```

Or via the command line:

```bash
edgemap --st your_tissue.h5ad \
        --gwas your_trait.tsv \
        --gwas-label your_trait_name \
        --output results/your_analysis
```

### Input format requirements

**Spatial transcriptomics** (`h5ad`):
- `.X`: raw integer counts (genes x cells/spots)
- `.obsm["spatial"]`: 2D spatial coordinates (n_spots x 2)
- `.var_names`: gene symbols (must be unique)

**GWAS summary statistics** (tab-separated text):
- `SNP`: rsID (must overlap HapMap3 baseline SNPs)
- `Z`: Z-score (signed effect / SE)
- `N`: per-SNP sample size

Use LDSC's `munge_sumstats.py` to prepare GWAS data from standard formats.
