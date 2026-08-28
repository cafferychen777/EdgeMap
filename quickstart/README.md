# EdgeMap Quickstart

Minimal smoke test of the aggregate EdgeMap workflow. The script generates
synthetic spatial transcriptomics and GWAS inputs, builds the node and spatial
LR-gene annotations, and runs aggregate S-LDSC.

This example requires the external gsMap resource bundle. It validates the
installation, aggregate workflow, and output schemas; it does not reproduce the
manuscript analyses or run the separate 50,000-replicate empirical LR-context
calibration. With the maintained default seed, the aggregate screen is
negative and the LR-context gene-set branch is not triggered.

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

The maintained fixed-seed example takes approximately 12 seconds for the
reported EdgeMap computation on our test machine after the resource directory
is installed; dependency acquisition, Python startup, and file generation add
machine-dependent overhead. All output is written to `quickstart/output/`.

## What the script does

1. **Generates synthetic Visium data** (500 spots, ~200 genes) with three
   spatial domains and complementary ligand/receptor expression patterns used
   to construct spatially concentrated LR-gene scores.

2. **Generates synthetic GWAS summary statistics** (~1.2M HapMap3 SNPs)
   with random Z-scores mimicking a moderately polygenic trait.

3. **Runs the aggregate EdgeMap path**:
   - Loads ST data and builds a union-symmetrized spatial KNN graph
   - Computes a degree-normalized spatial LR activity proxy across neighborhoods
   - Computes node scores (expression specificity) and aggregate LR-gene
     scores (`edge` in the output schema) for each gene
   - Maps gene-level scores to SNP-level annotation LD scores via the
     gsMap SNP-gene weight matrix
   - Runs joint stratified LD score regression (S-LDSC) against the synthetic
     GWAS, testing the conditional association of the aggregate LR-gene
     annotation after baseline and node controls

## Output files

| File | Description |
|------|-------------|
| `results.json` | Pipeline summary: parameters, regression coefficients (tau, z, p), and annotation diagnostics |
| `lr_pair_stats.json` | Spatial LR-context statistics: mean activity proxy, number of active cells, and specificity score |
| `per_pair_sldsc.csv` | LR-context constituent-gene rankings (only produced when the aggregate LR-gene screen is positive at p < 0.05) |
| `demo_visium.h5ad` | The synthetic spatial transcriptomics dataset |
| `demo_gwas.tsv` | The synthetic GWAS summary statistics |

## How to interpret `results.json`

The key output is the S-LDSC regression under `"regression"`:

```
"ell_node": {                          # Node annotation (expression specificity)
    "tau": 2.91e-08,                   #   per-SNP heritability contribution
    "se": 1.00e-07,                    #   jackknife standard error
    "z": 0.291,                        #   z-score = tau / se
    "p_onesided": 0.385                #   one-sided p-value (H1: tau > 0)
},
"ell_edge": {                          # Aggregate spatial LR-gene annotation
    "tau": -1.60e-08,
    "se": 6.98e-08,
    "z": -0.228,
    "p_onesided": 0.590
},
"intercept": 1.154                     # Expect ~1.0; >>1 suggests confounding
```

**tau** is the per-SNP annotation coefficient: how much each additional unit
of the annotation's LD score contributes to expected chi-squared statistics.
A significantly positive `ell_edge` tau supports a conditional association
between trait heritability and the aggregate spatial LR-gene annotation after
the included baseline and node controls. It does not by itself establish a
causal communication mechanism.

**Interpretation for real data**: When the aggregate screen is positive,
EdgeMap additionally ranks annotations formed from the constituent genes of
active LR contexts. Each such annotation is a positive score-scaled membership
vector, so its `z` is invariant to that score and to ligand/receptor
orientation. The ranking is therefore directionless and gene-set based; it
does not identify the LR relation or interaction. The conditional z-scores are
also not formal tests without empirical calibration.

**Note**: The synthetic demo data uses random GWAS Z-scores, so the
results are not biologically meaningful. With the fixed seed, the aggregate
LR-gene result does not cross the one-sided 0.05 screen
(`edge_significant=false`), so the LR-context gene-set branch is not run.
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
- `.X`: raw integer counts (cells/spots x genes)
- `.obsm["spatial"]`: 2D spatial coordinates (n_spots x 2)
- `.var_names`: gene symbols (must be unique)

**GWAS summary statistics** (tab-separated text):
- `SNP`: rsID (must overlap HapMap3 baseline SNPs)
- `Z`: Z-score (signed effect / SE)
- `N`: per-SNP sample size

Use LDSC's `munge_sumstats.py` to prepare GWAS data from standard formats.
