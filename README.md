# EdgeMap

**Edge-centric heritability mapping via spatial cell–cell communication**

EdgeMap decomposes trait heritability into cell-intrinsic (node) and cell–cell communication (edge) components using spatial transcriptomics and GWAS summary statistics. When the edge component is significant, it identifies which ligand–receptor pairs carry the signal.

Existing methods (S-LDSC, scDRS, gsMap) map genetic risk to individual cells but treat each cell as an independent unit. EdgeMap tests a complementary hypothesis: **do genetic effects also concentrate at molecular interfaces between cells?**

> Yang C, Zhang X, Chen J. *Intercellular communication is a heritable dimension of human tissue architecture.* (2026)

## How it works

1. **Spatial communication** — Gaussian-weighted spatial graph + mass-action LR communication per cell per pair
2. **Node & edge scores** — Per-gene expression specificity (node) and communication specificity (edge) as local-vs-global enrichment
3. **SNP annotation** — Gene scores → SNP LD scores via gsMap's pre-computed SNP–gene weight matrix
4. **S-LDSC regression** — Joint regression yielding τ coefficients and p-values for node and edge heritability
5. **Per-pair resolution** — Conditional testing of individual LR pairs to pinpoint the channels driving edge signal

## Installation

```bash
git clone https://github.com/cafferychen777/EdgeMap.git
cd EdgeMap
pip install -e .
```

Dependencies (numpy, scipy, scanpy, scikit-learn, anndata) are installed automatically. Requires Python ≥ 3.10.

## Input preparation

### 1. Spatial transcriptomics data

An AnnData with a gene expression matrix and spatial coordinates in `.obsm["spatial"]`.

**From 10x Space Ranger output** (most common):

```python
import scanpy as sc

adata = sc.read_visium("/path/to/spaceranger/outs")
```

**From other platforms** (Slide-seq, MERFISH, STARmap, etc.): create an AnnData with your expression matrix and store coordinates as `adata.obsm["spatial"]` (shape `n_cells × 2`).

Requirements:
- **Raw counts** — EdgeMap applies its own normalization. If your data is already log1p-normalized, set `preprocessed` (see [Parameters](#parameters)).
- **Human gene symbols** — the bundled LR database ([LIANA Consensus](https://github.com/saezlab/liana), 4,624 pairs) uses human symbols. For non-human data, convert gene names to human orthologs first.
- For CLI usage, save to `.h5ad` first: `adata.write("my_tissue.h5ad")`

### 2. GWAS summary statistics

Tab-separated file with columns `SNP`, `Z`, `N` — the standard output of [ldsc munge_sumstats](https://github.com/bulik/ldsc):

```bash
python munge_sumstats.py \
    --sumstats raw_gwas.txt \
    --out munged_trait \
    --merge-alleles w_hm3.snplist
```

The output `munged_trait.sumstats.gz` can be passed directly to EdgeMap.

### 3. gsMap resource directory

EdgeMap requires pre-computed LD resources from [gsMap](https://github.com/LeonSong1995/gsMap):

```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz
tar -xzf gsMap_resource.tar.gz
```

Expected structure after extraction:

```
gsMap_resource/
├── quick_mode/
│   ├── baseline/                      # Baseline LD scores and M files
│   │   ├── baseline.{1..22}.l2.ldscore.feather
│   │   └── baseline.{1..22}.l2.M_5_50
│   └── snp_gene_weight_matrix.h5ad    # SNP–gene cis-window weight matrix
└── LDSC_resource/
    └── weights_hm3_no_hla/            # LD regression weights
        └── weights.{1..22}.l2.ldscore.gz
```

EdgeMap finds this directory via (in priority order):

1. `--resource-dir` (CLI) or `resource_dir=` (Python)
2. `EDGEMAP_RESOURCE_DIR` environment variable
3. Auto-detection: `data/gsMap_resource/` relative to installation

## Usage

### Command line

```bash
edgemap \
    --st my_tissue.h5ad \
    --gwas munged_trait.sumstats.gz \
    --gwas-label "Systolic blood pressure" \
    --output results/sbp_heart \
    --resource-dir /path/to/gsMap_resource
```

### Python API

```python
import scanpy as sc
import edgemap

adata = sc.read_visium("/path/to/spaceranger/outs")

edgemap.run(edgemap.PipelineConfig(
    gwas_sumstats="munged_trait.sumstats.gz",
    gwas_label="Systolic blood pressure",
    output_dir="results/sbp_heart",
    resource_dir="/path/to/gsMap_resource",
), adata=adata)

# Results stored in adata
adata.var["node_score"]   # per-gene expression specificity
adata.var["edge_score"]   # per-gene communication specificity
adata.uns["edgemap"]      # full results dict
```

For file-based workflows (e.g. batch scripts), pass a path instead:

```python
results = edgemap.run(edgemap.PipelineConfig(
    st_h5ad="my_tissue.h5ad",
    gwas_sumstats="munged_trait.sumstats.gz",
    gwas_label="Systolic blood pressure",
    output_dir="results/sbp_heart",
    resource_dir="/path/to/gsMap_resource",
))
```

### Parameters

| CLI | Python | Default | When to change |
|-----|--------|---------|----------------|
| `--k-spatial` | `spatial.k_spatial` | 6 | Increase for denser tissues (e.g. 10 for brain cortex), decrease for sparser layouts |
| `--dis-thr` | `spatial.dis_thr` | 3000 | Distance threshold in **coordinate units** (same as `.obsm["spatial"]`). For Visium pixel coordinates, 3000 ≈ 15 spot diameters. Adjust for other platforms or unit systems |
| `--n-blocks` | `regression.n_blocks` | 200 | Jackknife blocks for standard errors. Rarely needs changing |
| `--preprocessed` | `spatial.preprocessed` | off | Set if data is already log1p-normalized to skip normalization |

Python parameter example:

```python
edgemap.run(edgemap.PipelineConfig(
    gwas_sumstats="munged_trait.sumstats.gz",
    gwas_label="Systolic blood pressure",
    output_dir="results/sbp_heart",
    spatial=edgemap.SpatialConfig(k_spatial=10, dis_thr=5000),
    regression=edgemap.RegressionConfig(n_blocks=100),
), adata=adata)
```

## Output

All files are written to `--output` (`output_dir` in Python):

### `results.json`

| Field | Meaning |
|-------|---------|
| `regression.ell_node.tau / .z / .p_onesided` | Node (expression specificity) heritability enrichment |
| `regression.ell_edge.tau / .z / .p_onesided` | Edge (communication specificity) heritability enrichment |
| `edge_significant` | `true` if edge p < 0.05 |
| `node_edge_spearman` | Correlation between node and edge scores (low = complementary signals) |

**Interpretation:** A significant edge τ means trait-associated variants are enriched near genes whose spatial communication patterns are concentrated — the trait's genetic architecture acts through intercellular signaling, beyond what cell-intrinsic expression explains.

### `per_pair_sldsc.csv`

Generated only when edge is significant. Each row is one LR pair tested conditionally against baseline + node:

| Column | Meaning |
|--------|---------|
| `pair` | LR pair label (e.g. `VEGFA-FLT1`) |
| `tau / se / z` | Pair-specific heritability coefficient |

**Important:** The `z` column is a **ranking score**, not a calibrated test statistic. Per-pair annotations are extremely sparse, causing block-jackknife standard errors to deviate from their asymptotic distribution. Use `z` to identify the top-contributing LR pairs, but do not derive p-values from it via a normal approximation. Formal per-pair significance testing requires empirical null calibration (see paper Methods). An analytical calibration solution is under development and will be released in a future version.

### `lr_pair_stats.json`

Per-LR-pair communication diagnostics: mean intensity, active cell count, and spatial specificity score for all tested pairs.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `h5ad must contain .obsm['spatial']` | Ensure spatial coordinates exist in your h5ad |
| `Expression values look pre-processed` | Provide raw counts, or add `--preprocessed` |
| `gsMap resource directory not found` | Set `EDGEMAP_RESOURCE_DIR` or pass `--resource-dir` |
| No `per_pair_sldsc.csv` in output | Expected — edge was not significant (p ≥ 0.05) |

## License

MIT
