# EdgeMap

**Edge-centric heritability mapping via spatial cell–cell communication**

EdgeMap decomposes trait heritability into cell-intrinsic (**node**) and cell–cell communication (**edge**) components using spatial transcriptomics and GWAS summary statistics. The core question is simple: genetic effects may localize not only to cells themselves, but also to the molecular interfaces between neighboring cells.

Existing methods such as S-LDSC, scDRS, and gsMap map genetic risk to individual cells. EdgeMap tests the complementary hypothesis that heritability can also concentrate in spatially structured intercellular signaling.

## How it works

1. **Spatial communication** — Build a Gaussian-weighted spatial neighbor graph (`k=6`) and compute LR communication intensity per cell using mass-action kinetics with a bottleneck model for multi-subunit complexes.
2. **Node and edge scores** — Quantify where expression is spatially concentrated (**node**) and where communication is spatially concentrated (**edge**).
3. **SNP annotation** — Map gene-level scores to SNP-level LD scores using gsMap's pre-computed SNP–gene weight matrix.
4. **S-LDSC regression** — Regress GWAS chi-squared statistics on baseline + node + edge annotations to estimate node and edge heritability enrichment.
5. **Per-pair ranking** — If the aggregate edge signal is significant, run conditional S-LDSC for individual LR pairs against baseline + node to rank the channels driving the signal.

Runtime is typically **tens of seconds to a few minutes** per trait–tissue pair, depending on tissue size, the number of active LR pairs, disk I/O, and hardware.

## Installation

```bash
git clone https://github.com/cafferychen777/EdgeMap.git
cd EdgeMap
pip install -e .
```

This installs the core Python dependencies automatically, including `numpy`, `pandas`, `pyarrow`, `scipy`, `anndata`, `scanpy`, and `scikit-learn`. Requires Python >= 3.10.

## Input preparation

### 1. Spatial transcriptomics data

Provide an AnnData object with a gene expression matrix and spatial coordinates in `.obsm["spatial"]`.

**From 10x Space Ranger output**:

```python
import scanpy as sc

adata = sc.read_visium("/path/to/spaceranger/outs")
```

**From other platforms** (Slide-seq, MERFISH, STARmap, etc.): create an AnnData object with expression in `adata.X` and coordinates in `adata.obsm["spatial"]` (shape `n_cells x 2`).

Requirements:
- **Raw counts by default** — EdgeMap normalizes and log-transforms the data unless `--preprocessed` is set.
- **Gene filtering is always applied first** — genes expressed in fewer than 10 cells are removed before the normalization check. `--preprocessed` skips normalization and log1p, but not this filtering step.
- **Human gene symbols** — the bundled LIANA Consensus database uses human symbols. For non-human data, convert genes to human orthologs first.
- For CLI usage, save the AnnData object to `.h5ad` first: `adata.write("my_tissue.h5ad")`

### 2. GWAS summary statistics

Provide a tab-separated file with columns `SNP`, `Z`, and `N` — the standard output of [ldsc munge_sumstats](https://github.com/bulik/ldsc):

```bash
python munge_sumstats.py \
    --sumstats raw_gwas.txt \
    --out munged_trait \
    --merge-alleles w_hm3.snplist
```

The output `munged_trait.sumstats.gz` can be passed directly to EdgeMap.

### 3. gsMap resource directory

EdgeMap requires the pre-computed LD resources from
[gsMap](https://github.com/LeonSong1995/gsMap). The upstream archive is
approximately 621 MiB to download (650,877,553 bytes as checked on
3 August 2026) and is not downloaded automatically:

```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz
tar -xzf gsMap_resource.tar.gz
```

Expected structure after extraction:

```text
gsMap_resource/
├── quick_mode/
│   ├── baseline/
│   │   ├── baseline.{1..22}.l2.ldscore.feather
│   │   └── baseline.{1..22}.l2.M_5_50
│   └── snp_gene_weight_matrix.h5ad
└── LDSC_resource/
    └── weights_hm3_no_hla/
        └── weights.{1..22}.l2.ldscore.gz
```

Resource resolution order:

1. `--resource-dir` (CLI) or `resource_dir=` (Python)
2. `EDGEMAP_RESOURCE_DIR`
3. Auto-detection at `data/gsMap_resource` relative to the installed package or source tree

For reproducibility and clarity, passing `--resource-dir` explicitly is recommended.

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

adata.var["node_score"]
adata.var["edge_score"]
adata.uns["edgemap"]
```

For file-based workflows, pass `st_h5ad` instead:

```python
results = edgemap.run(edgemap.PipelineConfig(
    st_h5ad="my_tissue.h5ad",
    gwas_sumstats="munged_trait.sumstats.gz",
    gwas_label="Systolic blood pressure",
    output_dir="results/sbp_heart",
    resource_dir="/path/to/gsMap_resource",
))
```

## Parameters

| CLI | Python | Default | Description |
|-----|--------|---------|-------------|
| `--st` | `st_h5ad` | *(required)* | Path to the spatial transcriptomics `.h5ad` file |
| `--gwas` | `gwas_sumstats` | *(required)* | Path to munged GWAS summary statistics |
| `--gwas-label` | `gwas_label` | *(required)* | Human-readable trait label |
| `--output` | `output_dir` | `results` | Output directory |
| `--resource-dir` | `resource_dir` | auto-detect | gsMap resource directory |
| `--k-spatial` | `spatial.k_spatial` | 6 | Number of spatial neighbors |
| `--dis-thr` | `spatial.dis_thr` | 3000 | Distance threshold in the same units as `.obsm["spatial"]` |
| `--n-blocks` | `regression.n_blocks` | 200 | Jackknife blocks for standard errors |
| `--gene-chunk-size` | `score.gene_chunk_size` | auto | Genes per node-score chunk; useful for memory control on large datasets |
| `--preprocessed` | `spatial.preprocessed` | off | Skip normalization/log1p when the input is already preprocessed |
| — | `spatial.min_cells_per_gene` | 10 | Minimum number of cells required for a gene to be retained before scoring |

## Output

All files are written to `--output` (`output_dir` in Python).

### `results.json`

Primary summary output. The schema is concise but not minimal; the fields below are the main ones you will usually inspect.

| Field | Meaning |
|-------|---------|
| `gwas_label` | Trait label used for the run |
| `st_data` | Input ST source (`.h5ad` path or `AnnData (in-memory)`) |
| `params.k_spatial`, `params.dis_thr` | Spatial graph settings |
| `params.gene_chunk_size_requested`, `params.gene_chunk_size_resolved` | Requested and effective node-score chunk size |
| `n_genes` | Number of genes retained after preprocessing |
| `n_lr_pairs_active` | Number of active LR pairs in this dataset |
| `node_edge_spearman` | Spearman correlation between node and edge scores |
| `annotation_diagnostics` | Gene/SNP mapping diagnostics for the annotation-building step |
| `regression.ell_node` | Node heritability enrichment: `tau`, `se`, `z`, `p_twosided`, `p_onesided` |
| `regression.ell_edge` | Edge heritability enrichment: `tau`, `se`, `z`, `p_twosided`, `p_onesided` |
| `regression.intercept` | S-LDSC intercept |
| `regression.n_snps`, `regression.N_bar`, `regression.M_total` | Regression metadata |
| `edge_significant` | `true` if aggregate edge `p_onesided < 0.05` |
| `n_pairs_tested` | Number of LR pairs ranked in conditional S-LDSC (present only when generated) |
| `total_time_s` | End-to-end runtime |

Interpretation: a significant edge tau means trait-associated variants are enriched near genes whose spatial communication patterns are concentrated, beyond what cell-intrinsic expression specificity explains.

### `per_pair_sldsc.csv`

Generated only when the aggregate edge signal is significant. Each row is one LR pair tested conditionally against baseline + node.

| Column | Meaning |
|--------|---------|
| `pair` | LR pair label (for example `VEGFA-FLT1`) |
| `tau` | Pair-specific heritability coefficient |
| `se` | Block-jackknife standard error |
| `z` | Ranking score (`tau / se`) |

Use `z` for **ranking**, not for calibrated significance testing. Per-pair annotations are extremely sparse, so the normal approximation for `z` is not reliable here; formal per-pair significance requires empirical calibration.

### `lr_pair_stats.json`

Communication diagnostics for all active LR pairs.

| Field | Meaning |
|-------|---------|
| `mean_comm` | Mean communication intensity across cells |
| `n_active_cells` | Number of cells with nonzero communication |
| `pair_score` | Spatial specificity score for that LR pair |

## Repository scope

This public repository is intentionally the **Python package surface** of EdgeMap. Large resources, local analyses, manuscript assets, and figure-generation workflows are not part of the tracked public package tree.

The publication reproducibility contract and artifact-level release audit are
documented in [`reproducibility/`](reproducibility/README.md). The quickstart is
an aggregate-workflow installation and output-schema smoke test. It requires
the separately installed gsMap resource archive (approximately 621 MiB to
download as checked on 3 August 2026), does not trigger the conditional
per-pair branch with its fixed synthetic input, and does not run the
50,000-replicate empirical calibration used for manuscript per-pair inference.
A manuscript reproducibility archive must not be described as complete until
every row of the publication manifest has passed its release gate.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `h5ad must contain .obsm['spatial']` | Ensure spatial coordinates are present in the AnnData object |
| `Expression values look pre-processed` | Provide raw counts, or set `--preprocessed` |
| `gsMap resource directory not found` | Set `EDGEMAP_RESOURCE_DIR` or pass `--resource-dir` |
| No `per_pair_sldsc.csv` in output | Expected when the aggregate edge signal is not significant |

## Citation

If you use EdgeMap, please cite:

> Yang C, Zhang X, Chen J. *Intercellular communication is a heritable dimension of human tissue architecture.* bioRxiv. 2026. doi: 10.64898/2026.03.29.715138.

## License

MIT
