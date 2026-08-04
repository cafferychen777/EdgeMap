# EdgeMap

**Heritability mapping with spatially informed ligand–receptor gene annotations**

EdgeMap integrates spatial transcriptomics with GWAS summary statistics. Its
primary analysis tests whether trait heritability is conditionally associated
with a spatially weighted ligand–receptor (LR) gene annotation after accounting
for baseline genomic annotations and cell-intrinsic expression specificity.

The aggregate statistic is an annotation-level association. It does not by
itself establish that cell–cell communication is causal. A secondary analysis
uses curated, spatially active LR contexts to prioritize their small
constituent-gene sets; that analysis does not identify a directed LR relation or
a molecular interaction.

## How it works

1. **Spatial LR activity proxy** — Build a Gaussian-weighted spatial neighbor graph (`k=6`) and compute a mass-action-inspired expression proxy for each curated LR label, using a bottleneck rule for multi-subunit complexes. This proxy does not measure binding or signaling flux.
2. **Node and aggregate LR-gene scores** — Quantify where expression is spatially concentrated (**node**) and assign genes a spatially informed aggregate LR score (**edge**, retained as the public field name for compatibility).
3. **SNP annotation** — Map gene-level scores to SNP-level LD scores using gsMap's pre-computed SNP–gene weight matrix.
4. **S-LDSC regression** — Regress GWAS chi-squared statistics on baseline + node + aggregate LR-gene annotations to estimate their conditional associations with heritability.
5. **LR-context gene-set ranking** — By default, a positive aggregate LR-gene screen triggers conditional S-LDSC ranking of active LR-context constituent-gene annotations. Use `--rank-contexts` only for a prespecified exploratory setting that should be ranked regardless of the aggregate screen.

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
| `--rank-contexts` | `run_context_ranking` | off | Force exploratory LR-context constituent-gene ranking even when the aggregate screen is not positive |
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
| `regression.ell_edge` | Conditional aggregate LR-gene annotation result: `tau`, `se`, `z`, `p_twosided`, `p_onesided` |
| `regression.intercept` | S-LDSC intercept |
| `regression.n_snps`, `regression.N_bar`, `regression.M_total` | Regression metadata |
| `edge_significant` | `true` if the aggregate LR-gene annotation has `p_onesided < 0.05` (legacy field name retained for compatibility) |
| `n_pairs_tested` | Number of LR contexts whose constituent-gene annotations were ranked (present only when generated) |
| `total_time_s` | End-to-end runtime |

Interpretation: a significantly positive `ell_edge` tau is evidence of a
conditional association between trait heritability and the spatially weighted
LR-gene annotation, beyond the included baseline and node controls. This result
is not, on its own, evidence for a causal communication mechanism.

### `per_pair_sldsc.csv`

Generated when the aggregate LR-gene screen is positive, or when the user
explicitly passes `--rank-contexts` for a prespecified exploratory analysis.
Each row is indexed by an active LR context and tests the annotation formed by
the union of its ligand and receptor genes, conditionally on baseline + node.
The override does not provide multiplicity control or the publication's
separate empirical-null calibration.

| Column | Meaning |
|--------|---------|
| `pair` | LR context label (for example `VEGFA-FLT1`) |
| `tau` | Scale-dependent coefficient for the context's constituent-gene annotation |
| `se` | Block-jackknife standard error |
| `z` | Ranking score (`tau / se`) |

For compatibility with existing results, every participating gene is assigned
the context's positive spatial specificity score. This is a scale convention:
the resulting LD-score vector is that positive scalar times the binary
constituent-gene membership vector. Changing the scalar rescales `tau` and `se`
inversely but leaves `z` unchanged; reversing ligand and receptor labels also
leaves the annotation unchanged. Consequently, `z` prioritizes a directionless,
LR-context-indexed constituent-gene set. It does not identify the LR relation,
direction, interaction, or communication intensity.

Use `z` for **ranking**, not for calibrated significance testing. These
annotations are extremely sparse, so the normal approximation for `z` is not
reliable here; formal inference requires empirical calibration. Calibration
addresses the null distribution but does not change the estimand described
above.

### `lr_pair_stats.json`

Spatial LR activity diagnostics for all active LR contexts.

| Field | Meaning |
|-------|---------|
| `mean_comm` | Mean LR activity proxy across cells |
| `n_active_cells` | Number of cells with nonzero LR activity proxy |
| `pair_score` | Spatial specificity score for that LR context |

## Repository scope

This public repository is intentionally the **Python package surface** of EdgeMap. Large resources, local analyses, manuscript assets, and figure-generation workflows are not part of the tracked public package tree.

The publication reproducibility contract and artifact-level release audit are
documented in [`reproducibility/`](reproducibility/README.md). The quickstart is
an aggregate-workflow installation and output-schema smoke test. It requires
the separately installed gsMap resource archive (approximately 621 MiB to
download as checked on 3 August 2026), does not trigger the conditional
LR-context gene-set branch with its fixed synthetic input, and does not run the
50,000-replicate empirical calibration used for manuscript LR-context
inference.
A manuscript reproducibility archive must not be described as complete until
every row of the publication manifest has passed its release gate.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `h5ad must contain .obsm['spatial']` | Ensure spatial coordinates are present in the AnnData object |
| `Expression values look pre-processed` | Provide raw counts, or set `--preprocessed` |
| `gsMap resource directory not found` | Set `EDGEMAP_RESOURCE_DIR` or pass `--resource-dir` |
| No `per_pair_sldsc.csv` in output | Expected when the aggregate LR-gene screen is not positive |

## Citation

If you use EdgeMap, please cite:

> Yang C, Zhang X, Chen J. *Intercellular communication is a heritable dimension of human tissue architecture.* bioRxiv. 2026. doi: 10.64898/2026.03.29.715138.

## License

MIT
