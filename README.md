# EdgeMap

**Edge-centric heritability mapping via spatial cell–cell communication**

EdgeMap integrates spatial transcriptomics with GWAS summary statistics to partition trait heritability into cell-intrinsic (node) and intercellular (edge) components, then localizes edge signal to specific ligand–receptor channels.

> Yang C, Zhang X, Chen J. *Intercellular communication is a heritable dimension of human tissue architecture.* (2026)

## Overview

Existing methods (S-LDSC, scDRS, gsMap) map genetic risk to cells and tissues but treat each cell as an independent unit. EdgeMap tests a complementary hypothesis: **do genetic effects concentrate at molecular interfaces between cells?**

The pipeline:

1. **Spatial communication** — Builds a Gaussian-weighted spatial graph from ST coordinates and computes ligand–receptor (LR) communication intensity per cell per pair (mass-action kinetics with spatial diffusion).
2. **Node & edge scores** — Quantifies per-gene expression specificity (node) and communication specificity (edge) as local-vs-global enrichment ratios.
3. **SNP annotation** — Maps gene scores to SNP-level LD scores via gsMap's pre-computed SNP–gene weight matrix.
4. **S-LDSC regression** — Joint regression with baseline annotations, yielding τ coefficients and p-values for node and edge heritability.
5. **Per-pair resolution** — Conditionally tests individual LR pairs to identify the channels driving edge signal.

## Installation

```bash
pip install -e .
```

### Requirements

- Python ≥ 3.10
- [gsMap resource files](https://github.com/LeonSong1995/gsMap) (baseline LD scores, regression weights, SNP–gene weight matrix)

EdgeMap bundles the [LIANA Consensus](https://github.com/saezlab/liana) LR database (4,624 curated pairs) — no separate download needed.

## Quick start

### Command line

```bash
edgemap \
    --st visium_heart.h5ad \
    --gwas munged_sbp.tsv \
    --gwas-label "Systolic blood pressure" \
    --output results/sbp_heart \
    --resource-dir /path/to/gsMap_resource
```

### Python API

```python
import edgemap

results = edgemap.run(edgemap.PipelineConfig(
    st_h5ad="visium_heart.h5ad",
    gwas_sumstats="munged_sbp.tsv",
    gwas_label="Systolic blood pressure",
    output_dir="results/sbp_heart",
    resource_dir="/path/to/gsMap_resource",
))
```

### gsMap resource directory

EdgeMap needs gsMap's pre-computed LD resources. Specify the path via any of:

1. `--resource-dir` (CLI) or `resource_dir=` (Python) — highest priority
2. `EDGEMAP_RESOURCE_DIR` environment variable
3. Auto-detection of known filesystem locations

## Input

| Input | Format | Description |
|-------|--------|-------------|
| ST data | `.h5ad` | AnnData with **raw counts** and `.obsm["spatial"]` coordinates (use `--preprocessed` if already log1p-normalized) |
| GWAS sumstats | `.tsv` | Tab-separated with columns `SNP`, `Z`, `N` (use [ldsc munge_sumstats](https://github.com/bulik/ldsc)) |
| gsMap resources | directory | Baseline LD scores, regression weights, SNP–gene weight matrix |

## Output

EdgeMap writes to `--output`:

| File | Content |
|------|---------|
| `results.json` | τ coefficients, z-scores, p-values for node and edge annotations |
| `lr_pair_stats.json` | Per-LR-pair communication statistics |
| `per_pair_sldsc.csv` | Per-pair conditional S-LDSC results (if edge is significant) |

Key fields in `results.json`:

```
regression.ell_edge.z       Edge z-score
regression.ell_edge.p_onesided  Edge one-sided p-value
regression.ell_node.z       Node z-score
edge_significant            Whether edge τ passes p < 0.05
```

## Package structure

```
src/edgemap/
├── config.py       Resource resolution + parameter dataclasses
├── spatial.py      ST loading, spatial graph, LR communication
├── scores.py       Node (expression specificity) and edge (communication specificity) scores
├── annotation.py   Gene scores → SNP annotation LD scores
├── regression.py   Stratified LD score regression + block jackknife
├── simulation.py   Type-I error calibration and power analysis
├── pipeline.py     End-to-end orchestration
├── cli.py          Command-line interface
└── data/           Bundled LIANA Consensus LR database
```

## Citation

If you use EdgeMap in your research, please cite:

```
Yang C, Zhang X, Chen J. Intercellular communication is a heritable
dimension of human tissue architecture. (2026)
```

## License

MIT
