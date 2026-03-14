"""Command-line interface for EdgeMap."""

import argparse

from .config import PipelineConfig, SpatialConfig, RegressionConfig
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(
        prog="edgemap",
        description="EdgeMap: edge-centric heritability mapping "
                    "via spatial cell-cell communication",
    )
    parser.add_argument("--st", required=True,
                        help="Path to spatial transcriptomics h5ad file")
    parser.add_argument("--gwas", required=True,
                        help="Path to munged GWAS summary statistics (SNP/Z/N tsv)")
    parser.add_argument("--gwas-label", required=True,
                        help="Human-readable trait name")
    parser.add_argument("--output", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--resource-dir", default=None,
                        help="gsMap resource directory "
                             "(default: auto-detect or EDGEMAP_RESOURCE_DIR)")
    parser.add_argument("--k-spatial", type=int, default=6,
                        help="Number of spatial neighbors (default: 6)")
    parser.add_argument("--dis-thr", type=float, default=3000.0,
                        help="Distance threshold for spatial graph (default: 3000)")
    parser.add_argument("--n-blocks", type=int, default=200,
                        help="Number of jackknife blocks (default: 200)")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Skip normalization (data already log1p-normalized)")

    args = parser.parse_args()

    cfg = PipelineConfig(
        st_h5ad=args.st,
        gwas_sumstats=args.gwas,
        gwas_label=args.gwas_label,
        output_dir=args.output,
        resource_dir=args.resource_dir,
        spatial=SpatialConfig(
            k_spatial=args.k_spatial,
            dis_thr=args.dis_thr,
            preprocessed=args.preprocessed,
        ),
        regression=RegressionConfig(n_blocks=args.n_blocks),
    )

    run(cfg)
