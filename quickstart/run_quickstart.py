#!/usr/bin/env python3
"""
EdgeMap quickstart: core-pipeline smoke test on synthetic data.

Generates a small synthetic Visium-like dataset (~500 spots, ~200 genes)
and synthetic GWAS summary statistics, then runs the aggregate EdgeMap
workflow to demonstrate installation and output schemas. It does not run the
manuscript's separate empirical LR-context calibration, and the conditional
gene-set ranking branch is exercised only if the synthetic aggregate LR-gene
result passes its screening threshold.

Usage:
    python quickstart/run_quickstart.py

Requirements:
    - Python >= 3.10
    - edgemap installed (pip install edgemap)
    - gsMap resource directory available (set EDGEMAP_RESOURCE_DIR or
      pass --resource-dir)

Output:
    quickstart/output/
        results.json          -- pipeline summary with tau, z, and p-values
        lr_pair_stats.json    -- spatial statistics for active LR contexts
        per_pair_sldsc.csv    -- LR-context gene-set ranking (if aggregate screen is positive)
        demo_visium.h5ad      -- the synthetic spatial dataset
        demo_gwas.tsv         -- the synthetic GWAS summary statistics
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# 1. Synthetic Visium dataset
# ---------------------------------------------------------------------------

def make_demo_visium(
    n_spots: int = 500,
    seed: int = 42,
    lr_database_path: str | None = None,
) -> ad.AnnData:
    """Create a synthetic Visium-like AnnData with spatial structure.

    The dataset has three spatial domains arranged in a grid. Ligand and
    receptor genes are expressed in complementary domains so that the LR
    activity proxy is spatially concentrated at domain boundaries.

    Args:
        n_spots: Number of spots (default 500).
        seed: Random seed for reproducibility.
        lr_database_path: Path to LIANA consensus CSV (auto-detected if None).

    Returns:
        AnnData with raw integer counts and .obsm["spatial"].
    """
    rng = np.random.default_rng(seed)

    # -- Select genes: known LR pairs + filler genes -----------------------
    # Load the LR database to pick real gene symbols
    if lr_database_path is None:
        from edgemap.config import get_lr_database
        lr_database_path = str(get_lr_database())
    lr_db = pd.read_csv(lr_database_path)
    lr_db = lr_db[lr_db["resource"] == "consensus"]

    # Pick simple (non-heteromeric) LR pairs
    simple = lr_db[
        (~lr_db["source_genesymbol"].str.contains("_"))
        & (~lr_db["target_genesymbol"].str.contains("_"))
    ].drop_duplicates(subset=["source_genesymbol", "target_genesymbol"])

    # Select ~30 LR pairs for a realistic mix
    selected_pairs = simple.head(60)
    lr_genes = sorted(set(
        selected_pairs["source_genesymbol"].tolist()
        + selected_pairs["target_genesymbol"].tolist()
    ))

    # Add filler genes (housekeeping-like) to reach ~200 total
    filler_names = [f"GENE{i:04d}" for i in range(200 - len(lr_genes))]
    all_genes = lr_genes + filler_names
    n_genes = len(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    # -- Spatial coordinates: grid with three domains ----------------------
    side = int(np.ceil(np.sqrt(n_spots)))
    xs = np.tile(np.arange(side), side)[:n_spots]
    ys = np.repeat(np.arange(side), side)[:n_spots]
    coords = np.column_stack([xs, ys]).astype(np.float32) * 100.0  # Visium-like spacing

    # Assign spots to three spatial domains based on x-coordinate
    domain = np.zeros(n_spots, dtype=int)
    x_thirds = np.percentile(coords[:, 0], [33, 66])
    domain[coords[:, 0] > x_thirds[0]] = 1
    domain[coords[:, 0] > x_thirds[1]] = 2

    # -- Expression matrix: structured counts ------------------------------
    # Base expression: Poisson background for all genes
    counts = rng.poisson(lam=0.5, size=(n_spots, n_genes)).astype(np.float32)

    # Ligand genes: upregulated in domain 0 (left)
    # Receptor genes: upregulated in domain 2 (right)
    # This creates a concentrated LR activity proxy at the domain 0-1 boundary
    ligand_genes = selected_pairs["source_genesymbol"].unique()
    receptor_genes = selected_pairs["target_genesymbol"].unique()

    for g in ligand_genes:
        if g in gene_to_idx:
            gi = gene_to_idx[g]
            # Strong expression in domain 0, moderate in domain 1, low in domain 2
            counts[domain == 0, gi] += rng.poisson(lam=8.0, size=(domain == 0).sum())
            counts[domain == 1, gi] += rng.poisson(lam=3.0, size=(domain == 1).sum())

    for g in receptor_genes:
        if g in gene_to_idx:
            gi = gene_to_idx[g]
            # Strong in domain 1, moderate in domain 2, low in domain 0
            counts[domain == 1, gi] += rng.poisson(lam=8.0, size=(domain == 1).sum())
            counts[domain == 2, gi] += rng.poisson(lam=3.0, size=(domain == 2).sum())

    # Filler genes: uniformly expressed (no spatial pattern)
    for i, g in enumerate(filler_names):
        gi = gene_to_idx[g]
        counts[:, gi] = rng.poisson(lam=2.0, size=n_spots)

    # Convert to integer counts (required by EdgeMap preprocessing)
    counts = np.maximum(counts, 0).astype(np.float32)
    X = sparse.csr_matrix(counts)

    # -- Assemble AnnData --------------------------------------------------
    adata = ad.AnnData(
        X=X,
        var=pd.DataFrame(index=all_genes),
        obs=pd.DataFrame(
            {"domain": pd.Categorical(domain)},
            index=[f"spot_{i:04d}" for i in range(n_spots)],
        ),
    )
    adata.obsm["spatial"] = coords

    return adata


# ---------------------------------------------------------------------------
# 2. Synthetic GWAS summary statistics
# ---------------------------------------------------------------------------

def make_demo_gwas(
    resource_dir: str | Path | None = None,
    seed: int = 123,
) -> pd.DataFrame:
    """Create synthetic GWAS summary statistics matching the baseline SNPs.

    Generates random Z-scores for all HapMap3 SNPs in the baseline LD
    scores. Most Z-scores are drawn from N(0,1) (null), with a small
    fraction inflated to simulate polygenicity. The result is a realistic
    "moderately polygenic trait" that will produce non-trivial S-LDSC
    results.

    Args:
        resource_dir: Path to gsMap resources (auto-detected if None).
        seed: Random seed.

    Returns:
        DataFrame with columns [SNP, Z, N].
    """
    from edgemap.config import resolve_resource_dir
    rdir = resolve_resource_dir(resource_dir)

    # Collect all SNPs from the baseline LD scores
    snps = []
    for chrom in range(1, 23):
        bl = pd.read_feather(rdir / "quick_mode" / "baseline" / f"baseline.{chrom}.l2.ldscore.feather")
        snps.extend(bl["SNP"].tolist())

    rng = np.random.default_rng(seed)
    n = len(snps)

    # Generate Z-scores: mostly null, ~2% inflated (polygenic signal)
    z = rng.standard_normal(n)
    inflated = rng.random(n) < 0.02
    z[inflated] *= 3.0  # inflate a small fraction

    sumstats = pd.DataFrame({
        "SNP": snps,
        "Z": z,
        "N": np.full(n, 50000.0),  # typical GWAS sample size
    })

    return sumstats


# ---------------------------------------------------------------------------
# 3. Run the pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EdgeMap quickstart: core-pipeline smoke test on synthetic data",
    )
    parser.add_argument(
        "--resource-dir", default=None,
        help="gsMap resource directory (default: auto-detect via EDGEMAP_RESOURCE_DIR)",
    )
    parser.add_argument(
        "--output", default=str(Path(__file__).parent / "output"),
        help="Output directory (default: quickstart/output)",
    )
    parser.add_argument(
        "--n-spots", type=int, default=500,
        help="Number of spots in synthetic dataset (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    from edgemap.config import resolve_resource_dir

    # Resolve the external statistical resources before generating any output.
    # The demo cannot perform SNP annotation or S-LDSC without these files.
    resource_dir = resolve_resource_dir(args.resource_dir)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EdgeMap Quickstart")
    print("=" * 60)
    print(f"External gsMap resource: {resource_dir}")

    # -- Step A: Generate synthetic data -----------------------------------
    print("\n[A] Generating synthetic Visium dataset...")
    t0 = time.time()
    adata = make_demo_visium(n_spots=args.n_spots, seed=args.seed)
    st_path = out_dir / "demo_visium.h5ad"
    adata.write_h5ad(st_path)
    print(f"    {adata.shape[0]} spots x {adata.shape[1]} genes")
    print(f"    Saved to {st_path}")
    print(f"    {time.time() - t0:.1f}s")

    print("\n[B] Generating synthetic GWAS summary statistics...")
    t0 = time.time()
    # Keep ST and GWAS streams reproducible but independent. The maintained
    # offset yields a non-positive aggregate screen for the default smoke test.
    sumstats = make_demo_gwas(resource_dir=resource_dir, seed=args.seed + 2)
    gwas_path = out_dir / "demo_gwas.tsv"
    sumstats.to_csv(gwas_path, sep="\t", index=False)
    print(f"    {len(sumstats):,} SNPs, N=50,000")
    print(f"    Saved to {gwas_path}")
    print(f"    {time.time() - t0:.1f}s")

    # -- Step B: Run EdgeMap -----------------------------------------------
    print("\n[C] Running EdgeMap pipeline...")
    from edgemap import PipelineConfig, SpatialConfig, ScoreConfig, RegressionConfig, run

    cfg = PipelineConfig(
        st_h5ad=str(st_path),
        gwas_sumstats=str(gwas_path),
        gwas_label="demo_trait",
        output_dir=str(out_dir),
        resource_dir=str(resource_dir),
        spatial=SpatialConfig(
            k_spatial=6,
            dis_thr=300.0,  # smaller d_max for the 100-unit grid spacing
        ),
        score=ScoreConfig(),
        regression=RegressionConfig(n_blocks=200),
    )

    results = run(cfg, adata=adata)

    # -- Step C: Summarize -------------------------------------------------
    print("\n" + "=" * 60)
    print("QUICKSTART COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in {out_dir}/:")
    print("  demo_visium.h5ad      Synthetic spatial transcriptomics data")
    print("  demo_gwas.tsv         Synthetic GWAS summary statistics")
    print("  results.json          Pipeline results (tau, z, p-values)")
    print("  lr_pair_stats.json    Spatial statistics for active LR contexts")
    per_pair_output = out_dir / "per_pair_sldsc.csv"
    if per_pair_output.exists():
        print("  per_pair_sldsc.csv    LR-context constituent-gene rankings")
        print("\nThe aggregate screen was positive, so the LR-context gene-set")
        print("ranking branch ran. These z-scores are not empirically calibrated.")
    else:
        print("\nThe aggregate screen was not positive, so the LR-context gene-set")
        print("ranking branch did not run for this synthetic example.")
    print("The 50,000-replicate empirical calibration used for manuscript")
    print("LR-context inference is outside the scope of this smoke test.")

    print("\n--- How to interpret results.json ---")
    print()
    print("The key output is the S-LDSC regression, testing the conditional")
    print("association of the aggregate spatial LR-gene annotation (edge)")
    print("after baseline and expression-specificity (node) controls.")
    print()

    reg = results["regression"]
    for name in ["ell_node", "ell_edge"]:
        r = reg[name]
        label = "Node (expression specificity)" if name == "ell_node" \
                else "Edge (aggregate spatial LR-gene annotation)"
        sig = "SIGNIFICANT" if r["p_onesided"] < 0.05 else "not significant"
        print(f"  {label}:")
        print(f"    tau = {r['tau']:.4e}  (effect size per SNP)")
        print(f"    z   = {r['z']:.3f}")
        print(f"    p   = {r['p_onesided']:.4e}  ({sig} at alpha=0.05)")
        print()

    print(f"  Intercept: {reg['intercept']:.4f}  (expect ~1.0; >>1 indicates confounding)")
    print()
    print("NOTE: This is synthetic data, so results are not biologically")
    print("meaningful. The purpose is to exercise the aggregate path on one")
    print("fixed synthetic input and to illustrate the output format.")
    print()
    print("For real analyses, replace demo_visium.h5ad with your spatial")
    print("transcriptomics data and demo_gwas.tsv with munged GWAS")
    print("summary statistics from a trait of interest.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
