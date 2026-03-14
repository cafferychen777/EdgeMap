"""
End-to-end pipeline orchestration.

Chains all steps with timing and diagnostics.
Single spatial graph construction feeds both node scores and communication,
with consistent d_max filtering applied to both.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from scipy.stats import spearmanr

from .config import PipelineConfig, resolve_resource_dir
from .spatial import load_st, build_spatial_graph, load_lr_pairs, compute_communication
from .scores import compute_node_scores, compute_edge_scores
from .annotation import build_annotation_ldscores, build_per_pair_ldscores
from .regression import (
    load_baseline, load_regression_weights, load_sumstats,
    run_sldsc, run_per_pair_ldsc,
)


def run(cfg: PipelineConfig) -> dict:
    """Execute the full EdgeMap pipeline."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rdir = resolve_resource_dir(cfg.resource_dir)

    print("=" * 60)
    print(f"EdgeMap: {cfg.gwas_label} x {Path(cfg.st_h5ad).stem}")
    print("=" * 60)

    # -- Step 1: Load ST, build spatial graph -----------------------
    print(f"\n[1/7] Loading ST data and building spatial graph...")
    t0 = time.time()
    adata = load_st(cfg.st_h5ad, cfg.spatial)
    coords = adata.obsm["spatial"].astype(np.float32)

    W, knn_idx, knn_valid = build_spatial_graph(
        coords,
        cfg.spatial.k_spatial,
        cfg.spatial.dis_thr,
        kernel_bandwidth_frac=cfg.score.kernel_bandwidth_frac,
    )

    n_filtered = int((~knn_valid[:, 1:]).sum())
    print(f"  {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    print(f"  Spatial graph: {W.nnz:,} edges"
          + (f" ({n_filtered} neighbor slots beyond d_max)" if n_filtered else ""))
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 2: Spatial communication ---------------------------------
    print(f"\n[2/7] Computing spatial communication for LR pairs...")
    t0 = time.time()

    pairs = load_lr_pairs(adata)
    genes = adata.var_names.tolist()

    comm, pair_names, pair_genes = compute_communication(
        adata.X, W, pairs, genes,
    )

    print(f"  {len(pairs)} active LR pairs, {comm.shape[0]:,} cells")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 3: Edge scores (communication specificity) ------------
    print(f"\n[3/7] Computing edge scores (communication specificity)...")
    t0 = time.time()

    edge_arr, lr_stats = compute_edge_scores(
        comm, pair_names, pair_genes, genes, cfg.score,
    )
    del comm  # free communication matrix

    edge_scores = pd.Series(edge_arr, index=genes)
    print(f"  {(edge_arr > 0).sum()} genes with edge signal, {len(lr_stats)} active LR pairs")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 4: Node scores (expression specificity) ---------------
    print(f"\n[4/7] Computing node scores (expression specificity)...")
    t0 = time.time()

    # Pass sparse X directly — compute_node_scores densifies per chunk
    node_arr = compute_node_scores(adata.X, knn_idx, knn_valid, cfg.score)
    del adata  # free AnnData

    node_scores = pd.Series(node_arr, index=genes)
    print(f"  {(node_arr > 0).sum()} genes with node signal")
    print(f"  {time.time() - t0:.1f}s")

    # Diagnostic: node-edge correlation
    both = (node_arr > 0) & (edge_arr > 0)
    rho = spearmanr(node_arr[both], edge_arr[both]).statistic if both.sum() > 2 else 0.0
    print(f"  Node-Edge Spearman rho: {rho:.3f}")

    # -- Step 5: Annotation LD scores --------------------------------
    print(f"\n[5/7] Building annotation LD scores...")
    t0 = time.time()
    annot_ld, annot_diag = build_annotation_ldscores(node_scores, edge_scores, rdir)
    print(f"  {annot_diag['n_node_genes']} node genes, {annot_diag['n_edge_genes']} edge genes mapped")
    print(f"  Gene-level corr: {annot_diag['gene_corr']:.3f}, "
          f"SNP-level corr: {annot_diag['snp_corr']:.3f}")
    print(f"  ell_node mean={annot_ld['ell_node'].mean():.3f}, "
          f"ell_edge mean={annot_ld['ell_edge'].mean():.3f}")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 6: Load GWAS + baseline LD -----------------------------
    print(f"\n[6/7] Loading GWAS and baseline LD scores...")
    t0 = time.time()
    sumstats = load_sumstats(cfg.gwas_sumstats, cfg.regression)
    baseline, M_total = load_baseline(rdir)
    w_ld = load_regression_weights(rdir)
    print(f"  GWAS: {len(sumstats):,} SNPs, N={sumstats['N'].iloc[0]:.0f}")
    print(f"  Baseline: {len(baseline):,} SNPs, M_5_50={M_total:.0f}")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 7: S-LDSC regression -----------------------------------
    print(f"\n[7/7] Running S-LDSC regression...")
    t0 = time.time()
    results = run_sldsc(sumstats, baseline, annot_ld, w_ld, M_total, cfg.regression)

    print(f"\n  {'Annotation':<16} {'tau':>12} {'se':>12} {'z':>8} {'p(1-sided)':>12}")
    print(f"  {'-' * 62}")
    for name in ["ell_node", "ell_edge"]:
        r = results[name]
        sig = "***" if r["p_onesided"] < 0.001 else "**" if r["p_onesided"] < 0.01 \
              else "*" if r["p_onesided"] < 0.05 else ""
        print(f"  {name:<16} {r['tau']:>12.4e} {r['se']:>12.4e} "
              f"{r['z']:>8.3f} {r['p_onesided']:>12.4e} {sig}")
    print(f"  {'intercept':<16} {results['intercept']:>12.4f}")
    print(f"  {time.time() - t0:.1f}s")

    # -- Per-pair conditional testing (if aggregate edge is significant) --
    edge_p = results["ell_edge"]["p_onesided"]
    edge_sig = edge_p < 0.05
    per_pair_results = None

    if edge_sig and lr_stats:
        print(f"\n[7b/7] Edge significant — running per-LR-pair conditional S-LDSC...")
        t0 = time.time()

        # Build per-pair LD scores using each pair's specificity score
        pair_scores = {
            pname: stats["pair_score"]
            for pname, stats in lr_stats.items()
            if stats.get("pair_score", 0) > 1.0
        }

        pair_ld, snp_names_pair = build_per_pair_ldscores(
            pair_names, pair_genes, pair_scores, rdir,
        )

        if pair_ld:
            annot_ld_node = annot_ld[["SNP", "ell_node"]]
            per_pair_results = run_per_pair_ldsc(
                sumstats, baseline, annot_ld_node, pair_ld, snp_names_pair,
                w_ld, M_total, cfg.regression,
            )
            n_sig = (per_pair_results["p_bonferroni"] < 0.05).sum() if len(per_pair_results) > 0 else 0
            print(f"  Tested {len(pair_ld)} pairs, {n_sig} significant after Bonferroni")
            print(f"  {time.time() - t0:.1f}s")

    # -- Verdict -----------------------------------------------------
    total_time = time.time() - t_start

    print(f"\n{'=' * 60}")
    if edge_sig:
        print(f"RESULT: Edge tau SIGNIFICANT (p={edge_p:.4e})")
        print(f"  Cell-cell communication carries trait heritability")
        print(f"  beyond what expression specificity explains.")
    else:
        print(f"RESULT: Edge tau not significant (p={edge_p:.4e})")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    # -- Save --------------------------------------------------------
    output = {
        "gwas_label": cfg.gwas_label,
        "st_data": str(cfg.st_h5ad),
        "params": {
            "k_spatial": cfg.spatial.k_spatial,
            "dis_thr": cfg.spatial.dis_thr,
            "edge_agg_percentile": cfg.score.edge_agg_percentile,
            "kernel_bandwidth_frac": cfg.score.kernel_bandwidth_frac,
        },
        "n_genes": len(genes),
        "n_lr_pairs_active": len(lr_stats),
        "node_edge_spearman": float(rho),
        "annotation_diagnostics": annot_diag,
        "regression": {
            k: v for k, v in results.items()
            if k in ("ell_node", "ell_edge", "intercept", "n_snps", "N_bar", "M_total")
        },
        "edge_significant": bool(edge_sig),
        "total_time_s": round(total_time, 1),
    }

    if per_pair_results is not None and len(per_pair_results) > 0:
        per_pair_results.to_csv(out / "per_pair_sldsc.csv", index=False)
        output["n_pairs_tested"] = len(per_pair_results)
        output["n_pairs_significant"] = int((per_pair_results["p_bonferroni"] < 0.05).sum())

    # Write results.json AFTER all fields are populated
    with open(out / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(out / "lr_pair_stats.json", "w") as f:
        json.dump(lr_stats, f, indent=2)

    print(f"Saved to {out}/")
    return output
