"""
End-to-end pipeline orchestration.

Chains all steps with timing and diagnostics.
Single spatial graph construction feeds both node scores and the LR activity proxy,
with consistent d_max filtering applied to both.
"""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ._matrix import ensure_csc_matrix
from .config import PipelineConfig, resolve_gene_chunk_size, resolve_resource_dir
from .spatial import load_st, preprocess_st, build_spatial_graph, load_lr_pairs, compute_communication
from .scores import compute_node_scores, compute_edge_scores
from .annotation import build_annotation_ldscores, build_per_pair_ldscores
from .regression import (
    load_baseline, load_regression_weights, load_sumstats,
    run_sldsc, run_per_pair_ldsc,
)


try:
    _PACKAGE_VERSION = version("edgemap")
except PackageNotFoundError:
    _PACKAGE_VERSION = "unknown"


@contextmanager
def _atomic_output_path(target: Path):
    """Yield a temporary sibling path and atomically replace target on success."""
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        yield tmp_path
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_json_atomic(value: object, target: Path) -> None:
    """Serialize JSON without exposing a partially written destination."""
    with _atomic_output_path(target) as tmp_path:
        with open(tmp_path, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)


def run(cfg: PipelineConfig, adata: ad.AnnData | None = None) -> dict:
    """Execute the full EdgeMap pipeline.

    Args:
        cfg: Pipeline configuration.
        adata: Optional AnnData object. If provided, used instead of
               loading from cfg.st_h5ad. A copy is made for preprocessing;
               the original is not modified during computation. After
               completion, results are written to adata.var and adata.uns.
    """
    if adata is None and not cfg.st_h5ad:
        raise ValueError("Provide either st_h5ad in config or pass adata directly.")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rdir = resolve_resource_dir(cfg.resource_dir)

    st_label = Path(cfg.st_h5ad).stem if cfg.st_h5ad else "AnnData"
    print("=" * 60)
    print(f"EdgeMap: {cfg.gwas_label} x {st_label}")
    print("=" * 60)

    # -- Step 1: Load ST, build spatial graph -----------------------
    print("\n[1/7] Loading ST data and building spatial graph...")
    t0 = time.time()
    adata_input = adata  # keep reference for result write-back
    if adata is not None:
        adata_work = preprocess_st(adata.copy(), cfg.spatial)
    else:
        adata_work = load_st(cfg.st_h5ad, cfg.spatial)
    coords = adata_work.obsm["spatial"].astype(np.float32)
    X_work = ensure_csc_matrix(adata_work.X)
    score_cfg = replace(
        cfg.score,
        gene_chunk_size=resolve_gene_chunk_size(adata_work.shape[0], cfg.score.gene_chunk_size),
    )

    W, knn_idx, knn_valid = build_spatial_graph(
        coords,
        cfg.spatial.k_spatial,
        cfg.spatial.dis_thr,
        kernel_bandwidth_frac=score_cfg.kernel_bandwidth_frac,
    )

    n_filtered = int((~knn_valid[:, 1:]).sum())
    print(f"  {adata_work.shape[0]:,} cells x {adata_work.shape[1]:,} genes")
    print(f"  Spatial graph: {W.nnz:,} edges"
          + (f" ({n_filtered} neighbor slots beyond d_max)" if n_filtered else ""))
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 2: Spatial LR activity proxy ------------------------------
    print("\n[2/7] Computing spatial LR activity scores...")
    t0 = time.time()

    pairs = load_lr_pairs(
        adata_work,
        min_cell_pct=cfg.spatial.min_lr_cell_pct,
    )
    genes = adata_work.var_names.tolist()

    comm, pair_names, pair_genes = compute_communication(
        X_work, W, pairs, genes,
    )
    del W, coords

    print(f"  {len(pairs)} active LR contexts, {comm.shape[0]:,} cells")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 3: Aggregate spatial LR-gene scores -----------------------
    print("\n[3/7] Computing aggregate spatial LR-gene scores...")
    t0 = time.time()

    edge_arr, lr_stats = compute_edge_scores(
        comm, pair_names, pair_genes, genes, score_cfg,
    )
    del comm  # free LR activity matrix

    edge_scores = pd.Series(edge_arr, index=genes)
    print(f"  {(edge_arr > 0).sum()} genes with nonzero LR-gene score, "
          f"{len(lr_stats)} contexts with nonzero mean activity")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 4: Node scores (expression specificity) ---------------
    print("\n[4/7] Computing node scores (expression specificity)...")
    t0 = time.time()
    print(f"  Node score chunk size: {score_cfg.gene_chunk_size}")

    # Pass sparse X directly — compute_node_scores densifies per chunk
    node_arr = compute_node_scores(X_work, knn_idx, knn_valid, score_cfg)
    del X_work, knn_idx, knn_valid, adata_work

    node_scores = pd.Series(node_arr, index=genes)
    print(f"  {(node_arr > 0).sum()} genes with node signal")
    print(f"  {time.time() - t0:.1f}s")

    # Diagnostic: node-edge correlation
    both = (node_arr > 0) & (edge_arr > 0)
    rho = 0.0
    if both.sum() > 2:
        node_both = node_arr[both]
        edge_both = edge_arr[both]
        if not (
            np.all(node_both == node_both[0])
            or np.all(edge_both == edge_both[0])
        ):
            candidate_rho = float(spearmanr(node_both, edge_both).statistic)
            if np.isfinite(candidate_rho):
                rho = candidate_rho
    print(f"  Node-Edge Spearman rho: {rho:.3f}")

    # -- Step 5: Annotation LD scores --------------------------------
    print("\n[5/7] Building annotation LD scores...")
    t0 = time.time()
    annot_ld, annot_diag = build_annotation_ldscores(node_scores, edge_scores, rdir)
    del node_arr, edge_arr
    if adata_input is None:
        del node_scores, edge_scores
    print(f"  {annot_diag['n_node_genes']} node genes, {annot_diag['n_edge_genes']} edge genes mapped")
    print(f"  Gene-level corr: {annot_diag['gene_corr']:.3f}, "
          f"SNP-level corr: {annot_diag['snp_corr']:.3f}")
    print(f"  ell_node mean={annot_ld['ell_node'].mean():.3f}, "
          f"ell_edge mean={annot_ld['ell_edge'].mean():.3f}")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 6: Load GWAS + baseline LD -----------------------------
    print("\n[6/7] Loading GWAS and baseline LD scores...")
    t0 = time.time()
    sumstats = load_sumstats(cfg.gwas_sumstats, cfg.regression)
    baseline, M_total = load_baseline(rdir, copy=False)
    w_ld = load_regression_weights(rdir, copy=False)
    print(f"  GWAS: {len(sumstats):,} SNPs, N={sumstats['N'].iloc[0]:.0f}")
    print(f"  Baseline: {len(baseline):,} SNPs, M_5_50={M_total:.0f}")
    print(f"  {time.time() - t0:.1f}s")

    # -- Step 7: S-LDSC regression -----------------------------------
    print("\n[7/7] Running S-LDSC regression...")
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

    # -- Exploratory LR-context constituent-gene ranking ----------------
    edge_p = results["ell_edge"]["p_onesided"]
    edge_sig = edge_p < 0.05
    per_pair_results = None
    ranking_reason = None

    if (edge_sig or cfg.run_context_ranking) and lr_stats:
        ranking_reason = "aggregate_screen" if edge_sig else "explicit_request"
        if edge_sig:
            print("\n[7b/7] Aggregate LR-gene screen positive — "
                  "ranking LR-context constituent-gene annotations...")
        else:
            print("\n[7b/7] Explicit exploratory request — ranking LR-context "
                  "constituent-gene annotations despite a non-positive "
                  "aggregate screen...")
        t0 = time.time()

        # Positive PairScore gates contexts and retains the historical scale.
        # Within a context it is a scalar convention: z is score-invariant.
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
            n_tested = len(per_pair_results) if per_pair_results is not None else 0
            n_skipped = len(
                per_pair_results.attrs.get("skipped_unidentifiable", {})
            )
            print(f"  Ranked {n_tested} LR-context gene sets by z-score")
            if n_skipped:
                print(f"  Skipped {n_skipped} contexts without an identifiable "
                      "delete-block coefficient")
            if n_tested > 0:
                top = per_pair_results.iloc[0]
                print(f"  Top LR context: {top['pair']} (z={top['z']:.2f})")
            print(f"  {time.time() - t0:.1f}s")
        else:
            per_pair_results = pd.DataFrame(columns=["pair", "tau", "se", "z"])

    # -- Verdict -----------------------------------------------------
    total_time = time.time() - t_start

    print(f"\n{'=' * 60}")
    if edge_sig:
        print(f"RESULT: Aggregate LR-gene annotation tau SIGNIFICANT (p={edge_p:.4e})")
        print("  Positive conditional association with trait heritability after")
        print("  baseline and expression-specificity controls; not a causal claim.")
    else:
        print(f"RESULT: Aggregate LR-gene annotation tau not significant (p={edge_p:.4e})")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    # -- Save --------------------------------------------------------
    output = {
        "edgemap_version": _PACKAGE_VERSION,
        "gwas_label": cfg.gwas_label,
        "st_data": str(cfg.st_h5ad) if cfg.st_h5ad else "AnnData (in-memory)",
        "params": {
            "k_spatial": cfg.spatial.k_spatial,
            "dis_thr": cfg.spatial.dis_thr,
            "min_cells_per_gene": cfg.spatial.min_cells_per_gene,
            "min_lr_cell_pct": cfg.spatial.min_lr_cell_pct,
            "input_scale": cfg.spatial.resolved_input_scale,
            "edge_agg_percentile": score_cfg.edge_agg_percentile,
            "edge_agg_method": score_cfg.edge_agg_method,
            "node_agg_percentile": score_cfg.node_agg_percentile,
            "kernel_bandwidth_frac": score_cfg.kernel_bandwidth_frac,
            "gene_chunk_size_requested": cfg.score.gene_chunk_size,
            "gene_chunk_size_resolved": score_cfg.gene_chunk_size,
            "n_blocks": cfg.regression.n_blocks,
            "chisq_max_factor": cfg.regression.chisq_max_factor,
            "chisq_max_floor": cfg.regression.chisq_max_floor,
            "run_context_ranking": cfg.run_context_ranking,
        },
        "n_genes": len(genes),
        "n_lr_pairs_active": len(pairs),
        "n_lr_pairs_scored": len(lr_stats),
        "node_edge_spearman": float(rho),
        "annotation_diagnostics": annot_diag,
        "regression": {
            k: v for k, v in results.items()
            if k in ("ell_node", "ell_edge", "intercept", "n_snps", "N_bar", "M_total")
        },
        "edge_significant": bool(edge_sig),
        "total_time_s": round(total_time, 1),
    }

    per_pair_path = out / "per_pair_sldsc.csv"
    if per_pair_results is not None:
        output["n_pairs_tested"] = len(per_pair_results)
        output["context_ranking_reason"] = ranking_reason
        skipped = per_pair_results.attrs.get("skipped_unidentifiable", {})
        output["n_pairs_skipped_unidentifiable"] = len(skipped)
        if len(per_pair_results) > 0:
            with _atomic_output_path(per_pair_path) as tmp_path:
                per_pair_results.to_csv(tmp_path, index=False)
        else:
            per_pair_path.unlink(missing_ok=True)
    else:
        # This is a managed output. Removing it prevents a prior run's context
        # ranking from surviving beside an incompatible new results.json.
        per_pair_path.unlink(missing_ok=True)

    # Write summaries only after all fields and managed branch outputs agree.
    _write_json_atomic(output, out / "results.json")
    _write_json_atomic(lr_stats, out / "lr_pair_stats.json")

    # Write results back to user's AnnData if provided
    if adata_input is not None:
        adata_input.var["node_score"] = node_scores.reindex(adata_input.var_names, fill_value=0.0).values
        adata_input.var["edge_score"] = edge_scores.reindex(adata_input.var_names, fill_value=0.0).values
        adata_input.uns["edgemap"] = output

    print(f"Saved to {out}/")
    return output
