"""
Gene-level score computation: node (GSS) and spatial LR-gene (CSS).

Steps 3-4 of the pipeline.

Both scores follow the same design principle:
  1. Compute a per-unit specificity ratio (local / global).
  2. Aggregate to a single gene-level score.

Node: per-spot expression specificity → max over cells.
Edge: per-context LR activity specificity -> max (default) or mean over contexts per gene.

The edge score computes specificity per curated LR context first, then assigns
each gene the score of its strongest context (max, default) or the mean across
all active contexts (mean, for sensitivity analysis). The max is an EdgeMap
design choice that avoids mechanically increasing a score when a shared gene
appears in many labels; it is not a consensus rule inherited from CCC tools.
"""

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from ._matrix import ensure_csc_matrix
from .config import ScoreConfig, resolve_gene_chunk_size


_SPECIFICITY_TOLERANCE = 1e-12


def compute_node_scores(
    X: np.ndarray | sparse.spmatrix,
    knn_idx: np.ndarray,
    knn_valid: np.ndarray,
    cfg: ScoreConfig,
) -> np.ndarray:
    """Gene expression specificity: local vs global rank enrichment.

    For each gene g:
      1. Rank expression across all cells.
      2. For each cell i, compute geometric mean of ranks in valid neighborhood.
      3. Specificity(i, g) = local_geomean / global_geomean.
      4. node_score(g) = max over cells i.

    Ranks and averages are computed per gene-chunk so peak memory is
    O(n_cells * chunk_size) regardless of total gene count.

    Accepts sparse or dense X — sparse columns are densified per chunk,
    avoiding a full-matrix densification in the caller.

    Args:
        X: expression matrix (n_cells, n_genes), sparse or dense
        knn_idx: spatial KNN indices (n_cells, k+1) including self at column 0
        knn_valid: boolean mask (n_cells, k+1), True for neighbors within d_max
        cfg: score configuration

    Returns:
        node_scores: (n_genes,) array
    """
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional expression matrix")
    n, g = X.shape
    if n == 0 or g == 0:
        raise ValueError("X must contain at least one cell and one gene")
    knn_idx = np.asarray(knn_idx)
    knn_valid = np.asarray(knn_valid, dtype=bool)
    if (
        knn_idx.ndim != 2
        or knn_idx.shape != knn_valid.shape
        or knn_idx.shape[0] != n
        or knn_idx.shape[1] == 0
    ):
        raise ValueError("knn_idx and knn_valid must be aligned 2D cell-neighbor arrays")
    if not np.issubdtype(knn_idx.dtype, np.integer):
        raise TypeError("knn_idx must contain integer cell indices")
    if np.any(knn_idx < 0) or np.any(knn_idx >= n):
        raise ValueError("knn_idx contains out-of-range cell indices")
    if not np.array_equal(knn_idx[:, 0], np.arange(n)) or not np.all(
        knn_valid[:, 0]
    ):
        raise ValueError("knn_idx column 0 must be each cell itself and remain valid")

    values = X.data if sparse.issparse(X) else np.asarray(X)
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains non-finite expression values")

    k = knn_idx.shape[1]
    chunk = resolve_gene_chunk_size(n, cfg.gene_chunk_size)
    scores = np.empty(g, dtype=np.float32)

    # Per-cell count of valid neighbors for proper averaging
    n_valid = knn_valid.sum(axis=1, keepdims=True).astype(np.float32)  # (n, 1)
    n_valid = np.maximum(n_valid, 1.0)

    # Convert to CSC for efficient column slicing if sparse
    X_work = ensure_csc_matrix(X)

    for start in range(0, g, chunk):
        end = min(start + chunk, g)

        # Densify only this chunk
        if sparse.issparse(X_work):
            x_chunk = X_work[:, start:end].toarray().astype(np.float32)
        else:
            x_chunk = np.asarray(X_work[:, start:end], dtype=np.float32)

        # Rank within chunk (each column independently)
        # Average ranks preserve ties, making the score invariant to arbitrary
        # cell row order. This is essential for sparse ST matrices with many
        # equal zero-expression entries.
        ranks = rankdata(x_chunk, axis=0, method="average").astype(np.float32)
        np.log(ranks, out=ranks)  # in-place log: ranks → log_ranks
        global_mean = ranks.mean(axis=0)  # (chunk,)

        # Local geometric mean via masked neighbor indexing
        local = np.zeros_like(ranks)
        contrib = np.empty_like(ranks)  # reusable buffer
        for ki in range(k):
            np.take(ranks, knn_idx[:, ki], axis=0, out=contrib)
            np.multiply(contrib, knn_valid[:, ki:ki + 1], out=contrib)
            local += contrib
        local /= n_valid

        # Specificity = exp(local - global), aggregate over cells
        spec = np.exp(local - global_mean[np.newaxis, :])
        if cfg.node_agg_percentile >= 100.0:
            scores[start:end] = spec.max(axis=0)
        else:
            scores[start:end] = np.percentile(spec, cfg.node_agg_percentile, axis=0)

    return scores


def compute_edge_scores(
    comm: np.ndarray,
    pair_names: list[str],
    pair_genes: dict[str, tuple[list[str], list[str]]],
    gene_names: list[str],
    cfg: ScoreConfig,
) -> tuple[np.ndarray, dict]:
    """Context specificity per gene from spatially weighted LR activity proxies.

    Algorithm:
      1. For each LR pair, specificity(cell) = comm(cell) / mean(comm).
      2. pair_score = P-th percentile of specificity across cells.
      3. gene_score = agg(pair_score) over all pairs the gene participates in,
         where agg is max (default) or mean (cfg.edge_agg_method).

    Max avoids shared-subunit inflation: a gene appearing in 50 pairs gets
    the score of its strongest pair, not 50x accumulated signal.

    Args:
        comm: (n_cells, n_pairs) LR activity proxy from compute_communication
        pair_names: pair labels aligned with comm columns
        pair_genes: dict mapping label -> (lig_genes, rec_genes)
        gene_names: gene name list from expression matrix
        cfg: score configuration

    Returns:
        edge_scores: (n_genes,) array
        lr_stats: per-LR-pair statistics dict
    """
    comm = np.asarray(comm, dtype=np.float64)
    if comm.ndim != 2:
        raise ValueError("comm must be a two-dimensional cell-by-context matrix")
    if comm.shape[1] != len(pair_names):
        raise ValueError("pair_names must align with the columns of comm")
    if len(pair_names) != len(set(pair_names)):
        raise ValueError("pair_names must be unique")
    if np.any(np.isinf(comm)) or np.any(comm[np.isfinite(comm)] < 0):
        raise ValueError("comm must contain non-negative LR activity or NaN")

    n_genes = len(gene_names)
    if n_genes != len(set(gene_names)):
        raise ValueError("gene_names must be unique")
    name2i = {g: i for i, g in enumerate(gene_names)}

    if comm.shape[1] == 0:
        return np.zeros(n_genes, dtype=np.float64), {}

    scores = np.zeros(n_genes, dtype=np.float64)
    gene_counts = np.zeros(n_genes, dtype=np.int32)  # for mean aggregation
    lr_stats = {}
    use_mean = cfg.edge_agg_method == "mean"

    for pi, pname in enumerate(pair_names):
        if pname not in pair_genes:
            raise ValueError(f"pair_genes is missing context {pname!r}")
        ligs, recs = pair_genes[pname]
        missing = [g for g in ligs + recs if g not in name2i]
        if missing:
            raise ValueError(
                f"Context {pname!r} references genes absent from gene_names: "
                f"{', '.join(missing)}"
            )
        pair_comm = comm[:, pi]
        valid_cells = np.isfinite(pair_comm)
        if not np.any(valid_cells):
            continue
        pair_comm_valid = pair_comm[valid_cells]
        pair_mean = pair_comm_valid.mean()
        if pair_mean <= 0:
            continue

        # Specificity: spatial concentration of this context's activity proxy
        spec = pair_comm_valid / pair_mean
        pair_score = float(np.percentile(spec, cfg.edge_agg_percentile))

        lr_stats[pname] = {
            "mean_comm": float(pair_mean),
            "n_valid_cells": int(valid_cells.sum()),
            "n_active_cells": int((pair_comm_valid > 0).sum()),
            "pair_score": pair_score,
        }

        # Only keep spatially concentrated pairs
        if pair_score <= 1.0 + _SPECIFICITY_TOLERANCE:
            continue

        # Propagate to all participating genes
        for g in ligs + recs:
            gi = name2i[g]
            if use_mean:
                scores[gi] += pair_score
                gene_counts[gi] += 1
            else:
                if pair_score > scores[gi]:
                    scores[gi] = pair_score

    if use_mean:
        mask = gene_counts > 0
        scores[mask] /= gene_counts[mask]

    return scores, lr_stats
