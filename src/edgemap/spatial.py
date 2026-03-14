"""
Spatial graph construction and communication inference.

Steps 1-2 of the pipeline: ST data -> spatial graph -> communication scores.

The spatial graph (Gaussian-weighted KNN) serves two purposes:
  1. Define spatial neighborhoods for node score (expression specificity).
  2. Model ligand diffusion range for spatial-weighted communication.

Communication intensity per cell per LR pair is computed via spatial-weighted
product: comm(j) = R_eff(j) * (W @ L_eff)(j). This models mass-action kinetics
with spatial ligand diffusion: the Gaussian-weighted spatial graph models
ligand reaching nearby cells, and the product with receptor expression models
binding at the receiver. Validated by proximity ligation assay against actual
protein-protein interactions (CytoSignal, bioRxiv 2024).
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .config import SpatialConfig, get_lr_database


def _looks_like_counts(adata: ad.AnnData) -> bool:
    """Heuristic check: does X look like raw integer counts?

    Checks two properties:
      1. Non-negative values
      2. Integer-valued (within tolerance for float storage)

    Deliberately does NOT check magnitude — low-depth Visium data can
    legitimately have small max counts.
    """
    X = adata.X
    if sparse.issparse(X):
        sample = X.data[:10000]
    else:
        sample = np.asarray(X).ravel()[:10000]
    if len(sample) == 0:
        return True
    return bool(
        np.all(sample >= 0)
        and np.allclose(sample, np.round(sample), atol=0.01)
    )


def load_st(path: str, cfg: SpatialConfig) -> ad.AnnData:
    """Load and preprocess spatial transcriptomics data.

    Expects raw counts by default. Set cfg.preprocessed=True to skip
    normalization if data is already log1p-normalized.

    Raises ValueError if data looks pre-processed but preprocessed=False,
    to prevent silent double-normalization.
    """
    adata = ad.read_h5ad(path)

    if "spatial" not in adata.obsm:
        raise ValueError("h5ad must contain .obsm['spatial']")

    sc.pp.filter_genes(adata, min_cells=cfg.min_cells_per_gene)

    if cfg.preprocessed:
        return adata

    if not _looks_like_counts(adata):
        raise ValueError(
            "Expression values look pre-processed (non-integer or negative). "
            "EdgeMap expects raw counts. Either provide raw count data, or set "
            "preprocessed=True (CLI: --preprocessed) to skip normalization."
        )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def build_spatial_graph(
    coords: np.ndarray,
    k: int,
    d_max: float,
    kernel_bandwidth_frac: float = 1.0 / 3.0,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """KNN spatial graph with Gaussian distance decay.

    Returns:
        W: symmetric sparse weight matrix (n x n), edges beyond d_max excluded
        knn_idx: neighbor indices (n x k+1), column 0 = self
        knn_valid: boolean mask (n x k+1), True for neighbors within d_max
    """
    if d_max <= 0:
        raise ValueError("d_max must be > 0")
    if kernel_bandwidth_frac <= 0:
        raise ValueError("kernel_bandwidth_frac must be > 0")

    n = len(coords)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nn.fit(coords)
    dist, idx = nn.kneighbors(coords)

    # Vectorized graph construction -- no Python loops
    dist_k = dist[:, 1:]  # strip self-neighbor
    idx_k = idx[:, 1:]
    sigma = d_max * kernel_bandwidth_frac

    mask = dist_k <= d_max
    rows = np.repeat(np.arange(n), idx_k.shape[1]).reshape(n, -1)[mask]
    cols = idx_k[mask]
    wts = np.exp(-dist_k[mask] ** 2 / (2 * sigma**2))

    W = sparse.csr_matrix((wts, (rows, cols)), shape=(n, n))
    W = (W + W.T) * 0.5  # symmetrize

    # Validity mask: consistent neighborhood for node and edge scores
    knn_valid = dist <= d_max  # column 0 (self, dist=0) always True

    return W, idx, knn_valid


# ── LR database loading ─────────────────────────────────────────────


def load_lr_pairs(
    adata: ad.AnnData,
    heteromeric_delimiter: str = "_",
    min_cell_pct: float = 0.05,
) -> list[tuple[list[str], list[str], str]]:
    """Load LIANA Consensus LR pairs and filter to those active in this dataset.

    Uses the LIANA Consensus resource (4,624 curated LR pairs from the union
    of CellChatDB, CellPhoneDB, connectomeDB2020, Ramilowski2015, and others,
    filtered by literature support and protein localization).

    An LR pair is active when every subunit gene is expressed in at least
    min_cell_pct of cells. Heteromeric complexes (e.g. ERBB2_ERBB3) are
    split into individual subunit genes.

    Returns:
        List of (ligand_genes, receptor_genes, pair_label) tuples.
    """
    lr_path = get_lr_database()

    try:
        df = pd.read_csv(lr_path)
    except FileNotFoundError:
        warnings.warn(f"LR database not found at {lr_path}. Edge scores will be zero.")
        return []

    df = df[df["resource"] == "consensus"]

    # Precompute cells-per-gene for filtering (O(nnz), done once)
    X = adata.X
    n_cells = X.shape[0]
    min_cells = int(n_cells * min_cell_pct)
    if sparse.issparse(X):
        cells_per_gene = np.array((X > 0).sum(axis=0)).flatten()
    else:
        cells_per_gene = (np.asarray(X) > 0).sum(axis=0)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    gene_set = set(adata.var_names)

    def subunits_ok(complex_name: str) -> tuple[bool, list[str]]:
        genes = complex_name.split(heteromeric_delimiter)
        for g in genes:
            if g not in gene_set:
                return False, genes
            if cells_per_gene[gene_to_idx[g]] < min_cells:
                return False, genes
        return True, genes

    pairs = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        lig_ok, lig_genes = subunits_ok(str(row["source_genesymbol"]))
        rec_ok, rec_genes = subunits_ok(str(row["target_genesymbol"]))
        if not (lig_ok and rec_ok):
            continue
        label = f"{'_'.join(lig_genes)}-{'_'.join(rec_genes)}"
        if label in seen:
            continue
        seen.add(label)
        pairs.append((lig_genes, rec_genes, label))

    return pairs


# ── Spatial communication ────────────────────────────────────────────


def compute_communication(
    X: np.ndarray | sparse.spmatrix,
    W: sparse.csr_matrix,
    pairs: list[tuple[list[str], list[str], str]],
    gene_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, tuple[list[str], list[str]]]]:
    """Spatial-weighted communication intensity for each LR pair.

    Model (mass-action with spatial diffusion):
      L_eff(i) = min(expr(i, l) for l in ligand_subunits)   [bottleneck]
      R_eff(i) = min(expr(i, r) for r in receptor_subunits) [bottleneck]
      comm(i)  = (W @ L_eff)(i) * R_eff(i)

    The min over subunits implements the bottleneck model: a heteromeric
    complex functions at the rate of its least-expressed subunit.

    Expression is converted to natural scale (expm1 of log1p values) before
    computation, because L * R models mass-action kinetics on concentrations.

    Args:
        X: log1p-normalized expression (n_cells, n_genes), sparse or dense
        W: spatial weight matrix (n_cells, n_cells), sparse
        pairs: from load_lr_pairs
        gene_names: gene name list aligned with X columns

    Returns:
        comm: (n_cells, n_pairs) communication intensity
        pair_names: list of pair labels
        pair_genes: dict mapping label -> (lig_genes, rec_genes)
    """
    n_cells = X.shape[0]
    if not pairs:
        return np.empty((n_cells, 0), dtype=np.float64), [], {}

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Pre-extract all needed gene columns to natural scale (expm1).
    # Many pairs share genes (e.g. ITGB1 in 50+ pairs), so caching
    # avoids redundant sparse column extraction + todense + expm1.
    needed_indices = set()
    for ligs, recs, _ in pairs:
        for g in ligs + recs:
            needed_indices.add(gene_to_idx[g])
    col_cache = _preextract_columns(X, sorted(needed_indices))

    comm = np.empty((n_cells, len(pairs)), dtype=np.float64)
    pair_names = []
    pair_genes = {}

    for pi, (ligs, recs, label) in enumerate(pairs):
        L_eff = _subunit_eff_cached(col_cache, [gene_to_idx[g] for g in ligs])
        R_eff = _subunit_eff_cached(col_cache, [gene_to_idx[g] for g in recs])

        # Spatial-weighted product: diffused ligand × local receptor
        comm[:, pi] = np.asarray(W @ L_eff).flatten() * R_eff

        pair_names.append(label)
        pair_genes[label] = (ligs, recs)

    return comm, pair_names, pair_genes


def _preextract_columns(
    X: np.ndarray | sparse.spmatrix,
    indices: list[int],
) -> dict[int, np.ndarray]:
    """Extract gene columns to natural scale (expm1), one todense per gene."""
    cache = {}
    for idx in indices:
        col = X[:, idx]
        if sparse.issparse(col):
            col = np.asarray(col.todense()).flatten()
        else:
            col = np.asarray(col).flatten()
        cache[idx] = np.expm1(col)
    return cache


def _subunit_eff_cached(
    col_cache: dict[int, np.ndarray],
    col_indices: list[int],
) -> np.ndarray:
    """Effective expression from cached natural-scale columns.

    Single gene: that gene's cached column.
    Multiple subunits: min across subunits (bottleneck model).
    """
    if len(col_indices) == 1:
        return col_cache[col_indices[0]]
    return np.minimum.reduce([col_cache[i] for i in col_indices])
