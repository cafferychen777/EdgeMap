"""
Spatial graph construction and LR activity-proxy computation.

Steps 1-2 of the pipeline: ST data -> spatial graph -> communication scores.

The spatial graph (Gaussian-weighted KNN) serves two purposes:
  1. Define spatial neighborhoods for node score (expression specificity).
  2. Weight ligand expression around each receptor-expressing spatial unit.

The LR activity proxy per cell and curated LR label is the product of local
receptor expression and the Gaussian-weighted mean expression of neighboring
ligand. Row normalization makes uniform expression a uniform-activity null,
independent of graph degree. This mass-action-inspired expression proxy does
not measure diffusion, binding, signaling flux, or a molecular interaction.
"""
import math
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from ._matrix import ensure_csc_matrix
from .config import SpatialConfig, get_lr_database


def _stored_values(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Return explicit matrix values; implicit sparse zeros need no validation."""
    if sparse.issparse(X):
        return X.data
    return np.asarray(X).ravel()


def _validate_nonnegative_finite_expression(adata: ad.AnnData) -> None:
    """Reject expression scales that cannot represent counts or log1p values."""
    values = _stored_values(adata.X)
    if not np.all(np.isfinite(values)):
        raise ValueError("Expression matrix contains non-finite values")
    if np.any(values < 0):
        raise ValueError(
            "Expression matrix contains negative values. EdgeMap requires raw "
            "counts or non-negative log1p-normalized expression; scaled/z-scored "
            "values are not valid input."
        )


def _looks_like_counts(adata: ad.AnnData) -> bool:
    """Check whether X contains finite, non-negative integer counts.

    Checks two properties:
      1. Non-negative values
      2. Integer-valued (within tolerance for float storage)

    Deliberately does NOT check magnitude — low-depth Visium data can
    legitimately have small max counts.
    """
    values = _stored_values(adata.X)
    if len(values) == 0:
        return True
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.allclose(values, np.round(values), atol=0.01)
    )


def preprocess_st(adata: ad.AnnData, cfg: SpatialConfig) -> ad.AnnData:
    """Validate and preprocess spatial transcriptomics data (in-place).

    Expects raw counts by default. Set ``input_scale='log1p'`` (or the legacy
    ``preprocessed=True`` alias) only when X is already non-negative,
    log1p-normalized expression.

    Raises ValueError if coordinates or expression violate the declared scale.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("h5ad must contain .obsm['spatial']")
    if adata.n_obs == 0:
        raise ValueError("Expression matrix must contain at least one cell")
    if adata.n_vars == 0:
        raise ValueError("Expression matrix must contain at least one gene")
    if not adata.var_names.is_unique:
        raise ValueError(
            "Gene names (.var_names) must be unique. EdgeMap uses gene symbols "
            "as keys for ligand-receptor matching and SNP-gene mapping; "
            "aggregate duplicate genes upstream before running."
        )

    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 1:
        raise ValueError(
            ".obsm['spatial'] must be a finite 2D array with one row per cell"
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError(".obsm['spatial'] contains non-finite coordinates")

    _validate_nonnegative_finite_expression(adata)
    input_scale = cfg.resolved_input_scale
    if input_scale == "raw_counts" and not _looks_like_counts(adata):
        raise ValueError(
            "Expression values look pre-processed (non-integer). "
            "EdgeMap expects raw counts. Either provide raw count data, or set "
            "input_scale='log1p' (CLI: --input-scale log1p)."
        )

    # Validate the declared scale on the complete matrix before filtering so a
    # malformed low-prevalence gene cannot silently escape the data contract.
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells_per_gene)
    if adata.n_vars == 0:
        raise ValueError(
            "No genes remain after min_cells_per_gene filtering; lower the "
            "threshold or provide a larger expression matrix"
        )

    if input_scale == "log1p":
        return adata

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def load_st(path: str, cfg: SpatialConfig) -> ad.AnnData:
    """Load spatial transcriptomics data from h5ad and preprocess."""
    return preprocess_st(ad.read_h5ad(path), cfg)


def build_spatial_graph(
    coords: np.ndarray,
    k: int,
    d_max: float,
    kernel_bandwidth_frac: float = 1.0 / 3.0,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """KNN spatial graph with Gaussian distance decay.

    Returns:
        W: symmetric sparse weight matrix (n x n), edges beyond d_max excluded
        knn_idx: neighbor indices (n x min(k+1, n)), column 0 = self
        knn_valid: aligned boolean mask, True for neighbors within d_max
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or len(coords) == 0 or coords.shape[1] < 1:
        raise ValueError("coords must be a non-empty 2D array")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite values")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be >= 1")
    if not np.isfinite(d_max) or d_max <= 0:
        raise ValueError("d_max must be finite and > 0")
    if not np.isfinite(kernel_bandwidth_frac) or kernel_bandwidth_frac <= 0:
        raise ValueError("kernel_bandwidth_frac must be finite and > 0")

    n = len(coords)
    n_neighbors = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nn.fit(coords)
    raw_dist, raw_idx = nn.kneighbors(coords)

    # Put self in column zero explicitly. With duplicate coordinates sklearn's
    # tie ordering does not guarantee that the first returned neighbor is self.
    idx = np.empty((n, n_neighbors + 1), dtype=np.int64)
    dist = np.empty((n, n_neighbors + 1), dtype=np.float64)
    idx[:, 0] = np.arange(n)
    dist[:, 0] = 0.0
    for i in range(n):
        keep = raw_idx[i] != i
        candidates = raw_idx[i][keep][:n_neighbors]
        candidate_dist = raw_dist[i][keep][:n_neighbors]
        if len(candidates) < n_neighbors:
            raise RuntimeError("Unable to construct a self-excluding KNN graph")
        idx[i, 1:] = candidates
        dist[i, 1:] = candidate_dist

    # Vectorized sparse graph construction
    dist_k = dist[:, 1:]  # strip self-neighbor
    idx_k = idx[:, 1:]
    sigma = d_max * kernel_bandwidth_frac

    mask = dist_k <= d_max
    rows = np.repeat(np.arange(n), idx_k.shape[1]).reshape(n, -1)[mask]
    cols = idx_k[mask]
    wts = np.exp(-dist_k[mask] ** 2 / (2 * sigma**2))

    W = sparse.csr_matrix((wts, (rows, cols)), shape=(n, n))
    # Union symmetrization preserves the Gaussian weight for a one-sided KNN
    # edge instead of silently halving it.
    W = W.maximum(W.T).tocsr()

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
    of CellChatDB, CellPhoneDB, ICELLNET, connectomeDB2020, and CellTalkDB,
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
    if not np.isfinite(min_cell_pct) or not (0 <= min_cell_pct <= 1):
        raise ValueError("min_cell_pct must be finite and in [0, 1]")
    min_cells = max(math.ceil(n_cells * min_cell_pct), 1)
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
    """Spatially weighted LR activity proxy for each curated LR label.

    Mass-action-inspired expression proxy:
      L_eff(i) = min(expr(i, l) for l in ligand_subunits)   [bottleneck]
      R_eff(i) = min(expr(i, r) for r in receptor_subunits) [bottleneck]
      comm(i)  = row_mean_W(L_eff)(i) * R_eff(i)

    The min over subunits implements the bottleneck model: a heteromeric
    complex functions at the rate of its least-expressed subunit.

    Expression is converted back from log1p before multiplication. The product
    is an expression-based scoring convention, not a concentration, binding, or
    signaling-flux measurement.

    Args:
        X: log1p-normalized expression (n_cells, n_genes), sparse or dense
        W: spatial weight matrix (n_cells, n_cells), sparse. Neighbor exposure
            is row-normalized inside this function so uniform expression is a
            uniform-activity null regardless of graph degree. Cells without a
            valid spatial neighbor receive NaN and are excluded downstream
            because their neighboring-ligand mean is undefined.
        pairs: from load_lr_pairs
        gene_names: gene name list aligned with X columns

    Returns:
        comm: (n_cells, n_pairs) LR activity proxy; rows without a valid
            spatial neighbor are NaN
        pair_names: list of pair labels
        pair_genes: dict mapping label -> (lig_genes, rec_genes)
    """
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional expression matrix")
    n_cells, n_genes = X.shape
    if W.shape != (n_cells, n_cells):
        raise ValueError(
            f"W must have shape ({n_cells}, {n_cells}), got {W.shape}"
        )
    if len(gene_names) != n_genes:
        raise ValueError(
            "gene_names length must match the number of expression columns"
        )
    if len(gene_names) != len(set(gene_names)):
        raise ValueError("gene_names must be unique")

    expression_values = _stored_values(X)
    if not np.all(np.isfinite(expression_values)) or np.any(expression_values < 0):
        raise ValueError("X must contain finite, non-negative log1p expression")
    if not sparse.issparse(W):
        W = sparse.csr_matrix(W)
    else:
        W = W.tocsr()
    if not np.all(np.isfinite(W.data)) or np.any(W.data < 0):
        raise ValueError("W must contain finite, non-negative spatial weights")

    if not pairs:
        return np.empty((n_cells, 0), dtype=np.float64), [], {}

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    labels = [label for _, _, label in pairs]
    if len(labels) != len(set(labels)):
        raise ValueError("LR pair labels must be unique")
    missing_genes = sorted({
        gene
        for ligands, receptors, _ in pairs
        for gene in ligands + receptors
        if gene not in gene_to_idx
    })
    if missing_genes:
        preview = ", ".join(missing_genes[:5])
        suffix = "..." if len(missing_genes) > 5 else ""
        raise ValueError(f"LR pairs reference genes absent from X: {preview}{suffix}")

    # Convert to CSC for efficient column slicing (same as compute_node_scores)
    X_work = ensure_csc_matrix(X)

    # Pre-extract all needed gene columns to natural scale (expm1).
    # Many pairs share genes (e.g. ITGB1 in 50+ pairs), so caching
    # avoids redundant sparse column extraction + expm1.
    needed_indices = set()
    for ligs, recs, _ in pairs:
        for g in ligs + recs:
            needed_indices.add(gene_to_idx[g])
    col_cache = _preextract_columns(X_work, sorted(needed_indices))
    row_weight = np.asarray(W.sum(axis=1)).ravel()
    if not np.all(np.isfinite(row_weight)):
        raise ValueError("W has non-finite row-weight sums")
    inv_row_weight = np.zeros_like(row_weight, dtype=np.float64)
    nonisolated = row_weight > 0
    inv_row_weight[nonisolated] = 1.0 / row_weight[nonisolated]

    comm = np.empty((n_cells, len(pairs)), dtype=np.float64)
    pair_names = []
    pair_genes = {}

    for pi, (ligs, recs, label) in enumerate(pairs):
        L_eff = _subunit_eff_cached(col_cache, [gene_to_idx[g] for g in ligs])
        R_eff = _subunit_eff_cached(col_cache, [gene_to_idx[g] for g in recs])

        # Weighted mean neighboring ligand expression x local receptor expression.
        neighbor_ligand = np.full(n_cells, np.nan, dtype=np.float64)
        weighted_ligand = np.asarray(W @ L_eff).ravel()
        neighbor_ligand[nonisolated] = (
            weighted_ligand[nonisolated] * inv_row_weight[nonisolated]
        )
        pair_activity = neighbor_ligand * R_eff
        if not np.all(np.isfinite(pair_activity[nonisolated])):
            raise ValueError(
                f"Natural-scale LR activity overflowed for pair {label!r}; "
                "verify that X is ordinary log1p expression"
            )
        comm[:, pi] = pair_activity

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
            col = col.toarray().flatten()
        else:
            col = np.asarray(col).flatten()
        with np.errstate(over="ignore", invalid="ignore"):
            natural = np.expm1(np.asarray(col, dtype=np.float64))
        if not np.all(np.isfinite(natural)):
            raise ValueError(
                "Natural-scale expression overflowed during expm1; verify "
                "that X is ordinary log1p expression"
            )
        cache[idx] = natural
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
