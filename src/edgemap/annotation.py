"""
SNP annotation construction via gene-level scores.

Step 5 of the pipeline: gene scores -> SNP annotation LD scores.

The bridge between gene-level biology and SNP-level genetics is gsMap's
pre-computed SNP-gene weight matrix W (M_snps x G_genes).
Entry W[s,g] = sum_{s'} r^2(s,s') * 1(s' in cis(g)), encoding the
LD-weighted cis-indicator.

Multiplying W @ gene_scores gives annotation LD scores directly, combining
gene-SNP mapping and LD computation in a single sparse matrix-vector product.

No explicit orthogonalization is performed. The joint S-LDSC regression
with both node and edge annotations handles confounding by the
Frisch-Waugh-Lovell theorem: each tau coefficient already reflects the
unique contribution of its annotation after controlling for all others,
including all baseline annotations.
"""

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse

from .config import resolve_resource_dir


# Module-level cache: keyed by resolved resource path
_weight_cache: dict[str, tuple] = {}


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Correlation helper that returns 0.0 for degenerate inputs."""
    if len(x) < 3 or len(y) < 3:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _get_snp_gene_weights(
    resource_dir: str | Path | None = None,
) -> tuple[sparse.spmatrix, list[str], list[str]]:
    """Load the pre-computed SNP-gene weight matrix (cached).

    Keeps the matrix sparse -- the cis-window structure means most entries
    are zero, so sparse matmul is both faster and more memory-efficient.
    """
    rdir = resolve_resource_dir(resource_dir)
    key = str(rdir)
    if key not in _weight_cache:
        path = rdir / "quick_mode" / "snp_gene_weight_matrix.h5ad"
        wm = ad.read_h5ad(str(path))
        W = wm.X
        if not sparse.issparse(W):
            W = sparse.csr_matrix(W)
        else:
            W = W.tocsr()
        _weight_cache[key] = (
            W.astype(np.float64),
            wm.obs_names.tolist(),
            wm.var_names.tolist(),
        )
    return _weight_cache[key]


def build_annotation_ldscores(
    node_scores: pd.Series,
    edge_scores: pd.Series,
    resource_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Map gene-level scores to SNP-level annotation LD scores.

    1. Align gene scores to the weight matrix gene order.
    2. Sparse matmul: ell = W @ gene_scores.

    No orthogonalization: the joint S-LDSC regression with baseline
    annotations handles confounding via Frisch-Waugh-Lovell. Explicit
    Gram-Schmidt in gene space would be mathematically incorrect since
    W is not orthonormal (gene-space orthogonality != SNP-space orthogonality).

    Returns:
        annot_ld: DataFrame with columns [SNP, ell_node, ell_edge]
        diagnostics: dict with gene counts and correlation stats
    """
    W, snp_names, wm_genes = _get_snp_gene_weights(resource_dir)

    # Align gene scores to weight matrix columns
    node_vec = node_scores.reindex(wm_genes, fill_value=0.0).values.astype(np.float64)
    edge_vec = edge_scores.reindex(wm_genes, fill_value=0.0).values.astype(np.float64)

    # Diagnostic: gene-level correlation between the two scores
    both_active = (node_vec > 0) & (edge_vec > 0)
    corr_gene = _safe_corrcoef(node_vec[both_active], edge_vec[both_active])

    # Sparse matrix-vector product: O(nnz) instead of O(M * G)
    ell_node = np.asarray(W @ node_vec).flatten()
    ell_edge = np.asarray(W @ edge_vec).flatten()

    # SNP-level correlation (diagnostic for checking independence)
    both_snp = (ell_node > 0) & (ell_edge > 0)
    corr_snp = _safe_corrcoef(ell_node[both_snp], ell_edge[both_snp])

    annot_ld = pd.DataFrame({
        "SNP": snp_names,
        "ell_node": ell_node,
        "ell_edge": ell_edge,
    })

    diagnostics = {
        "n_node_genes": int((node_vec > 0).sum()),
        "n_edge_genes": int((edge_vec > 0).sum()),
        "n_both_genes": int(both_active.sum()),
        "gene_corr": corr_gene,
        "snp_corr": corr_snp,
    }

    return annot_ld, diagnostics


def build_per_pair_ldscores(
    pair_names: list[str],
    pair_genes: dict[str, tuple[list[str], list[str]]],
    pair_scores: dict[str, float],
    resource_dir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build per-LR-pair SNP annotation LD scores for conditional testing.

    For each pair p, constructs a gene-level score where only the participating
    genes receive the pair's specificity score, then maps to SNP-level via W.

    Args:
        pair_names: list of pair labels
        pair_genes: dict mapping label -> (lig_genes, rec_genes)
        pair_scores: dict mapping label -> PairScore value
        resource_dir: gsMap resource directory (auto-detected if None)

    Returns:
        pair_ld: dict mapping pair_name -> (n_snps,) LD score array
        snp_names: SNP name list aligned with the arrays
    """
    W, snp_names, wm_genes = _get_snp_gene_weights(resource_dir)
    name2i = {g: i for i, g in enumerate(wm_genes)}
    n_genes = len(wm_genes)

    # Filter to pairs with positive scores
    active_pairs = [
        pname for pname in pair_names
        if pair_scores.get(pname, 0.0) > 0
    ]
    if not active_pairs:
        return {}, snp_names

    # Batch: stack all gene vectors into a matrix, single sparse matmul
    gene_matrix = np.zeros((n_genes, len(active_pairs)), dtype=np.float64)
    for col, pname in enumerate(active_pairs):
        score = pair_scores[pname]
        ligs, recs = pair_genes[pname]
        for g in ligs + recs:
            if g in name2i:
                gene_matrix[name2i[g], col] = score

    ell_matrix = np.asarray(W @ gene_matrix)  # (n_snps, n_pairs) — single matmul

    pair_ld = {}
    for col, pname in enumerate(active_pairs):
        ell = ell_matrix[:, col]
        if ell.max() > 0:
            pair_ld[pname] = ell

    return pair_ld, snp_names
