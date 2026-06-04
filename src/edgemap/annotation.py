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


def build_multi_annotation_ldscores(
    gene_scores: dict[str, pd.Series],
    resource_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Map multiple gene-level score vectors to SNP-level annotation LD scores.

    Args:
        gene_scores: Mapping from annotation name (without ``ell_`` prefix) to
            gene-level score Series indexed by gene symbol.
        resource_dir: Optional gsMap resource directory.

    Returns:
        annot_ld: DataFrame with columns [SNP, ell_<name> ...]
        diagnostics: Per-annotation nonzero counts plus pairwise gene/SNP
            correlations when more than one annotation is supplied.
    """
    if not gene_scores:
        raise ValueError("gene_scores must contain at least one annotation")

    W, snp_names, wm_genes = _get_snp_gene_weights(resource_dir)

    names = list(gene_scores.keys())
    gene_matrix = np.column_stack([
        gene_scores[name].reindex(wm_genes, fill_value=0.0).values.astype(np.float64)
        for name in names
    ])
    ell_matrix = np.asarray(W @ gene_matrix)

    annot_ld = pd.DataFrame({"SNP": snp_names})
    diagnostics = {"annotations": {}}

    for i, name in enumerate(names):
        col_name = f"ell_{name}"
        annot_ld[col_name] = ell_matrix[:, i]
        diagnostics["annotations"][name] = {
            "n_genes": int((gene_matrix[:, i] > 0).sum()),
            "n_snps": int((ell_matrix[:, i] > 0).sum()),
        }

    pairwise = {}
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            both_gene = (gene_matrix[:, i] > 0) & (gene_matrix[:, j] > 0)
            both_snp = (ell_matrix[:, i] > 0) & (ell_matrix[:, j] > 0)
            pairwise[f"{left}__{right}"] = {
                "n_both_genes": int(both_gene.sum()),
                "gene_corr": _safe_corrcoef(gene_matrix[both_gene, i], gene_matrix[both_gene, j]),
                "n_both_snps": int(both_snp.sum()),
                "snp_corr": _safe_corrcoef(ell_matrix[both_snp, i], ell_matrix[both_snp, j]),
            }
    diagnostics["pairwise"] = pairwise

    return annot_ld, diagnostics


def build_annotation_ldscores(
    node_scores: pd.Series,
    edge_scores: pd.Series,
    resource_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Map gene-level node/edge scores to SNP-level annotation LD scores."""
    annot_ld, diagnostics = build_multi_annotation_ldscores(
        {"node": node_scores, "edge": edge_scores},
        resource_dir=resource_dir,
    )

    pair_diag = diagnostics["pairwise"].get("node__edge", {})
    flat_diag = {
        "n_node_genes": diagnostics["annotations"]["node"]["n_genes"],
        "n_edge_genes": diagnostics["annotations"]["edge"]["n_genes"],
        "n_both_genes": pair_diag.get("n_both_genes", 0),
        "gene_corr": pair_diag.get("gene_corr", 0.0),
        "snp_corr": pair_diag.get("snp_corr", 0.0),
    }
    return annot_ld[["SNP", "ell_node", "ell_edge"]], flat_diag


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


def build_per_pair_membership_ldscores(
    pair_names: list[str],
    pair_genes: dict[str, tuple[list[str], list[str]]],
    resource_dir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build per-LR-pair binary membership LD scores.

    For each pair p, creates a gene-level binary indicator (1 if the gene
    belongs to the pair's ligands or receptors, 0 otherwise), then maps to
    SNP-level via W.  This captures the LD structure around the pair's genes
    independently of their communication specificity score.

    Used as a control in per-pair S-LDSC to ensure the pair's tau reflects
    spatially structured communication rather than gene-level functional
    importance.

    Args:
        pair_names: list of pair labels to process
        pair_genes: dict mapping label -> (lig_genes, rec_genes)
        resource_dir: gsMap resource directory (auto-detected if None)

    Returns:
        pair_membership_ld: dict mapping pair_name -> (n_snps,) LD score array
        snp_names: SNP name list aligned with the arrays
    """
    W, snp_names, wm_genes = _get_snp_gene_weights(resource_dir)
    name2i = {g: i for i, g in enumerate(wm_genes)}
    n_genes = len(wm_genes)

    if not pair_names:
        return {}, snp_names

    gene_matrix = np.zeros((n_genes, len(pair_names)), dtype=np.float64)
    for col, pname in enumerate(pair_names):
        ligs, recs = pair_genes[pname]
        for g in ligs + recs:
            if g in name2i:
                gene_matrix[name2i[g], col] = 1.0

    ell_matrix = np.asarray(W @ gene_matrix)

    pair_membership_ld = {}
    for col, pname in enumerate(pair_names):
        ell = ell_matrix[:, col]
        if ell.max() > 0:
            pair_membership_ld[pname] = ell

    return pair_membership_ld, snp_names
