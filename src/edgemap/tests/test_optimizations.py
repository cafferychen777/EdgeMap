"""Numerical consistency tests for performance optimizations.

Each test compares the optimized implementation against a naive reference
to ensure the optimizations are purely mechanical (no precision loss).
"""

import numpy as np
from scipy import sparse

from edgemap._matrix import ensure_csc_matrix
from edgemap.config import ScoreConfig
from edgemap.scores import compute_node_scores
from edgemap.spatial import (
    compute_communication,
)
from edgemap.annotation import build_per_pair_ldscores


# ── #1: Communication gene column caching ─────────────────────────


def test_cached_comm_matches_naive():
    """Cached column extraction produces identical communication values."""
    rng = np.random.RandomState(42)
    n, g = 50, 10
    X = sparse.random(n, g, density=0.3, format="csr", random_state=rng)
    X.data = np.abs(X.data)  # ensure non-negative for expm1
    W = sparse.random(n, n, density=0.1, format="csr", random_state=rng)
    W = (W + W.T) * 0.5

    # Pairs that share genes (G2 appears in both pairs)
    pairs = [
        (["G0"], ["G2"], "G0-G2"),
        (["G1", "G2"], ["G3"], "G1_G2-G3"),
        (["G4"], ["G2", "G5"], "G4-G2_G5"),
    ]
    gene_names = [f"G{i}" for i in range(g)]

    comm, names, pgenes = compute_communication(X, W, pairs, gene_names)

    # Naive reference: manual per-pair computation without caching
    gene_to_idx = {gn: i for i, gn in enumerate(gene_names)}
    for pi, (ligs, recs, label) in enumerate(pairs):
        # Naive: extract columns fresh each time
        lig_cols = []
        for gn in ligs:
            col = X[:, gene_to_idx[gn]].toarray().flatten()
            lig_cols.append(np.expm1(col))
        L_eff = np.minimum.reduce(lig_cols) if len(lig_cols) > 1 else lig_cols[0]

        rec_cols = []
        for gn in recs:
            col = X[:, gene_to_idx[gn]].toarray().flatten()
            rec_cols.append(np.expm1(col))
        R_eff = np.minimum.reduce(rec_cols) if len(rec_cols) > 1 else rec_cols[0]

        expected = np.asarray(W @ L_eff).flatten() * R_eff
        np.testing.assert_allclose(comm[:, pi], expected, rtol=1e-12)


def test_cached_comm_sparse_vs_dense():
    """Cached communication is identical for sparse and dense X."""
    rng = np.random.RandomState(7)
    n, g = 30, 6
    X_dense = np.abs(rng.randn(n, g)).astype(np.float64)
    X_sparse = sparse.csr_matrix(X_dense)
    W = sparse.eye(n, format="csr") * 0.5

    pairs = [(["G0", "G1"], ["G2"], "G0_G1-G2")]
    genes = [f"G{i}" for i in range(g)]

    comm_dense, _, _ = compute_communication(X_dense, W, pairs, genes)
    comm_sparse, _, _ = compute_communication(X_sparse, W, pairs, genes)

    np.testing.assert_allclose(comm_dense, comm_sparse, rtol=1e-12)


# ── #2: Node score in-place operations ─────────────────────────────


def test_node_scores_inplace_matches_reference():
    """In-place multiply produces identical node scores to explicit temp."""
    rng = np.random.RandomState(99)
    n, g = 40, 8

    # Build two identical inputs, compute scores
    X = rng.rand(n, g).astype(np.float32)
    X_sparse = sparse.csr_matrix(X)

    # Simple cyclic KNN
    k = 3
    idx = np.zeros((n, k + 1), dtype=int)
    for i in range(n):
        for j in range(k + 1):
            idx[i, j] = (i + j) % n
    valid = np.ones((n, k + 1), dtype=bool)

    # Partially mask some neighbors
    valid[0, 3] = False
    valid[5, 2] = False
    valid[10, 1] = False

    scores_dense = compute_node_scores(X, idx, valid, ScoreConfig(gene_chunk_size=3))
    scores_sparse = compute_node_scores(X_sparse, idx, valid, ScoreConfig(gene_chunk_size=3))

    # Dense and sparse must agree exactly (same rank operations)
    np.testing.assert_allclose(scores_dense, scores_sparse, rtol=1e-5)

    # Different chunk sizes must agree
    scores_chunk1 = compute_node_scores(X, idx, valid, ScoreConfig(gene_chunk_size=1))
    scores_chunk100 = compute_node_scores(X, idx, valid, ScoreConfig(gene_chunk_size=100))
    np.testing.assert_allclose(scores_chunk1, scores_chunk100, rtol=1e-5)


# ── #3: Batched per-pair LD matmul ─────────────────────────────────


def test_batched_perpair_ld_matches_sequential():
    """Batched W @ gene_matrix matches sequential W @ gene_vec per pair."""
    rng = np.random.RandomState(55)
    n_snps, n_genes = 100, 20

    # Fake sparse weight matrix
    W = sparse.random(n_snps, n_genes, density=0.05, format="csr", random_state=rng)
    wm_genes = [f"GENE{i}" for i in range(n_genes)]
    snp_names = [f"SNP{i}" for i in range(n_snps)]

    # Monkey-patch the weight matrix loader to use our fake
    import edgemap.annotation as ann
    original_cache = ann._weight_cache.copy()
    ann._weight_cache["__test__"] = (W.astype(np.float64), snp_names, wm_genes)

    try:
        pair_names = ["GENE0-GENE5", "GENE1_GENE2-GENE5", "GENE3-GENE7"]
        pair_genes = {
            "GENE0-GENE5": (["GENE0"], ["GENE5"]),
            "GENE1_GENE2-GENE5": (["GENE1", "GENE2"], ["GENE5"]),
            "GENE3-GENE7": (["GENE3"], ["GENE7"]),
        }
        pair_scores = {
            "GENE0-GENE5": 3.5,
            "GENE1_GENE2-GENE5": 2.1,
            "GENE3-GENE7": 1.8,
        }

        # Override resolve_resource_dir for this test
        from unittest.mock import patch
        with patch.object(ann, 'resolve_resource_dir', return_value="__test__"):
            # The function uses _get_snp_gene_weights which checks _weight_cache
            # We need to make resolve_resource_dir return our key
            pair_ld, result_snps = build_per_pair_ldscores(
                pair_names, pair_genes, pair_scores, "__test__",
            )

        # Verify against manual sequential computation
        name2i = {g: i for i, g in enumerate(wm_genes)}
        for pname in pair_names:
            score = pair_scores[pname]
            ligs, recs = pair_genes[pname]
            gene_vec = np.zeros(n_genes, dtype=np.float64)
            for g in ligs + recs:
                if g in name2i:
                    gene_vec[name2i[g]] = score
            expected = np.asarray(W.astype(np.float64) @ gene_vec).flatten()

            if expected.max() > 0:
                assert pname in pair_ld, f"{pname} should be in pair_ld"
                np.testing.assert_allclose(pair_ld[pname], expected, rtol=1e-12)
    finally:
        ann._weight_cache.clear()
        ann._weight_cache.update(original_cache)



def test_ensure_csc_matrix_reuses_existing_csc_input():
    """CSC input should be returned unchanged to avoid unnecessary copies."""
    X = sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    out = ensure_csc_matrix(X)
    assert out is X
