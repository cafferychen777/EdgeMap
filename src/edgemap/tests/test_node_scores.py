import numpy as np
from scipy import sparse

from edgemap.config import ScoreConfig
from edgemap.scores import compute_node_scores


def _simple_knn(n, k=2):
    """Build trivial KNN: each cell's neighbors are the next k cells (cyclic)."""
    idx = np.zeros((n, k + 1), dtype=int)
    for i in range(n):
        idx[i, 0] = i  # self
        for j in range(1, k + 1):
            idx[i, j] = (i + j) % n
    valid = np.ones((n, k + 1), dtype=bool)
    return idx, valid


def test_node_scores_basic_shape():
    """Output shape matches number of genes."""
    n, g = 20, 5
    X = np.random.RandomState(42).rand(n, g).astype(np.float32)
    idx, valid = _simple_knn(n)
    scores = compute_node_scores(X, idx, valid, ScoreConfig())
    assert scores.shape == (g,)
    assert scores.dtype == np.float32


def test_node_scores_sparse_dense_agree():
    """Sparse and dense X produce identical node scores."""
    n, g = 30, 10
    rng = np.random.RandomState(123)
    X_dense = rng.rand(n, g).astype(np.float32)
    X_dense[X_dense < 0.3] = 0  # sparsify
    X_sparse = sparse.csr_matrix(X_dense)

    idx, valid = _simple_knn(n, k=3)

    scores_dense = compute_node_scores(X_dense, idx, valid, ScoreConfig())
    scores_sparse = compute_node_scores(X_sparse, idx, valid, ScoreConfig())

    np.testing.assert_allclose(scores_dense, scores_sparse, rtol=1e-5)


def test_knn_valid_masks_distant_neighbors():
    """When knn_valid masks out a neighbor, that neighbor's rank is excluded."""
    n, g = 10, 3
    X = np.random.RandomState(7).rand(n, g).astype(np.float32)
    idx, valid_all = _simple_knn(n, k=3)

    scores_all = compute_node_scores(X, idx, valid_all, ScoreConfig())

    # Mask out the farthest neighbor (column 3) for all cells
    valid_partial = valid_all.copy()
    valid_partial[:, 3] = False
    scores_partial = compute_node_scores(X, idx, valid_partial, ScoreConfig())

    # Scores should differ when neighborhood changes
    assert not np.allclose(scores_all, scores_partial)


def test_concentrated_expression_gets_high_score():
    """A gene concentrated in a spatial cluster scores higher than a spread-out gene."""
    n = 50
    rng = np.random.RandomState(42)
    X = np.zeros((n, 2), dtype=np.float32)
    # Gene 0: concentrated in first 5 cells (spatial cluster)
    X[:5, 0] = 10.0
    # Gene 1: same total expression spread uniformly with noise
    X[:, 1] = rng.exponential(1.0, n).astype(np.float32)

    idx, valid = _simple_knn(n, k=4)
    scores = compute_node_scores(X, idx, valid, ScoreConfig())

    assert scores[0] > scores[1]


def test_chunking_does_not_affect_result():
    """Different chunk sizes produce identical scores."""
    n, g = 20, 15
    X = np.random.RandomState(99).rand(n, g).astype(np.float32)
    idx, valid = _simple_knn(n, k=2)

    cfg_large = ScoreConfig(gene_chunk_size=100)
    cfg_small = ScoreConfig(gene_chunk_size=3)

    scores_large = compute_node_scores(X, idx, valid, cfg_large)
    scores_small = compute_node_scores(X, idx, valid, cfg_small)

    np.testing.assert_allclose(scores_large, scores_small, rtol=1e-5)
