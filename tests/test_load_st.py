import numpy as np
import pytest
import anndata as ad

from edgemap.config import SpatialConfig
from edgemap.spatial import load_st, _looks_like_counts


# ── _looks_like_counts ─────────────────────────────────────────────


def _make_adata(X, spatial=True):
    """Helper: wrap a matrix in an AnnData with spatial coords."""
    adata = ad.AnnData(X=X)
    if spatial:
        adata.obsm["spatial"] = np.random.rand(X.shape[0], 2)
    return adata


def test_counts_detected_for_integer_data():
    X = np.array([[5, 0, 3], [0, 12, 1]], dtype=np.float32)
    assert _looks_like_counts(_make_adata(X)) is True


def test_counts_detected_for_low_depth():
    """Low-depth Visium: max count can be small, should still be recognized."""
    X = np.array([[2, 0, 1], [0, 3, 0], [1, 0, 0]], dtype=np.float32)
    assert _looks_like_counts(_make_adata(X)) is True


def test_log_normalized_rejected():
    """log1p-normalized data has non-integer values → not counts."""
    X = np.log1p(np.array([[100, 0, 50], [0, 200, 10]], dtype=np.float32))
    assert _looks_like_counts(_make_adata(X)) is False


def test_negative_values_rejected():
    X = np.array([[1, -0.5], [2, 3]], dtype=np.float32)
    assert _looks_like_counts(_make_adata(X)) is False


# ── load_st ────────────────────────────────────────────────────────


def test_load_st_preprocessed_skips_normalization(tmp_path):
    """With preprocessed=True, data passes through without normalization."""
    X = np.log1p(np.array([[100, 50], [200, 10]], dtype=np.float32))
    adata = _make_adata(X)
    path = tmp_path / "test.h5ad"
    adata.write(path)

    cfg = SpatialConfig(preprocessed=True, min_cells_per_gene=1)
    result = load_st(str(path), cfg)

    # Values should be unchanged (no normalize_total + log1p applied)
    np.testing.assert_allclose(result.X.toarray() if hasattr(result.X, 'toarray') else result.X, X, rtol=1e-5)


def test_load_st_raw_counts_normalizes(tmp_path):
    """With preprocessed=False and integer counts, normalization is applied."""
    X = np.array([[100, 50], [200, 10]], dtype=np.float32)
    adata = _make_adata(X)
    path = tmp_path / "test.h5ad"
    adata.write(path)

    cfg = SpatialConfig(preprocessed=False, min_cells_per_gene=1)
    result = load_st(str(path), cfg)

    result_X = result.X.toarray() if hasattr(result.X, 'toarray') else np.asarray(result.X)
    # After normalize_total + log1p, values should differ from raw
    assert not np.allclose(result_X, X)
    # Should be non-negative (log1p output)
    assert np.all(result_X >= 0)


def test_load_st_raises_on_preprocessed_data_without_flag(tmp_path):
    """Passing log1p data without preprocessed=True raises ValueError."""
    X = np.log1p(np.array([[100, 50], [200, 10]], dtype=np.float32))
    adata = _make_adata(X)
    path = tmp_path / "test.h5ad"
    adata.write(path)

    cfg = SpatialConfig(preprocessed=False, min_cells_per_gene=1)
    with pytest.raises(ValueError, match="pre-processed"):
        load_st(str(path), cfg)


def test_load_st_raises_without_spatial(tmp_path):
    """Missing .obsm['spatial'] raises ValueError."""
    X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    adata = ad.AnnData(X=X)  # no spatial
    path = tmp_path / "test.h5ad"
    adata.write(path)

    cfg = SpatialConfig(min_cells_per_gene=1)
    with pytest.raises(ValueError, match="spatial"):
        load_st(str(path), cfg)


def test_load_st_raises_on_duplicate_gene_names(tmp_path):
    X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    adata = _make_adata(X)
    adata.var_names = ["G1", "G1"]
    path = tmp_path / "dup_genes.h5ad"
    adata.write(path)

    cfg = SpatialConfig(preprocessed=True, min_cells_per_gene=1)
    with pytest.raises(ValueError, match="must be unique"):
        load_st(str(path), cfg)
