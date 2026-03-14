import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from edgemap.spatial import (
    build_spatial_graph,
    compute_communication,
    load_lr_pairs,
    _looks_like_counts,
    _preextract_columns,
    _subunit_eff_cached,
)


def test_compute_communication_simple():
    """Verify comm(j) = (W @ L)(j) * R(j) on a 3-cell example."""
    # 3 cells, 2 genes: LIG at idx 0, REC at idx 1
    # Expression in log1p scale
    X = np.array([
        [1.0, 0.5],
        [0.0, 2.0],
        [0.5, 0.0],
    ])

    # Spatial weights: cell 0 → cell 1, cell 1 → cell 0, cell 2 isolated
    W = sparse.csr_matrix(np.array([
        [0.0, 0.8, 0.0],
        [0.8, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]))

    pairs = [(["LIG"], ["REC"], "LIG-REC")]
    genes = ["LIG", "REC"]

    comm, names, pgenes = compute_communication(X, W, pairs, genes)

    assert names == ["LIG-REC"]
    assert pgenes["LIG-REC"] == (["LIG"], ["REC"])
    assert comm.shape == (3, 1)

    # Manual: L_nat = expm1(X[:, 0]) = [e^1-1, 0, e^0.5-1] ≈ [1.718, 0, 0.649]
    # spatial_L = W @ L_nat:
    #   cell 0: 0.8 * 0     = 0
    #   cell 1: 0.8 * 1.718 = 1.374
    #   cell 2: 0
    # R_nat = expm1(X[:, 1]) = [e^0.5-1, e^2-1, 0] ≈ [0.649, 6.389, 0]
    # comm = spatial_L * R_nat = [0, 1.374*6.389, 0] ≈ [0, 8.78, 0]
    L_nat = np.expm1(X[:, 0])
    R_nat = np.expm1(X[:, 1])
    expected = (W @ L_nat) * R_nat

    np.testing.assert_allclose(comm[:, 0], expected, rtol=1e-10)


def test_compute_communication_heteromeric():
    """Heteromeric complex uses min of subunits."""
    # 3 cells, 3 genes: LIG1(idx 0), LIG2(idx 1), REC(idx 2)
    X = np.array([
        [2.0, 0.5, 1.0],  # LIG1 high, LIG2 low → L_eff = LIG2
        [0.5, 2.0, 0.5],  # LIG1 low, LIG2 high → L_eff = LIG1
        [1.0, 1.0, 1.5],  # equal
    ])
    W = sparse.eye(3, format="csr") * 0.5  # self-loop only

    pairs = [(["LIG1", "LIG2"], ["REC"], "LIG1_LIG2-REC")]
    genes = ["LIG1", "LIG2", "REC"]

    comm, names, pgenes = compute_communication(X, W, pairs, genes)

    L1 = np.expm1(X[:, 0])
    L2 = np.expm1(X[:, 1])
    L_eff = np.minimum(L1, L2)
    R_nat = np.expm1(X[:, 2])
    expected = (W @ L_eff) * R_nat

    np.testing.assert_allclose(comm[:, 0], expected, rtol=1e-10)


def test_compute_communication_sparse_X():
    """Works with sparse expression matrix."""
    X_dense = np.array([[1.0, 0.0], [0.0, 1.5], [0.5, 0.5]])
    X_sparse = sparse.csr_matrix(X_dense)
    W = sparse.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=float))

    pairs = [(["G1"], ["G2"], "G1-G2")]
    genes = ["G1", "G2"]

    c_dense, _, _ = compute_communication(X_dense, W, pairs, genes)
    c_sparse, _, _ = compute_communication(X_sparse, W, pairs, genes)

    np.testing.assert_allclose(c_dense, c_sparse, rtol=1e-10)


def test_compute_communication_no_pairs():
    """Empty pairs → empty output."""
    X = np.ones((5, 3))
    W = sparse.eye(5, format="csr")
    comm, names, pgenes = compute_communication(X, W, [], ["A", "B", "C"])
    assert comm.shape == (5, 0)
    assert names == []
    assert pgenes == {}


def test_subunit_expression_single():
    """Single gene: just expm1 of that column."""
    X = np.array([[0.0], [1.0], [2.0]])
    cache = _preextract_columns(X, [0])
    result = _subunit_eff_cached(cache, [0])
    np.testing.assert_allclose(result, np.expm1([0.0, 1.0, 2.0]))


def test_subunit_expression_multi():
    """Multiple subunits: min of expm1 values."""
    X = np.array([[2.0, 0.5], [0.5, 2.0], [1.0, 1.0]])
    cache = _preextract_columns(X, [0, 1])
    result = _subunit_eff_cached(cache, [0, 1])
    expected = np.minimum(np.expm1(X[:, 0]), np.expm1(X[:, 1]))
    np.testing.assert_allclose(result, expected)


def test_build_spatial_graph_invalid_dmax():
    """Distance threshold must be positive."""
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="d_max must be > 0"):
        build_spatial_graph(coords, k=1, d_max=0.0)
    with pytest.raises(ValueError, match="d_max must be > 0"):
        build_spatial_graph(coords, k=1, d_max=-1.0)


def test_build_spatial_graph_invalid_kernel_bandwidth():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="kernel_bandwidth_frac must be > 0"):
        build_spatial_graph(coords, k=1, d_max=2.0, kernel_bandwidth_frac=0.0)


def test_load_lr_pairs_filters_by_expression_resource_and_deduplicates(tmp_path, monkeypatch):
    adata = ad.AnnData(
        X=np.array(
            [
                [1, 1, 0, 1],  # G1, G2, G3, G4
                [1, 1, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 0, 1],
            ],
            dtype=np.float32,
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )

    lr_path = tmp_path / "lr.csv"
    pd.DataFrame(
        [
            # valid consensus pair
            {"resource": "consensus", "source_genesymbol": "G1", "target_genesymbol": "G2"},
            # duplicate of same pair
            {"resource": "consensus", "source_genesymbol": "G1", "target_genesymbol": "G2"},
            # filtered: non-consensus
            {"resource": "cellphonedb", "source_genesymbol": "G1", "target_genesymbol": "G2"},
            # filtered: missing gene
            {"resource": "consensus", "source_genesymbol": "MISSING", "target_genesymbol": "G2"},
            # filtered: low-expression gene G3
            {"resource": "consensus", "source_genesymbol": "G1", "target_genesymbol": "G3"},
            # valid heteromeric pair
            {"resource": "consensus", "source_genesymbol": "G1_G4", "target_genesymbol": "G2"},
        ]
    ).to_csv(lr_path, index=False)

    monkeypatch.setattr("edgemap.spatial.get_lr_database", lambda: lr_path)
    pairs = load_lr_pairs(adata, min_cell_pct=0.5)

    labels = [p[2] for p in pairs]
    assert labels == ["G1-G2", "G1_G4-G2"]


def test_load_lr_pairs_missing_database_warns_and_returns_empty(monkeypatch):
    adata = ad.AnnData(
        X=np.array([[1, 0], [0, 1]], dtype=np.float32),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    monkeypatch.setattr("edgemap.spatial.get_lr_database", lambda: "/no/such/lr.csv")

    with pytest.warns(UserWarning, match="LR database not found"):
        pairs = load_lr_pairs(adata)
    assert pairs == []


def test_build_spatial_graph_basic_properties():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    W, idx, valid = build_spatial_graph(coords, k=2, d_max=2.0, kernel_bandwidth_frac=0.5)

    assert W.shape == (4, 4)
    assert idx.shape == (4, 3)  # self + 2 neighbors
    assert valid.shape == (4, 3)
    assert np.all(valid[:, 0])  # self neighbors
    np.testing.assert_allclose(W.toarray(), W.toarray().T, rtol=1e-12)


def test_load_lr_pairs_sparse_input_branch(tmp_path, monkeypatch):
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [1, 1],
                    [1, 1],
                    [1, 0],
                ],
                dtype=np.float32,
            )
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    lr_path = tmp_path / "lr_sparse.csv"
    pd.DataFrame(
        [{"resource": "consensus", "source_genesymbol": "G1", "target_genesymbol": "G2"}]
    ).to_csv(lr_path, index=False)

    monkeypatch.setattr("edgemap.spatial.get_lr_database", lambda: lr_path)
    pairs = load_lr_pairs(adata, min_cell_pct=0.3)
    assert [p[2] for p in pairs] == ["G1-G2"]


def test_looks_like_counts_sparse_empty_data_returns_true():
    adata = ad.AnnData(X=sparse.csr_matrix((3, 4)))
    assert _looks_like_counts(adata) is True


def test_load_lr_pairs_min_cells_at_least_one(tmp_path, monkeypatch):
    """With very few cells, min_cells should be at least 1 (not 0)."""
    adata = ad.AnnData(
        X=np.array(
            [
                [0, 1],  # G1 absent, G2 present
                [0, 1],
            ],
            dtype=np.float32,
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    lr_path = tmp_path / "lr.csv"
    pd.DataFrame(
        [{"resource": "consensus", "source_genesymbol": "G1", "target_genesymbol": "G2"}]
    ).to_csv(lr_path, index=False)

    monkeypatch.setattr("edgemap.spatial.get_lr_database", lambda: lr_path)
    # min_cell_pct=0.05 on 2 cells → int(0.1) = 0 without the fix
    # G1 expressed in 0 cells, should be filtered out even with 2 cells
    pairs = load_lr_pairs(adata, min_cell_pct=0.05)
    assert pairs == []
