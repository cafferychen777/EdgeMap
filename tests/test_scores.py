import numpy as np

from edgemap.config import ScoreConfig
from edgemap.scores import compute_edge_scores


# ── Edge score computation tests ────────────────────────────────────


def _make_comm(pair_data: dict[str, np.ndarray], genes: list[str]):
    """Helper: build (comm, pair_names, pair_genes) from simple spec.

    pair_data maps "LIG-REC" -> per-cell communication array.
    Ligand and receptor genes are inferred from the key (split on '-',
    subunits split on '_').
    """
    pair_names = []
    pair_genes = {}
    cols = []

    for label, values in pair_data.items():
        parts = label.split("-", 1)
        ligs = parts[0].split("_")
        recs = parts[1].split("_")
        pair_names.append(label)
        pair_genes[label] = (ligs, recs)
        cols.append(values)

    comm = np.column_stack(cols) if cols else np.empty((0, 0))
    return comm, pair_names, pair_genes


def test_spatially_concentrated_pair_gets_high_score():
    """When communication is concentrated in a few cells, score > 1."""
    # Concentrated: two cells with high comm, rest zero
    pair_comm = np.array([10, 10, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    comm, names, pgenes = _make_comm({"LIG-REC": pair_comm}, ["LIG", "REC"])
    scores, stats = compute_edge_scores(comm, names, pgenes, ["LIG", "REC"], ScoreConfig())

    # mean=2.0, spec=[5,5,0,...], P95 ≈ 4.75
    assert scores[0] > 1.0  # LIG
    assert scores[1] > 1.0  # REC
    assert np.isclose(scores[0], scores[1])  # same pair → same score
    assert "LIG-REC" in stats


def test_uniform_communication_gets_zero_score():
    """Uniform communication → specificity = 1 → filtered out (score = 0)."""
    n = 10
    pair_comm = np.ones(n)

    comm, names, pgenes = _make_comm({"LIG-REC": pair_comm}, ["LIG", "REC"])
    scores, _ = compute_edge_scores(comm, names, pgenes, ["LIG", "REC"], ScoreConfig())

    np.testing.assert_array_equal(scores, [0.0, 0.0])


def test_shared_subunit_uses_max_not_sum():
    """A receptor in multiple pairs gets max(pair_score), not sum."""
    n = 20
    genes = ["ITGA1", "ITGA2", "ITGB1"]

    # Pair 1: ITGA1-ITGB1, strongly concentrated
    c1 = np.zeros(n)
    c1[:2] = 20.0
    # Pair 2: ITGA2-ITGB1, weakly concentrated
    c2 = np.ones(n)
    c2[0] = 4.0

    comm, names, pgenes = _make_comm(
        {"ITGA1-ITGB1": c1, "ITGA2-ITGB1": c2}, genes,
    )
    scores, _ = compute_edge_scores(comm, names, pgenes, genes, ScoreConfig())

    # ITGB1 score should equal ITGA1 score (both from pair1, the stronger pair)
    assert scores[2] == scores[0]  # ITGB1 = max(pair1, pair2) = pair1
    assert scores[2] > scores[1]   # ITGB1 > ITGA2 (pair2 is weaker)


def test_heteromeric_pair_propagates_to_all_subunits():
    """Heteromeric ligand like INHBA_INHBB propagates score to all subunits."""
    n = 10
    pair_comm = np.zeros(n)
    pair_comm[0] = 20.0
    genes = ["INHBA", "INHBB", "ACVR1B"]

    comm, names, pgenes = _make_comm({"INHBA_INHBB-ACVR1B": pair_comm}, genes)
    scores, stats = compute_edge_scores(comm, names, pgenes, genes, ScoreConfig())

    assert "INHBA_INHBB-ACVR1B" in stats
    assert scores[0] > 0  # INHBA
    assert scores[1] > 0  # INHBB
    assert scores[2] > 0  # ACVR1B
    # All three genes get the same pair score
    assert scores[0] == scores[1] == scores[2]


def test_empty_comm_returns_zeros():
    """No pairs → zero scores."""
    comm = np.empty((10, 0), dtype=np.float64)
    scores, stats = compute_edge_scores(comm, [], {}, ["A"], ScoreConfig())
    np.testing.assert_array_equal(scores, [0.0])
    assert stats == {}


def test_zero_communication_pair_skipped():
    """A pair with all-zero communication is skipped gracefully."""
    n = 10
    pair_comm = np.zeros(n)
    comm, names, pgenes = _make_comm({"LIG-REC": pair_comm}, ["LIG", "REC"])
    scores, stats = compute_edge_scores(comm, names, pgenes, ["LIG", "REC"], ScoreConfig())

    np.testing.assert_array_equal(scores, [0.0, 0.0])
    assert "LIG-REC" not in stats  # skipped because mean <= 0



def test_mean_edge_aggregation_averages_only_spatially_concentrated_pairs():
    """Mean aggregation should average retained pair scores rather than sum them."""
    n = 20
    concentrated_a = np.zeros(n)
    concentrated_a[:2] = 20.0
    concentrated_b = np.zeros(n)
    concentrated_b[:4] = 10.0
    uniform = np.ones(n)

    comm, names, pgenes = _make_comm(
        {
            "G1-G2": concentrated_a,
            "G1-G3": concentrated_b,
            "G1-G4": uniform,
        },
        ["G1", "G2", "G3", "G4"],
    )

    scores, stats = compute_edge_scores(
        comm,
        names,
        pgenes,
        ["G1", "G2", "G3", "G4"],
        ScoreConfig(edge_agg_method="mean"),
    )

    expected_mean = (stats["G1-G2"]["pair_score"] + stats["G1-G3"]["pair_score"]) / 2
    assert np.isclose(scores[0], expected_mean)
    assert np.isclose(scores[1], stats["G1-G2"]["pair_score"])
    assert np.isclose(scores[2], stats["G1-G3"]["pair_score"])
    assert scores[3] == 0.0
