import numpy as np
import pandas as pd

from edgemap.simulation import SCENARIOS, SimConfig, run_simulation, simulate_chisq


def test_simulate_chisq_null_has_reasonable_mean():
    rng = np.random.default_rng(123)
    n_snps = 20000
    baseline_ld = np.ones((n_snps, 2), dtype=np.float64)
    ell_node = np.zeros(n_snps, dtype=np.float64)
    ell_edge = np.zeros(n_snps, dtype=np.float64)
    baseline_tau = np.zeros(2, dtype=np.float64)

    chisq = simulate_chisq(
        baseline_ld=baseline_ld,
        ell_node=ell_node,
        ell_edge=ell_edge,
        N_bar=1000.0,
        M_total=1_000_000.0,
        tau_node=0.0,
        tau_edge=0.0,
        rng=rng,
        baseline_tau=baseline_tau,
    )

    assert chisq.shape == (n_snps,)
    assert np.all(chisq >= 0)
    assert abs(chisq.mean() - 1.0) < 0.08


def test_run_simulation_returns_expected_rows_and_columns():
    n_snps = 80
    snps = [f"rs{i}" for i in range(n_snps)]
    baseline = pd.DataFrame(
        {
            "SNP": snps,
            "base1": np.linspace(0.1, 1.0, n_snps),
            "base2": np.linspace(1.0, 0.1, n_snps),
        }
    )
    annot_ld = pd.DataFrame(
        {
            "SNP": snps,
            "ell_node": np.linspace(0.2, 0.8, n_snps),
            "ell_edge": np.linspace(0.8, 0.2, n_snps),
        }
    )
    w_ld = pd.DataFrame({"SNP": snps, "L2": np.ones(n_snps)})

    out = run_simulation(
        baseline=baseline,
        annot_ld=annot_ld,
        w_ld=w_ld,
        M_total=1_000_000.0,
        N_bar=5000.0,
        cfg=SimConfig(tau_node=0.0, tau_edge=0.0, n_reps=4, seed=7),
        n_blocks=10,
    )

    assert len(out) == 8  # 4 reps * 2 annotations
    assert set(out["annotation"]) == {"node", "edge"}
    assert {"rep", "annotation", "tau", "se", "z", "p_onesided", "p_twosided"} <= set(out.columns)
    assert np.all(np.isfinite(out["tau"]))
    assert np.all(np.isfinite(out["se"]))
    assert np.all((out["p_onesided"] >= 0) & (out["p_onesided"] <= 1))
    assert np.all((out["p_twosided"] >= 0) & (out["p_twosided"] <= 1))


def test_scenarios_include_required_architectures():
    assert {"null", "node_only", "edge_only", "mixed"} <= set(SCENARIOS.keys())
    assert SCENARIOS["null"].tau_node == 0.0
    assert SCENARIOS["null"].tau_edge == 0.0
    assert SCENARIOS["node_only"].tau_node > 0
    assert SCENARIOS["edge_only"].tau_edge > 0
