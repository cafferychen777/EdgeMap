import sys

from edgemap.cli import main


def test_cli_maps_all_args_to_pipeline_config(monkeypatch):
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgemap",
            "--st",
            "st.h5ad",
            "--gwas",
            "gwas.tsv",
            "--gwas-label",
            "LDL",
            "--output",
            "out_dir",
            "--resource-dir",
            "/tmp/resource",
            "--k-spatial",
            "9",
            "--dis-thr",
            "1234.5",
            "--n-blocks",
            "77",
            "--preprocessed",
        ],
    )
    monkeypatch.setattr("edgemap.cli.run", fake_run)

    main()
    cfg = captured["cfg"]

    assert cfg.st_h5ad == "st.h5ad"
    assert cfg.gwas_sumstats == "gwas.tsv"
    assert cfg.gwas_label == "LDL"
    assert cfg.output_dir == "out_dir"
    assert cfg.resource_dir == "/tmp/resource"
    assert cfg.spatial.k_spatial == 9
    assert cfg.spatial.dis_thr == 1234.5
    assert cfg.spatial.preprocessed is True
    assert cfg.regression.n_blocks == 77


def test_cli_uses_defaults_for_optional_args(monkeypatch):
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgemap",
            "--st",
            "st.h5ad",
            "--gwas",
            "gwas.tsv",
            "--gwas-label",
            "TraitX",
        ],
    )
    monkeypatch.setattr("edgemap.cli.run", fake_run)

    main()
    cfg = captured["cfg"]

    assert cfg.output_dir == "results"
    assert cfg.resource_dir is None
    assert cfg.spatial.k_spatial == 6
    assert cfg.spatial.dis_thr == 3000.0
    assert cfg.spatial.preprocessed is False
    assert cfg.regression.n_blocks == 200
