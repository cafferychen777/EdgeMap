from pathlib import Path

import pytest

from edgemap.config import (
    ScoreConfig,
    get_lr_database,
    resolve_gene_chunk_size,
    resolve_resource_dir,
)


def test_resolve_resource_dir_uses_override(tmp_path):
    out = resolve_resource_dir(tmp_path)
    assert out == tmp_path


def test_resolve_resource_dir_missing_override_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Resource directory not found"):
        resolve_resource_dir(tmp_path / "missing")


def test_resolve_resource_dir_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EDGEMAP_RESOURCE_DIR", str(tmp_path))
    out = resolve_resource_dir(None)
    assert out == tmp_path


def test_resolve_resource_dir_no_sources_raises(monkeypatch):
    monkeypatch.delenv("EDGEMAP_RESOURCE_DIR", raising=False)
    original_exists = Path.exists

    def fake_exists(self):
        # Keep unrelated path checks unchanged.
        if "gsMap_resource" not in str(self):
            return original_exists(self)
        return False

    monkeypatch.setattr(Path, "exists", fake_exists)
    with pytest.raises(FileNotFoundError, match="gsMap resource directory not found"):
        resolve_resource_dir(None)


def test_resolve_resource_dir_falls_back_to_known_candidate(monkeypatch):
    monkeypatch.delenv("EDGEMAP_RESOURCE_DIR", raising=False)
    original_exists = Path.exists

    def fake_exists(self):
        if "gsMap_resource" not in str(self):
            return original_exists(self)
        return str(self).endswith("data/gsMap_resource")

    monkeypatch.setattr(Path, "exists", fake_exists)
    out = resolve_resource_dir(None)
    assert str(out).endswith("data/gsMap_resource")


def test_get_lr_database_points_to_existing_csv():
    path = get_lr_database()
    assert str(path).endswith("liana_consensus.csv")
    assert Path(path).exists()


def test_resolve_gene_chunk_size_uses_explicit_value():
    assert resolve_gene_chunk_size(10_000, 321) == 321


def test_resolve_gene_chunk_size_auto_scales_and_clamps():
    assert resolve_gene_chunk_size(5_000, None) == 2000
    assert resolve_gene_chunk_size(20_000, None) == 838
    assert resolve_gene_chunk_size(1_000_000, None) == 16


def test_resolve_gene_chunk_size_rejects_invalid_values():
    with pytest.raises(ValueError, match="n_cells must be > 0"):
        resolve_gene_chunk_size(0, None)
    with pytest.raises(ValueError, match="gene_chunk_size must be > 0"):
        resolve_gene_chunk_size(10, 0)


def test_score_config_rejects_invalid_edge_agg_method():
    with pytest.raises(ValueError, match="edge_agg_method must be 'max' or 'mean'"):
        ScoreConfig(edge_agg_method="bad")
