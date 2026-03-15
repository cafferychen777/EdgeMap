"""
Configuration: resource resolution, default parameters, and data contracts.

Two categories of resources:
  1. LR database — small, bundled with the package.
  2. gsMap resources (baseline LD, weights, SNP-gene matrix) — large,
     user-provided via parameter, env var, or filesystem auto-detection.
"""

import os
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np


# ── Resource resolution ─────────────────────────────────────────────


def resolve_resource_dir(override: str | Path | None = None) -> Path:
    """Resolve gsMap resource directory.

    Resolution order:
        1. Explicit override parameter
        2. EDGEMAP_RESOURCE_DIR environment variable
        3. Known filesystem locations
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Resource directory not found: {p}")
        return p

    env = os.environ.get("EDGEMAP_RESOURCE_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # Auto-detect: look relative to the installed package location
    candidate = Path(__file__).resolve().parents[2] / "data" / "gsMap_resource"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "gsMap resource directory not found. "
        "Set EDGEMAP_RESOURCE_DIR or pass resource_dir to PipelineConfig."
    )


def get_lr_database() -> Path:
    """Path to the bundled LIANA Consensus LR database."""
    return files("edgemap").joinpath("data", "liana_consensus.csv")


def resolve_gene_chunk_size(n_cells: int, requested: int | None) -> int:
    """Resolve node-score gene chunk size.

    If the user provides an explicit chunk size, use it directly.
    Otherwise, choose the largest chunk whose four main float32 work buffers
    stay within roughly 256 MiB:

        x_chunk, ranks, local, contrib

    This keeps memory bounded on large datasets without changing results.
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be > 0")
    if requested is not None:
        if requested <= 0:
            raise ValueError("gene_chunk_size must be > 0")
        return requested

    target_bytes = 256 * 1024 * 1024
    bytes_per_gene = 4 * np.dtype(np.float32).itemsize * n_cells
    auto_chunk = max(target_bytes // bytes_per_gene, 1)
    return int(np.clip(auto_chunk, 16, 2000))


# ── Parameter dataclasses ───────────────────────────────────────────


@dataclass
class SpatialConfig:
    """Parameters for spatial graph and communication inference."""
    k_spatial: int = 6
    dis_thr: float = 3000.0
    min_cells_per_gene: int = 10
    preprocessed: bool = False


@dataclass
class ScoreConfig:
    """Parameters for node/edge score computation."""
    edge_agg_percentile: float = 95.0
    kernel_bandwidth_frac: float = 1.0 / 3.0
    gene_chunk_size: int | None = None


@dataclass
class RegressionConfig:
    """Parameters for S-LDSC regression."""
    n_blocks: int = 200
    chisq_max_factor: float = 0.001
    chisq_max_floor: float = 80.0

    def __post_init__(self) -> None:
        if self.n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")


@dataclass
class PipelineConfig:
    """Top-level config binding everything together."""
    st_h5ad: str = ""
    gwas_sumstats: str = ""
    gwas_label: str = ""
    output_dir: str = "results"
    resource_dir: str | None = None

    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
