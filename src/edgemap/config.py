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
    if not isinstance(n_cells, (int, np.integer)) or isinstance(n_cells, bool):
        raise TypeError("n_cells must be an integer")
    if n_cells <= 0:
        raise ValueError("n_cells must be > 0")
    if requested is not None:
        if not isinstance(requested, (int, np.integer)) or isinstance(requested, bool):
            raise TypeError("gene_chunk_size must be an integer")
        if requested <= 0:
            raise ValueError("gene_chunk_size must be > 0")
        return int(requested)

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
    min_lr_cell_pct: float = 0.05
    preprocessed: bool = False
    input_scale: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.k_spatial, (int, np.integer)) or isinstance(
            self.k_spatial, bool
        ):
            raise TypeError("k_spatial must be an integer")
        if self.k_spatial < 1:
            raise ValueError("k_spatial must be >= 1")
        if not np.isfinite(self.dis_thr) or self.dis_thr <= 0:
            raise ValueError("dis_thr must be finite and > 0")
        if not isinstance(
            self.min_cells_per_gene, (int, np.integer)
        ) or isinstance(self.min_cells_per_gene, bool):
            raise TypeError("min_cells_per_gene must be an integer")
        if self.min_cells_per_gene < 1:
            raise ValueError("min_cells_per_gene must be >= 1")
        if not np.isfinite(self.min_lr_cell_pct) or not (
            0 <= self.min_lr_cell_pct <= 1
        ):
            raise ValueError("min_lr_cell_pct must be finite and in [0, 1]")
        if self.input_scale not in (None, "raw_counts", "log1p"):
            raise ValueError("input_scale must be 'raw_counts' or 'log1p'")
        if self.preprocessed and self.input_scale == "raw_counts":
            raise ValueError(
                "preprocessed=True conflicts with input_scale='raw_counts'"
            )

    @property
    def resolved_input_scale(self) -> str:
        """Return the explicit expression scale, preserving the legacy flag."""
        if self.input_scale is not None:
            return self.input_scale
        return "log1p" if self.preprocessed else "raw_counts"


@dataclass
class ScoreConfig:
    """Parameters for node/edge score computation."""
    edge_agg_percentile: float = 95.0
    node_agg_percentile: float = 100.0  # 100.0 = max (default); <100 for sensitivity
    kernel_bandwidth_frac: float = 1.0 / 3.0
    gene_chunk_size: int | None = None
    edge_agg_method: str = "max"  # "max" or "mean"

    def __post_init__(self) -> None:
        if self.edge_agg_method not in ("max", "mean"):
            raise ValueError(f"edge_agg_method must be 'max' or 'mean', got '{self.edge_agg_method}'")
        if not np.isfinite(self.edge_agg_percentile) or not (
            0 < self.edge_agg_percentile <= 100
        ):
            raise ValueError(f"edge_agg_percentile must be in (0, 100], got {self.edge_agg_percentile}")
        if not np.isfinite(self.node_agg_percentile) or not (
            0 < self.node_agg_percentile <= 100
        ):
            raise ValueError(f"node_agg_percentile must be in (0, 100], got {self.node_agg_percentile}")
        if not np.isfinite(self.kernel_bandwidth_frac) or self.kernel_bandwidth_frac <= 0:
            raise ValueError("kernel_bandwidth_frac must be finite and > 0")


@dataclass
class RegressionConfig:
    """Parameters for S-LDSC regression."""
    n_blocks: int = 200
    chisq_max_factor: float = 0.001
    chisq_max_floor: float = 80.0

    def __post_init__(self) -> None:
        if not isinstance(self.n_blocks, (int, np.integer)) or isinstance(
            self.n_blocks, bool
        ):
            raise TypeError("n_blocks must be an integer")
        if self.n_blocks < 2:
            raise ValueError("n_blocks must be >= 2")
        if not np.isfinite(self.chisq_max_factor) or self.chisq_max_factor <= 0:
            raise ValueError("chisq_max_factor must be finite and > 0")
        if not np.isfinite(self.chisq_max_floor) or self.chisq_max_floor <= 0:
            raise ValueError("chisq_max_floor must be finite and > 0")


@dataclass
class PipelineConfig:
    """Top-level config binding everything together."""
    st_h5ad: str = ""
    gwas_sumstats: str = ""
    gwas_label: str = ""
    output_dir: str = "results"
    resource_dir: str | None = None
    run_context_ranking: bool = False

    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
