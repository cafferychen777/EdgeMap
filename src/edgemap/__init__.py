"""
EdgeMap: heritability mapping with spatially informed ligand-receptor
gene annotations.
"""

from importlib.metadata import version as _version

__version__ = _version("edgemap")

from .config import (
    PipelineConfig as PipelineConfig,
    RegressionConfig as RegressionConfig,
    ScoreConfig as ScoreConfig,
    SpatialConfig as SpatialConfig,
)
from .pipeline import run as run

__all__ = [
    "__version__",
    "PipelineConfig",
    "SpatialConfig",
    "ScoreConfig",
    "RegressionConfig",
    "run",
]
