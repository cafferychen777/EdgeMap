"""
EdgeMap: Decompose trait heritability into node-intrinsic
and edge-interactive components via spatial cell-cell communication.
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
