"""
EdgeMap: Decompose trait heritability into node-intrinsic
and edge-interactive components via spatial cell-cell communication.
"""

__version__ = "0.1.0"

from .config import PipelineConfig, SpatialConfig, ScoreConfig, RegressionConfig
from .pipeline import run
