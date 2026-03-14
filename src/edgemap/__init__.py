"""
EdgeMap: Decompose trait heritability into node-intrinsic
and edge-interactive components via spatial cell-cell communication.
"""

from importlib.metadata import version as _version

__version__ = _version("edgemap")

from .config import PipelineConfig, SpatialConfig, ScoreConfig, RegressionConfig
from .pipeline import run
