"""Stereo depth estimation module for ReSee.

Supports two backends:
- HITNet (TensorFlow + Metal GPU) - Primary
- StereoSGBM (OpenCV CPU) - Fallback
"""

from .processor import (
    DepthProcessor,
    DepthResult,
    check_backends_available,
    get_best_available_backend,
)
from .colormap import apply_colormap, COLORMAPS

__all__ = [
    "DepthProcessor",
    "DepthResult",
    "apply_colormap",
    "COLORMAPS",
    "check_backends_available",
    "get_best_available_backend",
]
