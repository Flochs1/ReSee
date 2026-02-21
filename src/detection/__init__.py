"""Object detection and tracking module."""

from .object_tracker import ObjectTracker, TrackedObject
from .visualization import draw_tracks
from .pipeline import DetectionPipeline
from .detection_config import get_detection_settings

__all__ = [
    'ObjectTracker',
    'TrackedObject',
    'draw_tracks',
    'DetectionPipeline',
    'get_detection_settings',
]
