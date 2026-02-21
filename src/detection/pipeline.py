"""Detection pipeline combining detector, tracker, and visualization."""

import time
import numpy as np
from typing import List, Optional, Tuple

from .object_tracker import ObjectTracker, TrackedObject
from .visualization import draw_tracks
from .detection_config import get_detection_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPipeline:
    """Combined detection + tracking pipeline."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.detector = None
        self.tracker: Optional[ObjectTracker] = None
        self.tracks: List[TrackedObject] = []
        self.frame_count = 0

        if not enabled:
            logger.info("Detection pipeline disabled")
            return

        settings = get_detection_settings()

        if not settings.detection.enabled:
            logger.info("Detection disabled in config")
            self.enabled = False
            return

        try:
            # Initialize detector
            if settings.detection.use_coreml:
                from .coreml_detector import CoreMLDetector
                self.detector = CoreMLDetector(
                    model_path=settings.detection.model_path,
                    confidence_threshold=settings.detection.confidence_threshold
                )
                logger.info("Using CoreML detector (no PyTorch)")
            else:
                from .yolo_detector import YOLODetector
                self.detector = YOLODetector(
                    model_path=settings.detection.model_path,
                    confidence_threshold=settings.detection.confidence_threshold
                )
                logger.info("Using YOLO detector (requires PyTorch)")

            # Initialize tracker
            self.tracker = ObjectTracker(
                iou_threshold=settings.tracking.iou_threshold,
                max_age_seconds=settings.tracking.max_age_seconds,
                depth_history_frames=settings.tracking.depth_history_frames
            )

            logger.info("Detection pipeline ready")

        except Exception as e:
            logger.error(f"Detection init failed: {e}")
            self.enabled = False

    def process(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, List[TrackedObject]]:
        """Run detection and tracking on frame."""
        if not self.enabled or self.detector is None or self.tracker is None:
            return frame, []

        if timestamp is None:
            timestamp = time.monotonic()

        self.frame_count += 1

        # Detect objects
        detections = self.detector.detect(frame)

        # Update tracker
        self.tracks = self.tracker.update(detections, depth_map, timestamp)

        # Draw overlays
        frame = draw_tracks(frame, self.tracks)

        return frame, self.tracks

    def get_tracks(self) -> List[TrackedObject]:
        return self.tracks

    def is_enabled(self) -> bool:
        return self.enabled and self.detector is not None
