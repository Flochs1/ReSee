"""Detection pipeline combining detector, tracker, ReID, and visualization."""

import time
import numpy as np
from typing import List, Optional, Tuple

from .object_tracker import ObjectTracker, TrackedObject
from .reid_model import ReIDEmbedder
from .visualization import draw_tracks
from .detection_config import get_detection_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPipeline:
    """Combined detection + tracking + ReID pipeline."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.detector = None
        self.tracker: Optional[ObjectTracker] = None
        self.reid_model: Optional[ReIDEmbedder] = None
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

            # Initialize ReID model if enabled
            reid_cfg = settings.tracking.reid
            if reid_cfg.enabled:
                self.reid_model = ReIDEmbedder(
                    embedding_dim=reid_cfg.embedding_dim,
                    use_coreml=reid_cfg.model_path is not None,
                    model_path=reid_cfg.model_path
                )
                logger.info("ReID enabled for appearance-based tracking")

            # Initialize tracker with ReID
            self.tracker = ObjectTracker(
                iou_threshold=settings.tracking.iou_threshold,
                max_age_seconds=settings.tracking.max_age_seconds,
                depth_history_frames=settings.tracking.depth_history_frames,
                reid_model=self.reid_model,
                reid_weight=reid_cfg.weight if reid_cfg.enabled else 0.0,
                reid_threshold=reid_cfg.threshold if reid_cfg.enabled else 0.5
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
        """Run detection, ReID, and tracking on frame."""
        if not self.enabled or self.detector is None or self.tracker is None:
            return frame, []

        if timestamp is None:
            timestamp = time.monotonic()

        self.frame_count += 1
        t_start = time.perf_counter()

        # Detect objects
        t_detect_start = time.perf_counter()
        detections = self.detector.detect(frame)
        t_detect = (time.perf_counter() - t_detect_start) * 1000

        # Update tracker with frame for ReID embedding extraction
        t_track_start = time.perf_counter()
        self.tracks = self.tracker.update(detections, depth_map, timestamp, frame=frame)
        t_track = (time.perf_counter() - t_track_start) * 1000

        # Draw overlays
        t_draw_start = time.perf_counter()
        frame = draw_tracks(frame, self.tracks)
        t_draw = (time.perf_counter() - t_draw_start) * 1000

        t_total = (time.perf_counter() - t_start) * 1000

        # Log detailed timing for every frame
        logger.info(
            f"Pipeline frame {self.frame_count}: "
            f"detect={t_detect:.1f}ms, track+reid={t_track:.1f}ms, "
            f"draw={t_draw:.1f}ms, total={t_total:.1f}ms, "
            f"dets={len(detections)}, tracks={len(self.tracks)}"
        )

        return frame, self.tracks

    def get_tracks(self) -> List[TrackedObject]:
        return self.tracks

    def is_enabled(self) -> bool:
        return self.enabled and self.detector is not None
