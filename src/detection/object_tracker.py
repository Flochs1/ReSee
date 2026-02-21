"""IoU-based object tracking with depth history and closing speed computation."""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .yolo_detector import Detection
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with depth history."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    depth_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: float = 0.0  # timestamp
    frames_tracked: int = 0
    closing_speed: float = 0.0  # m/s, positive = approaching

    def get_current_depth(self) -> float:
        """Get most recent depth value, or -1 if no valid depth."""
        if self.depth_history:
            return self.depth_history[-1][1]
        return -1.0


class ObjectTracker:
    """
    IoU-based object tracker with depth history tracking.

    Tracks objects across frames using Intersection over Union (IoU) matching,
    maintains depth history for each track, and computes closing speed.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age_seconds: float = 1.0,
        depth_history_frames: int = 30
    ):
        """
        Initialize object tracker.

        Args:
            iou_threshold: Minimum IoU for matching detections to tracks.
            max_age_seconds: Seconds before dropping unseen tracks.
            depth_history_frames: Number of depth samples to keep per track.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age_seconds
        self.depth_history_size = depth_history_frames

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id: int = 0

        logger.info(
            f"Object tracker initialized (iou={iou_threshold}, "
            f"max_age={max_age_seconds}s, history={depth_history_frames})"
        )

    def update(
        self,
        detections: List[Detection],
        depth_map: Optional[np.ndarray],
        timestamp: float
    ) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of current frame detections.
            depth_map: Depth map in meters (H x W float32), or None.
            timestamp: Current timestamp (monotonic).

        Returns:
            List of active tracked objects.
        """
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)

        # Update matched tracks
        for det_idx, track_id in matched:
            det = detections[det_idx]
            track = self.tracks[track_id]

            track.bbox = det.bbox
            track.confidence = det.confidence
            track.last_seen = timestamp
            track.frames_tracked += 1

            # Sample depth
            if depth_map is not None:
                depth = self._sample_depth(det.bbox, depth_map)
                if depth > 0:
                    track.depth_history.append((timestamp, depth))

            # Update closing speed
            track.closing_speed = self._compute_closing_speed(track)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track = TrackedObject(
                track_id=self.next_id,
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                depth_history=deque(maxlen=self.depth_history_size),
                last_seen=timestamp,
                frames_tracked=1
            )

            # Sample initial depth
            if depth_map is not None:
                depth = self._sample_depth(det.bbox, depth_map)
                if depth > 0:
                    track.depth_history.append((timestamp, depth))

            self.tracks[self.next_id] = track
            self.next_id += 1

        # Remove stale tracks
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if (timestamp - track.last_seen) > self.max_age
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        # Return active tracks
        return list(self.tracks.values())

    def _match_detections(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.

        Returns:
            Tuple of (matched pairs, unmatched detection indices, unmatched track ids).
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())

        # Compute IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))

        for d_idx, det in enumerate(detections):
            for t_idx, tid in enumerate(track_ids):
                track = self.tracks[tid]
                # Only match same class
                if det.class_id == track.class_id:
                    iou_matrix[d_idx, t_idx] = self._compute_iou(det.bbox, track.bbox)

        # Greedy matching (could use Hungarian algorithm for optimal matching)
        matched = []
        matched_dets = set()
        matched_tracks = set()

        while True:
            # Find best remaining match
            if iou_matrix.size == 0:
                break

            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break

            d_idx, t_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

            matched.append((d_idx, track_ids[t_idx]))
            matched_dets.add(d_idx)
            matched_tracks.add(track_ids[t_idx])

            # Remove matched row and column
            iou_matrix[d_idx, :] = 0
            iou_matrix[:, t_idx] = 0

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]

        return matched, unmatched_dets, unmatched_tracks

    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _sample_depth(bbox: Tuple[int, int, int, int], depth_map: np.ndarray) -> float:
        """
        Sample depth from center region of bounding box.

        Args:
            bbox: Bounding box (x1, y1, x2, y2).
            depth_map: Depth map in meters.

        Returns:
            Median depth value, or -1.0 if invalid.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Sample center region (avoid edges with depth artifacts)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        margin_x = max(1, (x2 - x1) // 4)
        margin_y = max(1, (y2 - y1) // 4)

        region_y1 = max(0, cy - margin_y)
        region_y2 = min(h, cy + margin_y)
        region_x1 = max(0, cx - margin_x)
        region_x2 = min(w, cx + margin_x)

        region = depth_map[region_y1:region_y2, region_x1:region_x2]

        # Filter invalid depths
        valid = region[(region > 0.1) & (region < 20.0)]

        if len(valid) > 0:
            return float(np.median(valid))
        return -1.0

    @staticmethod
    def _compute_closing_speed(track: TrackedObject) -> float:
        """
        Compute closing speed from depth history.

        Positive = approaching, negative = receding.

        Args:
            track: Tracked object with depth history.

        Returns:
            Closing speed in m/s.
        """
        history = track.depth_history
        if len(history) < 5:  # Need at least ~0.5s of data
            return 0.0

        times = np.array([t for t, d in history])
        depths = np.array([d for t, d in history])

        # Filter valid depths
        valid = depths > 0
        if np.sum(valid) < 3:
            return 0.0

        times_valid = times[valid]
        depths_valid = depths[valid]

        # Need sufficient time span
        time_span = times_valid[-1] - times_valid[0]
        if time_span < 0.3:
            return 0.0

        # Linear regression for smooth speed estimate
        try:
            slope, _ = np.polyfit(times_valid, depths_valid, 1)
            # Negate: negative slope = depth decreasing = approaching = positive closing speed
            return -slope
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
