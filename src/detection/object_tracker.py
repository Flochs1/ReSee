"""IoU-based object tracking with depth history, ReID, and closing speed computation."""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

from .yolo_detector import Detection
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from .reid_model import ReIDEmbedder

logger = get_logger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with depth history and appearance embedding."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    depth_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: float = 0.0  # timestamp
    frames_tracked: int = 0
    closing_speed: float = 0.0  # m/s, positive = approaching
    embedding: Optional[np.ndarray] = None  # ReID appearance embedding

    def get_current_depth(self) -> float:
        """Get most recent depth value, or -1 if no valid depth."""
        if self.depth_history:
            return self.depth_history[-1][1]
        return -1.0


class ObjectTracker:
    """
    IoU + ReID object tracker with depth history tracking.

    Tracks objects across frames using Intersection over Union (IoU) matching
    combined with appearance-based ReID embeddings. Maintains depth history
    for each track and computes closing speed.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age_seconds: float = 5.0,  # Keep showing at last position for 5s
        depth_history_frames: int = 30,
        reid_model: Optional["ReIDEmbedder"] = None,
        reid_weight: float = 0.4,  # Give more weight to appearance
        reid_threshold: float = 0.4,  # Lower = easier to match by appearance
        graveyard_seconds: float = 60.0,  # Keep dead tracks 60s for resurrection
        resurrection_threshold: float = 0.40  # ReID similarity to resurrect (lower = easier match)
    ):
        """
        Initialize object tracker.

        Args:
            iou_threshold: Minimum IoU for matching detections to tracks.
            max_age_seconds: Seconds before moving tracks to graveyard.
            depth_history_frames: Number of depth samples to keep per track.
            reid_model: Optional ReID embedding model for appearance matching.
            reid_weight: Weight for ReID score in combined matching (0-1).
                         Combined score = (1-reid_weight)*IoU + reid_weight*cosine_sim
            reid_threshold: Minimum cosine similarity for ReID matching.
            graveyard_seconds: How long to keep dead tracks for potential resurrection.
            resurrection_threshold: ReID similarity needed to resurrect a dead track.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age_seconds
        self.depth_history_size = depth_history_frames
        self.reid_model = reid_model
        self.reid_weight = reid_weight
        self.reid_threshold = reid_threshold
        self.graveyard_seconds = graveyard_seconds
        self.resurrection_threshold = resurrection_threshold

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id: int = 0

        # Graveyard: recently deleted tracks that can be resurrected via ReID
        # Key: track_id, Value: (TrackedObject, death_timestamp)
        self._graveyard: Dict[int, Tuple[TrackedObject, float]] = {}

        reid_status = "enabled" if reid_model else "disabled"
        logger.info(
            f"Object tracker initialized (iou={iou_threshold}, "
            f"max_age={max_age_seconds}s, graveyard={graveyard_seconds}s, "
            f"reid={reid_status}, reid_weight={reid_weight})"
        )

    def update(
        self,
        detections: List[Detection],
        depth_map: Optional[np.ndarray],
        timestamp: float,
        frame: Optional[np.ndarray] = None
    ) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of current frame detections.
            depth_map: Depth map in meters (H x W float32), or None.
            timestamp: Current timestamp (monotonic).
            frame: BGR image for ReID embedding extraction (optional).

        Returns:
            List of active tracked objects.
        """
        # Extract embeddings for detections if ReID is enabled
        t_reid_start = time.perf_counter()
        det_embeddings: List[Optional[np.ndarray]] = []
        if self.reid_model is not None and frame is not None:
            for det in detections:
                emb = self.reid_model.extract(frame, det.bbox)
                det_embeddings.append(emb)
        else:
            det_embeddings = [None] * len(detections)
        t_reid = (time.perf_counter() - t_reid_start) * 1000

        # Match detections to existing tracks
        t_match_start = time.perf_counter()
        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            detections, det_embeddings
        )
        t_match = (time.perf_counter() - t_match_start) * 1000

        # Update matched tracks
        t_update_start = time.perf_counter()
        for det_idx, track_id in matched:
            det = detections[det_idx]
            track = self.tracks[track_id]

            track.bbox = det.bbox
            track.confidence = det.confidence
            track.last_seen = timestamp
            track.frames_tracked += 1

            # Update embedding with exponential moving average
            if det_embeddings[det_idx] is not None:
                if track.embedding is None:
                    track.embedding = det_embeddings[det_idx]
                else:
                    # EMA update: 0.7 old + 0.3 new for smooth appearance updates
                    track.embedding = 0.7 * track.embedding + 0.3 * det_embeddings[det_idx]
                    # Re-normalize
                    norm = np.linalg.norm(track.embedding)
                    if norm > 1e-6:
                        track.embedding = track.embedding / norm

            # Sample depth
            if depth_map is not None:
                depth = self._sample_depth(det.bbox, depth_map)
                if depth > 0:
                    track.depth_history.append((timestamp, depth))

            # Update closing speed
            track.closing_speed = self._compute_closing_speed(track)

        # Create new tracks for unmatched detections (or resurrect from graveyard)
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            det_emb = det_embeddings[det_idx]

            # Try to resurrect from graveyard using ReID
            resurrected_id = self._try_resurrect(det, det_emb, timestamp)

            if resurrected_id is not None:
                # Resurrection successful - track already added back to self.tracks
                continue

            # No resurrection match - create new track
            track = TrackedObject(
                track_id=self.next_id,
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                depth_history=deque(maxlen=self.depth_history_size),
                last_seen=timestamp,
                frames_tracked=1,
                embedding=det_emb
            )

            # Sample initial depth
            if depth_map is not None:
                depth = self._sample_depth(det.bbox, depth_map)
                if depth > 0:
                    track.depth_history.append((timestamp, depth))

            self.tracks[self.next_id] = track
            self.next_id += 1

        # Move stale tracks to graveyard (don't delete completely)
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if (timestamp - track.last_seen) > self.max_age
        ]
        for tid in stale_ids:
            track = self.tracks[tid]
            # Only graveyard if track has embedding for resurrection
            if track.embedding is not None:
                self._graveyard[tid] = (track, timestamp)
                logger.info(f"Track #{tid} ({track.class_name}) moved to graveyard")
            del self.tracks[tid]

        # Clean up old graveyard entries
        expired_graves = [
            tid for tid, (track, death_time) in self._graveyard.items()
            if (timestamp - death_time) > self.graveyard_seconds
        ]
        for tid in expired_graves:
            logger.info(f"Track #{tid} expired from graveyard")
            del self._graveyard[tid]

        # Merge active tracks with graveyard if they match (handles the case where
        # a new track was created but it's actually the same as a graveyard track)
        self._merge_with_graveyard(timestamp)

        t_update = (time.perf_counter() - t_update_start) * 1000
        t_total = t_reid + t_match + t_update

        # Log detailed timing breakdown for every frame with detections
        n_dets = len(detections)
        if n_dets > 0:
            reid_per_crop = t_reid / n_dets if n_dets > 0 else 0
            logger.info(
                f"Tracker: {n_dets} dets, {len(self.tracks)} tracks | "
                f"reid={t_reid:.1f}ms ({reid_per_crop:.1f}ms/crop), "
                f"match={t_match:.1f}ms, update={t_update:.1f}ms, "
                f"total={t_total:.1f}ms"
            )

        # Return active tracks
        return list(self.tracks.values())

    def _match_detections(
        self,
        detections: List[Detection],
        det_embeddings: Optional[List[Optional[np.ndarray]]] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU + ReID.

        When ReID is enabled, computes a combined score:
            combined = (1 - reid_weight) * IoU + reid_weight * cosine_similarity

        Returns:
            Tuple of (matched pairs, unmatched detection indices, unmatched track ids).
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())

        track_ids = list(self.tracks.keys())
        use_reid = (
            self.reid_model is not None and
            det_embeddings is not None and
            any(e is not None for e in det_embeddings)
        )

        # Compute score matrix (IoU or combined IoU + ReID)
        score_matrix = np.zeros((len(detections), len(track_ids)))
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        reid_matrix = np.zeros((len(detections), len(track_ids)))

        for d_idx, det in enumerate(detections):
            for t_idx, tid in enumerate(track_ids):
                track = self.tracks[tid]

                # Only match same class
                if det.class_id != track.class_id:
                    continue

                # Compute IoU
                iou = self._compute_iou(det.bbox, track.bbox)
                iou_matrix[d_idx, t_idx] = iou

                # Compute ReID similarity if available
                reid_sim = 0.0
                if use_reid and det_embeddings[d_idx] is not None and track.embedding is not None:
                    from .reid_model import ReIDEmbedder
                    reid_sim = ReIDEmbedder.cosine_similarity(
                        det_embeddings[d_idx], track.embedding
                    )
                    # Clamp to [0, 1] range (cosine can be negative)
                    reid_sim = max(0.0, reid_sim)
                reid_matrix[d_idx, t_idx] = reid_sim

                # Combined score
                if use_reid and track.embedding is not None and det_embeddings[d_idx] is not None:
                    score = (1 - self.reid_weight) * iou + self.reid_weight * reid_sim
                else:
                    score = iou

                score_matrix[d_idx, t_idx] = score

        # Greedy matching on combined score
        matched = []
        matched_dets = set()
        matched_tracks = set()

        while True:
            if score_matrix.size == 0:
                break

            max_score = np.max(score_matrix)

            # Find best match position
            d_idx, t_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
            iou_val = iou_matrix[d_idx, t_idx]
            reid_val = reid_matrix[d_idx, t_idx]

            # Check thresholds
            # Must pass IoU threshold OR have strong ReID match (no IoU required for ReID!)
            passes_iou = iou_val >= self.iou_threshold
            passes_reid = use_reid and reid_val >= self.reid_threshold
            # For very strong ReID matches, allow even with zero IoU (handles rotation/movement)
            passes_strong_reid = use_reid and reid_val >= 0.55

            if not (passes_iou or passes_reid or passes_strong_reid):
                break

            # If only ReID matched (no IoU), log it
            if passes_reid and not passes_iou:
                tid = track_ids[t_idx]
                logger.info(f"ReID-only match: det→track #{tid} (reid={reid_val:.2f}, iou={iou_val:.2f})")

            matched.append((d_idx, track_ids[t_idx]))
            matched_dets.add(d_idx)
            matched_tracks.add(track_ids[t_idx])

            # Remove matched row and column
            score_matrix[d_idx, :] = 0
            score_matrix[:, t_idx] = 0
            iou_matrix[d_idx, :] = 0
            iou_matrix[:, t_idx] = 0
            reid_matrix[d_idx, :] = 0
            reid_matrix[:, t_idx] = 0

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]

        return matched, unmatched_dets, unmatched_tracks

    def _try_resurrect(
        self,
        det: Detection,
        det_embedding: Optional[np.ndarray],
        timestamp: float
    ) -> Optional[int]:
        """
        Try to resurrect a track from the graveyard using ReID matching.

        This handles occlusion: when an object reappears after being hidden,
        we match it back to its original track ID instead of creating a new one.

        Args:
            det: The unmatched detection.
            det_embedding: Embedding for the detection (may be None).
            timestamp: Current timestamp.

        Returns:
            Track ID if resurrected, None otherwise.
        """
        if det_embedding is None or not self._graveyard:
            return None

        best_match_id: Optional[int] = None
        best_similarity = self.resurrection_threshold

        for tid, (dead_track, death_time) in self._graveyard.items():
            # Must be same class
            if dead_track.class_id != det.class_id:
                continue

            # Must have embedding
            if dead_track.embedding is None:
                continue

            # Compute cosine similarity
            similarity = float(np.dot(det_embedding, dead_track.embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = tid

        if best_match_id is not None:
            # Resurrect the track!
            dead_track, death_time = self._graveyard[best_match_id]

            # Update track with new detection info
            dead_track.bbox = det.bbox
            dead_track.confidence = det.confidence
            dead_track.last_seen = timestamp
            dead_track.frames_tracked += 1

            # Update embedding with EMA
            dead_track.embedding = 0.7 * dead_track.embedding + 0.3 * det_embedding
            norm = np.linalg.norm(dead_track.embedding)
            if norm > 1e-6:
                dead_track.embedding = dead_track.embedding / norm

            # Move back to active tracks
            self.tracks[best_match_id] = dead_track
            del self._graveyard[best_match_id]

            time_dead = timestamp - death_time
            logger.info(
                f"RESURRECTED track #{best_match_id} ({dead_track.class_name}) "
                f"after {time_dead:.1f}s (similarity={best_similarity:.3f})"
            )

            return best_match_id

        return None

    def _merge_with_graveyard(self, timestamp: float) -> None:
        """
        Check if any active tracks should be merged with graveyard tracks.

        This handles the case where:
        1. Object A is tracked as #1
        2. We look away, #1 goes to graveyard
        3. We see object A again from different angle, creates #2
        4. Now #1 (graveyard) and #2 (active) are the same object

        We merge by keeping the older (graveyard) ID and deleting the newer one.
        """
        if not self._graveyard:
            return

        merge_threshold = 0.5  # Similarity needed to merge

        # Find merges: active track → graveyard track
        merges: List[Tuple[int, int, float]] = []  # (active_id, grave_id, similarity)

        for active_id, active_track in self.tracks.items():
            if active_track.embedding is None:
                continue

            for grave_id, (grave_track, death_time) in self._graveyard.items():
                # Must be same class
                if active_track.class_id != grave_track.class_id:
                    continue

                if grave_track.embedding is None:
                    continue

                # Compute similarity
                similarity = float(np.dot(active_track.embedding, grave_track.embedding))

                if similarity >= merge_threshold:
                    merges.append((active_id, grave_id, similarity))

        # Apply merges (keep older graveyard ID, delete newer active ID)
        merged_active_ids: Set[int] = set()
        merged_grave_ids: Set[int] = set()
        for active_id, grave_id, similarity in merges:
            if active_id in merged_active_ids:
                continue  # Already merged
            if grave_id in merged_grave_ids:
                continue  # Graveyard entry already merged

            # Get the active track's current state
            active_track = self.tracks[active_id]
            grave_track, death_time = self._graveyard[grave_id]

            # Update graveyard track with active track's position
            grave_track.bbox = active_track.bbox
            grave_track.confidence = active_track.confidence
            grave_track.last_seen = timestamp
            grave_track.frames_tracked += active_track.frames_tracked

            # Merge embeddings
            grave_track.embedding = 0.5 * grave_track.embedding + 0.5 * active_track.embedding
            norm = np.linalg.norm(grave_track.embedding)
            if norm > 1e-6:
                grave_track.embedding = grave_track.embedding / norm

            # Move graveyard track back to active, delete the newer track
            self.tracks[grave_id] = grave_track
            del self.tracks[active_id]
            del self._graveyard[grave_id]

            merged_active_ids.add(active_id)
            merged_grave_ids.add(grave_id)

            logger.info(
                f"MERGED track #{active_id} into #{grave_id} ({grave_track.class_name}) "
                f"(similarity={similarity:.3f})"
            )

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
