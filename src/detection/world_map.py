"""World map tracking with ego-motion estimation using stationary anchors and ReID constellations."""

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from .object_tracker import TrackedObject
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_stationary_classes() -> Set[str]:
    """Load stationary class names from config."""
    config_path = Path("config/coco_classes.yaml")
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using fallback")
        return {'potted plant', 'fire hydrant', 'stop sign', 'bench', 'traffic light'}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        stationary = {
            name for name, props in config.get('classes', {}).items()
            if props.get('stationary', False)
        }
        logger.info(f"Loaded {len(stationary)} stationary classes from config")
        return stationary
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {'potted plant', 'fire hydrant', 'stop sign'}


STATIONARY_CLASSES: Set[str] = load_stationary_classes()


@dataclass
class WorldObject:
    """An object with world-fixed coordinates."""
    track_id: int
    class_name: str
    world_x: float          # meters, world coordinates (right is positive)
    world_y: float          # meters, world coordinates (forward is positive)
    depth: float            # current depth from camera
    is_anchor: bool         # stationary object used for ego-motion estimation
    last_seen: float        # timestamp
    confidence: float


@dataclass
class AnchorData:
    """Stored data for a known anchor point."""
    track_id: int
    class_name: str         # store class name for persistence
    world_x: float          # fixed world X position
    world_y: float          # fixed world Y position
    initial_depth: float    # depth when first seen
    last_seen: float        # last time this anchor was detected
    embedding: Optional[np.ndarray] = None  # ReID embedding for relocalization
    last_camera_x: float = 0.0  # camera position when last seen
    last_camera_y: float = 0.0  # camera position when last seen


@dataclass
class CameraState:
    """Camera position and orientation in world coordinates."""
    x: float = 0.0          # world position (starts at origin)
    y: float = 0.0          # world position
    heading: float = 0.0    # radians, 0 = initial forward direction


@dataclass
class AnchorObservation:
    """Current observation of an anchor."""
    track_id: int
    angle: float            # angle in frame (radians)
    depth: float            # observed depth (meters)
    world_x: float          # known world X
    world_y: float          # known world Y


@dataclass
class LandmarkHistory:
    """Historical observations of a ReID'd landmark (non-stationary object)."""
    track_id: int
    class_name: str
    # List of (timestamp, world_x, world_y, depth, embedding_norm) tuples
    observations: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    last_seen: float = 0.0

    def add_observation(
        self,
        timestamp: float,
        world_x: float,
        world_y: float,
        depth: float,
        embedding: Optional[np.ndarray] = None
    ) -> None:
        """Add a new observation."""
        emb_norm = float(np.linalg.norm(embedding)) if embedding is not None else 0.0
        self.observations.append((timestamp, world_x, world_y, depth, emb_norm))
        self.last_seen = timestamp
        # Keep last 30 observations
        if len(self.observations) > 30:
            self.observations = self.observations[-30:]

    def get_average_position(self, max_age: float = 2.0, current_time: float = 0.0) -> Optional[Tuple[float, float]]:
        """Get average position from recent observations."""
        recent = [
            (wx, wy) for t, wx, wy, d, e in self.observations
            if current_time - t < max_age
        ]
        if len(recent) < 2:
            return None
        avg_x = sum(wx for wx, wy in recent) / len(recent)
        avg_y = sum(wy for wx, wy in recent) / len(recent)
        return avg_x, avg_y


class WorldMap:
    """
    Maintains world-fixed positions for tracked objects and estimates camera ego-motion.

    Uses stationary objects (anchors) to estimate how the camera has moved.
    The algorithm:
    1. Observe anchors at (angle, depth) in current frame
    2. Compare to expected (angle, depth) given anchor's known world position
    3. Solve for camera (x, y, heading) that minimizes anchor drift
    4. Non-anchor objects are positioned using corrected camera state
    """

    def __init__(
        self,
        fov_degrees: float = 75.0,
        anchor_persistence_seconds: float = 60.0,  # Time-based expiration (for stationary)
        anchor_max_distance: float = 20.0,  # Distance-based expiration in meters
        min_anchors_for_estimation: int = 1,
        use_reid_ransac: bool = True,
        min_landmarks_for_ransac: int = 3,
        ransac_iterations: int = 50,
        ransac_inlier_threshold: float = 0.5,  # meters
        relocalization_threshold: float = 0.35,  # ReID similarity for anchor matching (lower = easier)
    ):
        """
        Initialize world map.

        Args:
            fov_degrees: Horizontal field of view of the camera.
            anchor_persistence_seconds: How long to keep anchors after last detection.
            min_anchors_for_estimation: Minimum anchors needed for ego-motion estimation.
            use_reid_ransac: Enable RANSAC-based pose estimation from ReID'd landmarks.
            min_landmarks_for_ransac: Minimum landmarks needed for RANSAC.
            ransac_iterations: Number of RANSAC iterations.
            ransac_inlier_threshold: Distance threshold for RANSAC inliers (meters).
        """
        self.fov_rad = math.radians(fov_degrees)
        self.anchor_persistence_seconds = anchor_persistence_seconds
        self.anchor_max_distance = anchor_max_distance
        self.min_anchors_for_estimation = min_anchors_for_estimation

        # RANSAC settings
        self.use_reid_ransac = use_reid_ransac
        self.min_landmarks_for_ransac = min_landmarks_for_ransac
        self.ransac_iterations = ransac_iterations
        self.ransac_inlier_threshold = ransac_inlier_threshold

        # Relocalization settings
        self.relocalization_threshold = relocalization_threshold

        # World state
        self.camera = CameraState()

        # Known anchors with their FIXED world positions (keyed by track_id)
        self._anchors: Dict[int, AnchorData] = {}

        # Landmark history for ReID-based RANSAC (keyed by track_id)
        self._landmarks: Dict[int, LandmarkHistory] = {}

        # For debugging/visualization
        self._last_ego_motion = (0.0, 0.0, 0.0)  # (dx, dy, dheading)
        self._last_ransac_inliers = 0

        logger.info(
            f"WorldMap initialized (fov={fov_degrees}, "
            f"anchor_persistence={anchor_persistence_seconds}s/{anchor_max_distance}m, "
            f"stationary_classes={len(STATIONARY_CLASSES)}, "
            f"reid_ransac={'enabled' if use_reid_ransac else 'disabled'}, "
            f"relocalization_threshold={relocalization_threshold})"
        )

    def _find_matching_anchor(
        self,
        track: "TrackedObject",
        seen_track_ids: Set[int]
    ) -> Optional[int]:
        """
        Find an existing anchor that matches this track by ReID embedding.

        This enables relocalization - when you return to a scene after looking away,
        new tracks can be matched to previously seen anchors by appearance.

        Args:
            track: The tracked object to match.
            seen_track_ids: Track IDs already matched this frame (to avoid duplicates).

        Returns:
            Matching anchor's track_id, or None if no match found.
        """
        if track.embedding is None:
            logger.debug(f"Relocalization skip: track #{track.track_id} has no embedding")
            return None

        best_match_id: Optional[int] = None
        best_similarity = self.relocalization_threshold
        candidates = []

        for anchor_id, anchor in self._anchors.items():
            # Skip if already matched this frame
            if anchor_id in seen_track_ids:
                continue

            # Must be same class
            if anchor.class_name != track.class_name:
                continue

            # Must have embedding
            if anchor.embedding is None:
                continue

            # Compute cosine similarity
            similarity = float(np.dot(track.embedding, anchor.embedding))
            candidates.append((anchor_id, similarity))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = anchor_id

        # Log all candidates for debugging
        if candidates:
            candidates.sort(key=lambda x: -x[1])  # Sort by similarity descending
            top_candidates = candidates[:3]  # Show top 3
            logger.debug(
                f"Relocalization candidates for track #{track.track_id} ({track.class_name}): "
                f"{[(aid, f'{sim:.3f}') for aid, sim in top_candidates]}, "
                f"threshold={self.relocalization_threshold}"
            )

        if best_match_id is not None:
            logger.info(
                f"RELOCALIZATION: matched track #{track.track_id} ({track.class_name}) "
                f"to anchor #{best_match_id} (similarity={best_similarity:.3f})"
            )
        elif candidates:
            # No match but had candidates - log why
            logger.debug(
                f"Relocalization FAILED for track #{track.track_id} ({track.class_name}): "
                f"best similarity {candidates[0][1]:.3f} < threshold {self.relocalization_threshold}"
            )

        return best_match_id

    def _get_angle_in_frame(self, bbox: tuple, frame_width: int) -> float:
        """Calculate the angle of an object from camera center based on bbox position."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        norm_x = center_x / frame_width  # 0 to 1
        return (norm_x - 0.5) * self.fov_rad  # -fov/2 to +fov/2

    def _expected_observation(
        self,
        world_x: float,
        world_y: float,
        cam_x: float,
        cam_y: float,
        cam_heading: float
    ) -> Tuple[float, float]:
        """
        Calculate expected (angle, depth) for a world point from camera position.

        Returns:
            (angle, depth) - angle in radians from camera center, depth in meters
        """
        rel_x = world_x - cam_x
        rel_y = world_y - cam_y
        depth = math.sqrt(rel_x * rel_x + rel_y * rel_y)

        if depth < 0.01:
            return 0.0, 0.01

        # World angle of object relative to world forward (Y+)
        world_angle = math.atan2(rel_x, rel_y)

        # Angle in camera frame = world_angle - camera_heading
        angle_in_frame = world_angle - cam_heading

        # Normalize to -pi to pi
        while angle_in_frame > math.pi:
            angle_in_frame -= 2 * math.pi
        while angle_in_frame < -math.pi:
            angle_in_frame += 2 * math.pi

        return angle_in_frame, depth

    def _estimate_ego_motion(
        self,
        observations: List[AnchorObservation]
    ) -> Tuple[float, float, float]:
        """
        Estimate camera motion (dx, dy, dheading) from anchor observations.

        Uses iterative least squares to find camera position that minimizes
        the error between observed and expected anchor positions.

        Args:
            observations: List of anchor observations with known world positions.

        Returns:
            (dx, dy, dheading) - estimated camera motion since last frame
        """
        if len(observations) < self.min_anchors_for_estimation:
            return 0.0, 0.0, 0.0

        # Current camera state as starting point
        cam_x = self.camera.x
        cam_y = self.camera.y
        cam_heading = self.camera.heading

        # Iterative refinement (simple gradient descent)
        learning_rate = 0.5
        iterations = 10

        for _ in range(iterations):
            # Compute gradients
            grad_x = 0.0
            grad_y = 0.0
            grad_heading = 0.0

            for obs in observations:
                # Expected observation given current camera estimate
                exp_angle, exp_depth = self._expected_observation(
                    obs.world_x, obs.world_y, cam_x, cam_y, cam_heading
                )

                # Errors
                angle_error = obs.angle - exp_angle
                depth_error = obs.depth - exp_depth

                # Normalize angle error
                while angle_error > math.pi:
                    angle_error -= 2 * math.pi
                while angle_error < -math.pi:
                    angle_error += 2 * math.pi

                # Approximate gradients
                # For heading: if we see object more to the right than expected,
                # we've rotated left (heading decreased)
                grad_heading -= angle_error * 2.0

                # For position: move toward observed position
                rel_x = obs.world_x - cam_x
                rel_y = obs.world_y - cam_y
                dist = math.sqrt(rel_x * rel_x + rel_y * rel_y)
                if dist > 0.01:
                    # If depth_error > 0, object is further than expected, camera moved forward
                    grad_x -= (rel_x / dist) * depth_error
                    grad_y -= (rel_y / dist) * depth_error

            # Normalize by number of observations
            n = len(observations)
            grad_x /= n
            grad_y /= n
            grad_heading /= n

            # Update camera estimate
            cam_x += learning_rate * grad_x
            cam_y += learning_rate * grad_y
            cam_heading += learning_rate * grad_heading

            # Reduce learning rate
            learning_rate *= 0.8

        # Return delta from original position
        dx = cam_x - self.camera.x
        dy = cam_y - self.camera.y
        dheading = cam_heading - self.camera.heading

        # Clamp to reasonable values (prevent jumps)
        max_translation = 0.5  # max 0.5m per frame
        max_rotation = math.radians(30)  # max 30 degrees per frame (allow faster correction)

        dx = max(-max_translation, min(max_translation, dx))
        dy = max(-max_translation, min(max_translation, dy))
        dheading = max(-max_rotation, min(max_rotation, dheading))

        return dx, dy, dheading

    def _compute_direct_heading(
        self,
        observations: List[AnchorObservation]
    ) -> Optional[float]:
        """
        Directly compute the camera heading from recognized anchors.

        For each anchor, we know:
        - Its world position (world_x, world_y)
        - Our camera position (cam_x, cam_y)
        - The angle we observe it at in the frame

        From this we can directly compute:
        heading = atan2(world_x - cam_x, world_y - cam_y) - observed_angle

        Returns:
            Estimated absolute heading, or None if not enough data.
        """
        if len(observations) < 1:
            return None

        heading_estimates = []

        for obs in observations:
            # Vector from camera to anchor in world coordinates
            rel_x = obs.world_x - self.camera.x
            rel_y = obs.world_y - self.camera.y

            # World angle of the anchor from camera position
            world_angle = math.atan2(rel_x, rel_y)

            # If we observe the anchor at angle obs.angle in the frame,
            # then our heading = world_angle - obs.angle
            estimated_heading = world_angle - obs.angle

            # Normalize to -pi to pi
            while estimated_heading > math.pi:
                estimated_heading -= 2 * math.pi
            while estimated_heading < -math.pi:
                estimated_heading += 2 * math.pi

            heading_estimates.append(estimated_heading)

        if not heading_estimates:
            return None

        # Use median for robustness (handles outliers)
        heading_estimates.sort()
        median_heading = heading_estimates[len(heading_estimates) // 2]

        return median_heading

    def _estimate_pose_ransac(
        self,
        tracks: List[TrackedObject],
        frame_width: int,
        timestamp: float
    ) -> Tuple[float, float, float, int]:
        """
        Estimate camera pose using RANSAC on ReID'd object constellation.

        Uses 3+ re-identified objects with historical positions to find a
        rigid transform (dx, dy, dheading) that best explains the observations.

        Args:
            tracks: Current tracked objects with embeddings.
            frame_width: Width of the camera frame.
            timestamp: Current timestamp.

        Returns:
            (dx, dy, dheading, num_inliers) - estimated motion and inlier count.
        """
        # Collect landmarks with both current observation and history
        correspondences: List[Tuple[int, float, float, float, float]] = []
        # (track_id, observed_world_x, observed_world_y, expected_world_x, expected_world_y)

        for track in tracks:
            depth = track.get_current_depth()
            if depth <= 0 or track.embedding is None:
                continue

            # Skip stationary objects (already handled by anchor system)
            if track.class_name in STATIONARY_CLASSES:
                continue

            # Calculate current observed world position
            angle_in_frame = self._get_angle_in_frame(track.bbox, frame_width)
            world_angle = self.camera.heading + angle_in_frame
            obs_world_x = self.camera.x + depth * math.sin(world_angle)
            obs_world_y = self.camera.y + depth * math.cos(world_angle)

            # Check if we have history for this track
            if track.track_id in self._landmarks:
                landmark = self._landmarks[track.track_id]
                expected_pos = landmark.get_average_position(
                    max_age=2.0, current_time=timestamp
                )
                if expected_pos is not None:
                    exp_x, exp_y = expected_pos
                    correspondences.append((
                        track.track_id,
                        obs_world_x, obs_world_y,
                        exp_x, exp_y
                    ))

        # Need at least 3 correspondences for RANSAC
        if len(correspondences) < self.min_landmarks_for_ransac:
            return 0.0, 0.0, 0.0, 0

        # RANSAC to find best rigid transform
        best_dx, best_dy, best_dheading = 0.0, 0.0, 0.0
        best_inliers = 0

        for _ in range(self.ransac_iterations):
            # Sample 2 correspondences to estimate transform
            if len(correspondences) < 2:
                break

            sample = random.sample(correspondences, 2)

            # Estimate rigid transform from sample
            dx, dy, dheading = self._estimate_rigid_transform(sample)

            # Count inliers
            inliers = 0
            for _, obs_x, obs_y, exp_x, exp_y in correspondences:
                # Apply inverse transform to observed position
                # (what would observed position be if camera hadn't moved?)
                cos_h = math.cos(-dheading)
                sin_h = math.sin(-dheading)
                corrected_x = (obs_x - dx) * cos_h - (obs_y - dy) * sin_h
                corrected_y = (obs_x - dx) * sin_h + (obs_y - dy) * cos_h

                # Check distance to expected position
                dist = math.sqrt((corrected_x - exp_x)**2 + (corrected_y - exp_y)**2)
                if dist < self.ransac_inlier_threshold:
                    inliers += 1

            if inliers > best_inliers:
                best_inliers = inliers
                best_dx, best_dy, best_dheading = dx, dy, dheading

        # Refine estimate using all inliers
        if best_inliers >= self.min_landmarks_for_ransac:
            inlier_correspondences = []
            for corr in correspondences:
                _, obs_x, obs_y, exp_x, exp_y = corr
                cos_h = math.cos(-best_dheading)
                sin_h = math.sin(-best_dheading)
                corrected_x = (obs_x - best_dx) * cos_h - (obs_y - best_dy) * sin_h
                corrected_y = (obs_x - best_dx) * sin_h + (obs_y - best_dy) * cos_h
                dist = math.sqrt((corrected_x - exp_x)**2 + (corrected_y - exp_y)**2)
                if dist < self.ransac_inlier_threshold:
                    inlier_correspondences.append(corr)

            if len(inlier_correspondences) >= 2:
                best_dx, best_dy, best_dheading = self._estimate_rigid_transform(
                    inlier_correspondences
                )

        # Clamp to reasonable values
        max_translation = 0.3
        max_rotation = math.radians(20)  # Allow larger corrections
        best_dx = max(-max_translation, min(max_translation, best_dx))
        best_dy = max(-max_translation, min(max_translation, best_dy))
        best_dheading = max(-max_rotation, min(max_rotation, best_dheading))

        return best_dx, best_dy, best_dheading, best_inliers

    def _estimate_rigid_transform(
        self,
        correspondences: List[Tuple[int, float, float, float, float]]
    ) -> Tuple[float, float, float]:
        """
        Estimate rigid transform (dx, dy, dheading) from correspondences.

        Uses least squares to find the transform that maps expected positions
        to observed positions.

        Args:
            correspondences: List of (track_id, obs_x, obs_y, exp_x, exp_y).

        Returns:
            (dx, dy, dheading) - estimated camera motion.
        """
        if len(correspondences) < 2:
            return 0.0, 0.0, 0.0

        # Extract points
        obs_points = np.array([(c[1], c[2]) for c in correspondences])
        exp_points = np.array([(c[3], c[4]) for c in correspondences])

        # Compute centroids
        obs_centroid = np.mean(obs_points, axis=0)
        exp_centroid = np.mean(exp_points, axis=0)

        # Center the points
        obs_centered = obs_points - obs_centroid
        exp_centered = exp_points - exp_centroid

        # Estimate rotation using cross-covariance
        # H = sum(exp_i * obs_i^T)
        H = exp_centered.T @ obs_centered

        # SVD to get rotation
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Extract angle
        dheading = math.atan2(R[1, 0], R[0, 0])

        # Compute translation
        # obs_centroid = R @ exp_centroid + t
        # t = obs_centroid - R @ exp_centroid
        rotated_exp = R @ exp_centroid
        dx = obs_centroid[0] - rotated_exp[0]
        dy = obs_centroid[1] - rotated_exp[1]

        return dx, dy, dheading

    def _update_landmarks(
        self,
        tracks: List[TrackedObject],
        frame_width: int,
        timestamp: float
    ) -> None:
        """
        Update landmark history with current observations.

        Args:
            tracks: Current tracked objects.
            frame_width: Width of the camera frame.
            timestamp: Current timestamp.
        """
        for track in tracks:
            depth = track.get_current_depth()
            if depth <= 0:
                continue

            # Skip stationary objects (handled separately as anchors)
            if track.class_name in STATIONARY_CLASSES:
                continue

            # Calculate world position
            angle_in_frame = self._get_angle_in_frame(track.bbox, frame_width)
            world_angle = self.camera.heading + angle_in_frame
            world_x = self.camera.x + depth * math.sin(world_angle)
            world_y = self.camera.y + depth * math.cos(world_angle)

            # Update or create landmark
            if track.track_id not in self._landmarks:
                self._landmarks[track.track_id] = LandmarkHistory(
                    track_id=track.track_id,
                    class_name=track.class_name
                )

            self._landmarks[track.track_id].add_observation(
                timestamp, world_x, world_y, depth, track.embedding
            )

        # Clean up old landmarks
        stale_ids = [
            tid for tid, lm in self._landmarks.items()
            if timestamp - lm.last_seen > self.anchor_persistence_seconds
        ]
        for tid in stale_ids:
            del self._landmarks[tid]

    def update(
        self,
        tracks: List[TrackedObject],
        frame_width: int,
        timestamp: Optional[float] = None
    ) -> List[WorldObject]:
        """
        Update world map with new tracked objects.

        Args:
            tracks: Current tracked objects from ObjectTracker.
            frame_width: Width of the camera frame in pixels.
            timestamp: Current timestamp (uses time.monotonic() if None).

        Returns:
            List of world objects to render.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # === PHASE 1: Collect anchor observations (with relocalization) ===
        anchor_observations: List[AnchorObservation] = []
        phase1_matched_ids: Set[int] = set()

        for track in tracks:
            depth = track.get_current_depth()
            if depth <= 0:
                continue

            is_anchor_class = track.class_name in STATIONARY_CLASSES
            if not is_anchor_class:
                continue

            # Check if this is a known anchor (by track ID or ReID)
            matched_anchor_id: Optional[int] = None

            if track.track_id in self._anchors:
                matched_anchor_id = track.track_id
            else:
                # Try relocalization by ReID
                matched_anchor_id = self._find_matching_anchor(track, phase1_matched_ids)

            if matched_anchor_id is not None:
                anchor = self._anchors[matched_anchor_id]
                angle = self._get_angle_in_frame(track.bbox, frame_width)
                phase1_matched_ids.add(matched_anchor_id)

                anchor_observations.append(AnchorObservation(
                    track_id=matched_anchor_id,
                    angle=angle,
                    depth=depth,
                    world_x=anchor.world_x,
                    world_y=anchor.world_y
                ))

        # === PHASE 2: Estimate ego-motion from anchor drift ===
        dx, dy, dheading = 0.0, 0.0, 0.0

        if anchor_observations:
            dx, dy, dheading = self._estimate_ego_motion(anchor_observations)

            # === PHASE 2a: Direct heading correction from recognized anchors ===
            # When we see anchors, directly compute what our heading should be
            direct_heading = self._compute_direct_heading(anchor_observations)
            if direct_heading is not None:
                heading_error = direct_heading - self.camera.heading

                # Normalize to -pi to pi
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi

                # If we have multiple anchors, trust the direct computation more
                if len(anchor_observations) >= 2:
                    # Strong correction - directly use computed heading
                    correction_strength = 0.7
                else:
                    # Single anchor - be more conservative
                    correction_strength = 0.4

                # Apply direct heading correction (overrides gradient descent dheading)
                dheading = heading_error * correction_strength

                if abs(heading_error) > math.radians(5):
                    logger.info(
                        f"Heading correction: error={math.degrees(heading_error):.1f}°, "
                        f"correcting by {math.degrees(dheading):.1f}° "
                        f"({len(anchor_observations)} anchors)"
                    )

        # === PHASE 2b: RANSAC refinement using ReID'd landmarks ===
        if self.use_reid_ransac:
            ransac_dx, ransac_dy, ransac_dheading, num_inliers = self._estimate_pose_ransac(
                tracks, frame_width, timestamp
            )
            self._last_ransac_inliers = num_inliers

            if num_inliers >= self.min_landmarks_for_ransac:
                # Blend anchor estimate with RANSAC estimate
                # Weight by number of observations
                anchor_weight = len(anchor_observations)
                ransac_weight = num_inliers * 0.5  # ReID less trusted than stationary anchors

                total_weight = anchor_weight + ransac_weight
                if total_weight > 0:
                    blend_anchor = anchor_weight / total_weight
                    blend_ransac = ransac_weight / total_weight

                    dx = blend_anchor * dx + blend_ransac * ransac_dx
                    dy = blend_anchor * dy + blend_ransac * ransac_dy
                    dheading = blend_anchor * dheading + blend_ransac * ransac_dheading

                    if num_inliers >= 3:
                        logger.debug(
                            f"RANSAC correction: {num_inliers} inliers, "
                            f"dx={ransac_dx:.3f}m, dy={ransac_dy:.3f}m, "
                            f"dheading={math.degrees(ransac_dheading):.2f} deg"
                        )

        self._last_ego_motion = (dx, dy, dheading)

        # Update camera state
        self.camera.x += dx
        self.camera.y += dy
        self.camera.heading += dheading

        if abs(dx) > 0.01 or abs(dy) > 0.01 or abs(dheading) > 0.001:
            logger.debug(
                f"Ego-motion: dx={dx:.3f}m, dy={dy:.3f}m, "
                f"dheading={math.degrees(dheading):.2f} deg, "
                f"anchors={len(anchor_observations)}, "
                f"ransac_inliers={self._last_ransac_inliers}"
            )

        # === PHASE 3: Process all tracks with corrected camera state ===
        world_objects: List[WorldObject] = []
        seen_anchor_ids: Set[int] = set()  # Track which anchors we've matched this frame

        for track in tracks:
            depth = track.get_current_depth()
            if depth <= 0:
                continue

            is_anchor_class = track.class_name in STATIONARY_CLASSES
            angle_in_frame = self._get_angle_in_frame(track.bbox, frame_width)

            if is_anchor_class:
                # ANCHOR: Use stored world position, or create new anchor
                matched_anchor_id: Optional[int] = None

                # First check if this track ID is already known
                if track.track_id in self._anchors:
                    matched_anchor_id = track.track_id
                else:
                    # Try relocalization: match by ReID embedding to existing anchors
                    matched_anchor_id = self._find_matching_anchor(track, seen_anchor_ids)

                # Also try spatial proximity match as fallback
                if matched_anchor_id is None:
                    # Calculate where this object would be in world coordinates
                    world_angle = self.camera.heading + angle_in_frame
                    tentative_x = self.camera.x + depth * math.sin(world_angle)
                    tentative_y = self.camera.y + depth * math.cos(world_angle)

                    # Find closest anchor of same class within 1.5m
                    closest_dist = 1.5  # meters
                    for anchor_id, anchor in self._anchors.items():
                        if anchor_id in seen_anchor_ids:
                            continue
                        if anchor.class_name != track.class_name:
                            continue

                        dist = math.sqrt(
                            (anchor.world_x - tentative_x) ** 2 +
                            (anchor.world_y - tentative_y) ** 2
                        )
                        if dist < closest_dist:
                            closest_dist = dist
                            matched_anchor_id = anchor_id
                            logger.info(
                                f"SPATIAL MATCH: track #{track.track_id} ({track.class_name}) "
                                f"matched to anchor #{anchor_id} (dist={dist:.2f}m)"
                            )

                if matched_anchor_id is not None:
                    # Known anchor (by track ID, ReID match, or spatial) - use its FIXED world position
                    anchor = self._anchors[matched_anchor_id]
                    anchor.last_seen = timestamp
                    anchor.last_camera_x = self.camera.x
                    anchor.last_camera_y = self.camera.y
                    seen_anchor_ids.add(matched_anchor_id)

                    # Update embedding with EMA for better future matching
                    if track.embedding is not None:
                        if anchor.embedding is None:
                            anchor.embedding = track.embedding.copy()
                        else:
                            anchor.embedding = 0.8 * anchor.embedding + 0.2 * track.embedding
                            norm = np.linalg.norm(anchor.embedding)
                            if norm > 1e-6:
                                anchor.embedding = anchor.embedding / norm

                    world_objects.append(WorldObject(
                        track_id=matched_anchor_id,
                        class_name=track.class_name,
                        world_x=anchor.world_x,
                        world_y=anchor.world_y,
                        depth=depth,
                        is_anchor=True,
                        last_seen=timestamp,
                        confidence=track.confidence
                    ))
                else:
                    # New anchor - calculate and store its world position
                    world_angle = self.camera.heading + angle_in_frame
                    world_x = self.camera.x + depth * math.sin(world_angle)
                    world_y = self.camera.y + depth * math.cos(world_angle)

                    self._anchors[track.track_id] = AnchorData(
                        track_id=track.track_id,
                        class_name=track.class_name,
                        world_x=world_x,
                        world_y=world_y,
                        initial_depth=depth,
                        last_seen=timestamp,
                        embedding=track.embedding.copy() if track.embedding is not None else None,
                        last_camera_x=self.camera.x,
                        last_camera_y=self.camera.y
                    )
                    seen_anchor_ids.add(track.track_id)

                    logger.info(
                        f"New anchor #{track.track_id} ({track.class_name}) at "
                        f"world ({world_x:.2f}, {world_y:.2f}), depth={depth:.2f}m"
                    )

                    world_objects.append(WorldObject(
                        track_id=track.track_id,
                        class_name=track.class_name,
                        world_x=world_x,
                        world_y=world_y,
                        depth=depth,
                        is_anchor=True,
                        last_seen=timestamp,
                        confidence=track.confidence
                    ))
            else:
                # NON-ANCHOR: Calculate world position using current camera state
                world_angle = self.camera.heading + angle_in_frame
                world_x = self.camera.x + depth * math.sin(world_angle)
                world_y = self.camera.y + depth * math.cos(world_angle)

                world_objects.append(WorldObject(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    world_x=world_x,
                    world_y=world_y,
                    depth=depth,
                    is_anchor=False,
                    last_seen=timestamp,
                    confidence=track.confidence
                ))

        # === PHASE 4: Add persisted anchors that weren't seen this frame ===
        seen_track_ids = {t.track_id for t in tracks}
        for tid, anchor in list(self._anchors.items()):
            if tid not in seen_track_ids:
                # Anchor not currently detected - check if still valid
                age = timestamp - anchor.last_seen

                # Distance from where we last saw this anchor
                dist_from_last = math.sqrt(
                    (self.camera.x - anchor.last_camera_x) ** 2 +
                    (self.camera.y - anchor.last_camera_y) ** 2
                )

                # Keep if within time AND distance limits
                if age <= self.anchor_persistence_seconds and dist_from_last <= self.anchor_max_distance:
                    # Still valid, add to output
                    world_objects.append(WorldObject(
                        track_id=tid,
                        class_name=anchor.class_name,
                        world_x=anchor.world_x,
                        world_y=anchor.world_y,
                        depth=anchor.initial_depth,
                        is_anchor=True,
                        last_seen=anchor.last_seen,
                        confidence=0.5  # Reduced confidence for unseen
                    ))
                else:
                    # Too old or too far, remove
                    reason = f"age={age:.1f}s" if age > self.anchor_persistence_seconds else f"dist={dist_from_last:.1f}m"
                    logger.info(f"Removing stale anchor #{tid} ({anchor.class_name}): {reason}")
                    del self._anchors[tid]

        # === PHASE 5: Update landmark history for RANSAC ===
        if self.use_reid_ransac:
            self._update_landmarks(tracks, frame_width, timestamp)

        return world_objects

    def get_camera_state(self) -> CameraState:
        """Get current camera state."""
        return self.camera

    def get_last_ego_motion(self) -> Tuple[float, float, float]:
        """Get last estimated ego-motion (dx, dy, dheading)."""
        return self._last_ego_motion

    def get_anchor_count(self) -> int:
        """Get number of known anchors."""
        return len(self._anchors)

    def get_landmark_count(self) -> int:
        """Get number of tracked ReID landmarks."""
        return len(self._landmarks)

    def get_last_ransac_inliers(self) -> int:
        """Get number of inliers from last RANSAC estimation."""
        return self._last_ransac_inliers

    def reset(self) -> None:
        """Reset world map to initial state."""
        self.camera = CameraState()
        self._anchors.clear()
        self._landmarks.clear()
        self._last_ego_motion = (0.0, 0.0, 0.0)
        self._last_ransac_inliers = 0
        logger.info("WorldMap reset")
