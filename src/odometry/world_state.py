"""World state management for visual odometry."""

import math
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Deque

from .camera_pose import CameraPose
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WorldObject:
    """
    An object positioned in world coordinates.

    Converted from camera-relative TrackedObject using current camera pose.
    """
    track_id: int
    class_name: str
    confidence: float
    world_x: float  # World X position (meters)
    world_y: float  # World Y position (meters)
    depth: float    # Distance from camera (meters)
    closing_speed: float = 0.0


class WorldState:
    """
    Manages camera pose and world coordinate transformations.

    Maintains:
    - Current camera pose (position + heading)
    - Pose history for trajectory visualization
    - Transforms between camera-relative and world coordinates
    """

    def __init__(self, max_history: int = 500):
        """
        Initialize world state.

        Args:
            max_history: Maximum number of poses to keep in trajectory history.
        """
        self.camera_pose = CameraPose()
        self.pose_history: Deque[CameraPose] = deque(maxlen=max_history)
        self.max_history = max_history

        # Add initial pose to history
        self.pose_history.append(self.camera_pose.copy())

        logger.info(f"World state initialized (history={max_history})")

    def update_pose(
        self,
        R: Optional[np.ndarray],
        t: Optional[np.ndarray],
        vo_position: tuple,
        vo_heading: float,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Update camera pose from visual odometry.

        Args:
            R: Rotation matrix (may be None if VO failed).
            t: Translation vector (may be None if VO failed).
            vo_position: (x, y) position from VO.
            vo_heading: Heading from VO in radians.
            timestamp: Optional timestamp, uses current time if not provided.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # Update pose from VO estimates
        self.camera_pose.x = vo_position[0]
        self.camera_pose.y = vo_position[1]
        self.camera_pose.heading = vo_heading
        self.camera_pose.timestamp = timestamp

        # Add to history (only if we've moved significantly or rotated)
        if self._should_add_to_history():
            self.pose_history.append(self.camera_pose.copy())

    def _should_add_to_history(self) -> bool:
        """Check if current pose differs enough from last history entry."""
        if len(self.pose_history) == 0:
            return True

        last = self.pose_history[-1]
        dist = math.sqrt(
            (self.camera_pose.x - last.x) ** 2 +
            (self.camera_pose.y - last.y) ** 2
        )
        angle_diff = abs(self.camera_pose.heading - last.heading)

        # Add if moved > 0.1m or rotated > 5 degrees
        return dist > 0.1 or angle_diff > math.radians(5)

    def transform_tracks_to_world(
        self,
        tracks: list,
        frame_width: int,
        fov_rad: float
    ) -> List[WorldObject]:
        """
        Transform camera-relative tracked objects to world coordinates.

        Args:
            tracks: List of TrackedObject instances.
            frame_width: Width of the camera frame in pixels.
            fov_rad: Horizontal field of view in radians.

        Returns:
            List of WorldObject instances in world coordinates.
        """
        world_objects = []

        for track in tracks:
            depth = track.get_current_depth()

            if depth <= 0:
                continue

            # Get center X of bounding box
            x1, y1, x2, y2 = track.bbox
            bbox_center_x = (x1 + x2) / 2

            # Calculate angle from center of frame
            norm_x = bbox_center_x / frame_width
            angle = (norm_x - 0.5) * fov_rad  # -fov/2 to +fov/2

            # Camera-relative position (camera faces +Y in its local frame)
            cam_x = depth * math.sin(angle)
            cam_y = depth * math.cos(angle)

            # Transform to world coordinates
            world_x, world_y = self._camera_to_world(cam_x, cam_y)

            world_objects.append(WorldObject(
                track_id=track.track_id,
                class_name=track.class_name,
                confidence=track.confidence,
                world_x=world_x,
                world_y=world_y,
                depth=depth,
                closing_speed=track.closing_speed
            ))

        return world_objects

    def _camera_to_world(self, cam_x: float, cam_y: float) -> tuple:
        """
        Transform a point from camera-relative to world coordinates.

        Args:
            cam_x: X position relative to camera (positive = right).
            cam_y: Y position relative to camera (positive = forward).

        Returns:
            (world_x, world_y) position in world coordinates.
        """
        # Rotate by camera heading
        cos_h = math.cos(self.camera_pose.heading)
        sin_h = math.sin(self.camera_pose.heading)

        # Apply rotation (heading = 0 means camera faces +Y world)
        world_x = self.camera_pose.x + cam_x * cos_h + cam_y * sin_h
        world_y = self.camera_pose.y - cam_x * sin_h + cam_y * cos_h

        return world_x, world_y

    def reset(self) -> None:
        """Reset world state to origin."""
        self.camera_pose = CameraPose()
        self.pose_history.clear()
        self.pose_history.append(self.camera_pose.copy())
        logger.info("World state reset to origin")

    def get_trajectory_points(self) -> List[tuple]:
        """
        Get trajectory points for visualization.

        Returns:
            List of (x, y) tuples representing the camera's path.
        """
        return [(p.x, p.y) for p in self.pose_history]
