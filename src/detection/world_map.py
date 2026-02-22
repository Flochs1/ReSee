"""World map tracking with heading estimation using stationary anchors."""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .object_tracker import TrackedObject
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Only potted plants are stationary anchors for now
STATIONARY_CLASSES: Set[str] = {'potted plant'}


@dataclass
class WorldObject:
    """An object with world-fixed coordinates."""
    track_id: int
    class_name: str
    world_x: float          # meters, world coordinates (right is positive)
    world_y: float          # meters, world coordinates (forward is positive)
    depth: float            # current depth from camera
    is_anchor: bool         # stationary object used for heading estimation
    last_seen: float        # timestamp
    confidence: float


@dataclass
class AnchorData:
    """Stored data for a known anchor point."""
    track_id: int
    world_x: float          # fixed world X position
    world_y: float          # fixed world Y position
    initial_angle: float    # angle in frame when first seen
    initial_depth: float    # depth when first seen
    last_seen: float        # last time this anchor was detected


@dataclass
class CameraState:
    """Camera position and orientation in world coordinates."""
    x: float = 0.0          # world position (starts at origin)
    y: float = 0.0          # world position
    heading: float = 0.0    # radians, 0 = initial forward direction


class WorldMap:
    """
    Maintains world-fixed positions for tracked objects and estimates camera heading.

    Uses stationary objects (anchors) to estimate how the camera has rotated.
    Non-anchor objects are only shown while currently detected (no persistence).
    """

    def __init__(
        self,
        fov_degrees: float = 75.0,
        anchor_persistence_seconds: float = 5.0,
        heading_smoothing: float = 0.1,
    ):
        """
        Initialize world map.

        Args:
            fov_degrees: Horizontal field of view of the camera.
            anchor_persistence_seconds: How long to keep anchors after last detection.
            heading_smoothing: Smoothing factor for heading updates (0-1).
        """
        self.fov_rad = math.radians(fov_degrees)
        self.anchor_persistence_seconds = anchor_persistence_seconds
        self.heading_smoothing = heading_smoothing

        # World state
        self.camera = CameraState()

        # Known anchors with their FIXED world positions (keyed by track_id)
        self._anchors: Dict[int, AnchorData] = {}

        logger.info(
            f"WorldMap initialized (fov={fov_degrees}°, "
            f"anchor_persistence={anchor_persistence_seconds}s)"
        )

    def _get_angle_in_frame(self, bbox: tuple, frame_width: int) -> float:
        """Calculate the angle of an object from camera center based on bbox position."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        norm_x = center_x / frame_width  # 0 to 1
        return (norm_x - 0.5) * self.fov_rad  # -fov/2 to +fov/2

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

        world_objects: List[WorldObject] = []

        # === PHASE 1: Estimate heading change using known anchors ===
        # Temporarily disabled - just set heading to 0 for debugging
        # TODO: Re-enable once basic rendering is working

        # === PHASE 2: Process current tracks ===
        for track in tracks:
            depth = track.get_current_depth()
            if depth <= 0:
                continue

            is_anchor_class = track.class_name in STATIONARY_CLASSES
            angle_in_frame = self._get_angle_in_frame(track.bbox, frame_width)

            if is_anchor_class:
                # ANCHOR: Use stored world position, or create new anchor
                if track.track_id in self._anchors:
                    # Known anchor - use its FIXED world position
                    anchor = self._anchors[track.track_id]
                    anchor.last_seen = timestamp

                    world_objects.append(WorldObject(
                        track_id=track.track_id,
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
                        world_x=world_x,
                        world_y=world_y,
                        initial_angle=angle_in_frame,
                        initial_depth=depth,
                        last_seen=timestamp
                    )

                    logger.info(
                        f"New anchor #{track.track_id} ({track.class_name}) at "
                        f"world ({world_x:.2f}, {world_y:.2f}), depth={depth:.2f}m, "
                        f"angle={math.degrees(angle_in_frame):.1f}°"
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
                # NON-ANCHOR: Calculate world position, no persistence
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

        # === PHASE 3: Add persisted anchors that weren't seen this frame ===
        seen_track_ids = {t.track_id for t in tracks}
        for tid, anchor in list(self._anchors.items()):
            if tid not in seen_track_ids:
                # Anchor not currently detected - check if still valid
                age = timestamp - anchor.last_seen
                if age <= self.anchor_persistence_seconds:
                    # Still valid, add to output
                    world_objects.append(WorldObject(
                        track_id=tid,
                        class_name='potted plant',  # We know it's an anchor class
                        world_x=anchor.world_x,
                        world_y=anchor.world_y,
                        depth=anchor.initial_depth,  # Use last known depth
                        is_anchor=True,
                        last_seen=anchor.last_seen,
                        confidence=0.5  # Reduced confidence for unseen
                    ))
                else:
                    # Too old, remove
                    logger.info(f"Removing stale anchor #{tid}")
                    del self._anchors[tid]

        return world_objects

    def get_camera_state(self) -> CameraState:
        """Get current camera state."""
        return self.camera

    def reset(self) -> None:
        """Reset world map to initial state."""
        self.camera = CameraState()
        self._anchors.clear()
        logger.info("WorldMap reset")
