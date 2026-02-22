"""Obstacle analysis and navigation advice for visually impaired users."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Obstacle:
    """An analyzed obstacle with position and danger classification."""

    track_id: int
    class_name: str
    depth_m: float  # Distance to obstacle in meters
    lateral_position: str  # "left", "center", "right"
    lateral_angle: float  # Angle from center in degrees (-fov/2 to +fov/2)
    closing_speed: float  # m/s, positive = approaching
    danger_level: str  # "clear", "caution", "warning", "danger"
    confidence: float


class ObstacleAdvisor:
    """
    Analyzes tracked objects and provides navigation advice.

    Classifies obstacles by:
    - Lateral position (left, center, right)
    - Danger level based on distance and closing speed
    """

    # Priority classes that warrant more prominent warnings
    PRIORITY_CLASSES = {
        "person", "car", "bicycle", "motorcycle", "dog",
        "truck", "bus", "cat", "horse"
    }

    def __init__(
        self,
        fov_degrees: float = 75.0,
        center_corridor_degrees: float = 30.0,
        danger_m: float = 1.5,
        warning_m: float = 3.0,
        caution_m: float = 5.0,
        fast_approach_speed: float = 1.0
    ):
        """
        Initialize obstacle advisor.

        Args:
            fov_degrees: Horizontal field of view of camera.
            center_corridor_degrees: Width of center corridor (total, +/- half).
            danger_m: Distance threshold for danger level.
            warning_m: Distance threshold for warning level.
            caution_m: Distance threshold for caution level.
            fast_approach_speed: Closing speed (m/s) that escalates danger.
        """
        self.fov_degrees = fov_degrees
        self.center_half = center_corridor_degrees / 2.0
        self.danger_m = danger_m
        self.warning_m = warning_m
        self.caution_m = caution_m
        self.fast_approach_speed = fast_approach_speed

        logger.info(
            f"Obstacle advisor initialized (fov={fov_degrees}, "
            f"danger<{danger_m}m, warning<{warning_m}m, caution<{caution_m}m)"
        )

    def analyze_path(
        self,
        tracks: List,
        frame_width: int
    ) -> List[Obstacle]:
        """
        Analyze tracked objects and classify obstacles.

        Args:
            tracks: List of TrackedObject instances from object tracker.
            frame_width: Width of the camera frame in pixels.

        Returns:
            List of Obstacle instances sorted by danger level (most dangerous first).
        """
        obstacles = []

        for track in tracks:
            depth = track.get_current_depth()

            # Skip objects without valid depth
            if depth <= 0:
                continue

            # Calculate lateral angle from center
            x1, y1, x2, y2 = track.bbox
            bbox_center_x = (x1 + x2) / 2.0
            norm_x = bbox_center_x / frame_width  # 0 to 1
            lateral_angle = (norm_x - 0.5) * self.fov_degrees  # -fov/2 to +fov/2

            # Classify lateral position
            if abs(lateral_angle) <= self.center_half:
                lateral_position = "center"
            elif lateral_angle < 0:
                lateral_position = "left"
            else:
                lateral_position = "right"

            # Classify danger level
            danger_level = self._classify_danger(
                depth,
                lateral_position,
                track.closing_speed,
                track.class_name
            )

            obstacles.append(Obstacle(
                track_id=track.track_id,
                class_name=track.class_name,
                depth_m=depth,
                lateral_position=lateral_position,
                lateral_angle=lateral_angle,
                closing_speed=track.closing_speed,
                danger_level=danger_level,
                confidence=track.confidence
            ))

        # Sort by danger level priority, then by depth
        danger_order = {"danger": 0, "warning": 1, "caution": 2, "clear": 3}
        obstacles.sort(key=lambda o: (danger_order.get(o.danger_level, 4), o.depth_m))

        return obstacles

    def _classify_danger(
        self,
        depth: float,
        lateral_position: str,
        closing_speed: float,
        class_name: str
    ) -> str:
        """
        Classify danger level based on distance, position, and closing speed.

        Side obstacles are treated much more leniently - they're only relevant
        for awareness, not for "stop" commands.

        Args:
            depth: Distance to obstacle in meters.
            lateral_position: "left", "center", or "right".
            closing_speed: Closing speed in m/s (positive = approaching).
            class_name: Object class name.

        Returns:
            Danger level: "danger", "warning", "caution", or "clear".
        """
        # Side obstacles: NEVER danger/warning - user can dodge them
        # Only provide awareness (caution) for very close side objects
        if lateral_position != "center":
            if depth < 1.0:
                # Very close on side - just awareness, not a command
                return "caution"
            else:
                return "clear"

        # Center obstacles: full danger assessment
        is_fast_approaching = closing_speed > self.fast_approach_speed

        # Danger: very close OR fast approaching within warning range
        if depth < self.danger_m:
            return "danger"
        elif is_fast_approaching and depth < self.warning_m:
            return "danger"

        # Warning: within warning range, or fast approaching within caution
        if depth < self.warning_m:
            return "warning"
        elif is_fast_approaching and depth < self.caution_m:
            return "warning"

        # Caution: within caution range
        if depth < self.caution_m:
            return "caution"

        return "clear"

    def get_advice(
        self,
        obstacles: List[Obstacle],
        is_moving: bool
    ) -> Tuple[str, str]:
        """
        Generate navigation advice based on obstacles.

        Args:
            obstacles: List of Obstacle instances from analyze_path().
            is_moving: Whether the user is currently walking.

        Returns:
            Tuple of (advice_text, priority) where priority is
            "urgent", "high", "normal", or "low".
        """
        if not obstacles:
            if is_moving:
                return "path clear", "low"
            return "", "low"

        # Filter to center obstacles only for stop/dodge decisions
        center_obstacles = [o for o in obstacles if o.lateral_position == "center"]
        center_dangers = [o for o in center_obstacles if o.danger_level == "danger"]
        center_warnings = [o for o in center_obstacles if o.danger_level == "warning"]

        # URGENT: Danger in center - must stop or dodge
        if center_dangers:
            obs = center_dangers[0]
            direction = self._get_escape_direction(obs, obstacles)
            if direction:
                return f"stop {obs.class_name} ahead go {direction}", "urgent"
            return f"stop {obs.class_name} directly ahead", "urgent"

        # WARNING: Close obstacles in center
        if center_warnings:
            obs = center_warnings[0]
            dist_str = f"{obs.depth_m:.0f} meters" if obs.depth_m >= 2 else "close"
            direction = self._get_escape_direction(obs, obstacles)
            if direction:
                return f"{obs.class_name} {dist_str} go {direction}", "high"
            return f"{obs.class_name} {dist_str} ahead", "high"

        # Only provide additional info when moving
        if is_moving:
            # CAUTION: Side obstacles - just awareness, no direction command
            side_cautions = [
                o for o in obstacles
                if o.lateral_position != "center" and o.danger_level == "caution"
            ]
            if side_cautions:
                obs = side_cautions[0]
                return f"{obs.class_name} on your {obs.lateral_position}", "normal"

            # Center caution - something ahead but not urgent
            center_cautions = [o for o in center_obstacles if o.danger_level == "caution"]
            if center_cautions:
                obs = center_cautions[0]
                return f"{obs.class_name} ahead", "normal"

            return "path clear", "low"

        return "", "low"

    def _get_escape_direction(
        self,
        center_obstacle: Obstacle,
        all_obstacles: List[Obstacle]
    ) -> Optional[str]:
        """
        Determine which direction to dodge based on obstacle position and clearance.

        Logic:
        1. If obstacle is slightly left of center, prefer dodging right (and vice versa)
        2. Only recommend a direction if that side is actually clear enough
        3. If both sides blocked, return None (just say "stop")

        Args:
            center_obstacle: The center obstacle we need to avoid.
            all_obstacles: All analyzed obstacles.

        Returns:
            "left" or "right" if that direction is safe to dodge, None otherwise.
        """
        # Minimum clearance needed to recommend a direction (meters)
        min_clearance = 2.0

        # Find closest obstacle on each side
        left_min_depth = float('inf')
        right_min_depth = float('inf')

        for obs in all_obstacles:
            if obs.lateral_position == "left":
                left_min_depth = min(left_min_depth, obs.depth_m)
            elif obs.lateral_position == "right":
                right_min_depth = min(right_min_depth, obs.depth_m)

        # Check if each side is clear enough
        left_clear = left_min_depth >= min_clearance
        right_clear = right_min_depth >= min_clearance

        # If neither side is clear, don't recommend a direction
        if not left_clear and not right_clear:
            return None

        # If only one side is clear, recommend that side
        if left_clear and not right_clear:
            return "left"
        if right_clear and not left_clear:
            return "right"

        # Both sides are clear - use obstacle's position to decide
        # If obstacle is slightly left of center, go right (natural dodge)
        # If obstacle is slightly right of center, go left
        if center_obstacle.lateral_angle < -3:  # Obstacle left of center
            return "right"
        elif center_obstacle.lateral_angle > 3:  # Obstacle right of center
            return "left"

        # Obstacle is dead center - pick side with more clearance
        if right_min_depth > left_min_depth + 0.5:
            return "right"
        elif left_min_depth > right_min_depth + 0.5:
            return "left"

        # Both equally clear and obstacle dead center - no preference
        # Still recommend one direction rather than nothing
        return "right"  # Default to right if truly equal
