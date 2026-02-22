"""Route visualization on bird's eye view with smooth flowing curves."""

import math
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from .route_types import PlannedRoute
from src.detection.object_tracker import TrackedObject
from .tts_output import TTSOutput

# Classes that are typically floor/ground and shouldn't trigger avoidance
FLOOR_CLASSES = {'floor', 'ground', 'carpet', 'rug', 'mat'}

# Classes that are typically walls/fixed structures (avoid only if directly ahead)
WALL_CLASSES = {'wall', 'door', 'window', 'pillar', 'column'}

# DANGEROUS: Roads should be avoided at all costs
ROAD_CLASSES = {'road', 'street', 'pavement', 'asphalt', 'highway', 'lane', 'crosswalk'}


class NavigationVoice:
    """
    Speaks navigation directions based on curve heading.
    """

    def __init__(self, interval: float = 2.0):
        """
        Args:
            interval: Minimum seconds between voice commands
        """
        self.interval = interval
        self.tts = TTSOutput(voice="Samantha", rate=220, enabled=True)
        self._last_speak_time = 0.0
        self._last_command = ""

    def update(self, curve: np.ndarray, center_x: int, center_y: int) -> Optional[str]:
        """
        Determine and speak navigation direction based on curve.

        Args:
            curve: The route curve points
            center_x, center_y: Camera/user position

        Returns:
            The spoken command, or None if not spoken
        """
        now = time.time()
        if now - self._last_speak_time < self.interval:
            return None

        if len(curve) < 10:
            return None

        # Get direction from curve near the start (where user should head)
        # Sample a point ~20% along the curve for immediate direction
        sample_idx = min(len(curve) // 5, len(curve) - 1)
        target_point = curve[sample_idx]

        # Calculate angle from user to this point
        dx = target_point[0] - center_x
        dy = center_y - target_point[1]  # Flip Y since screen Y is inverted

        # Angle in degrees (0 = forward/up, positive = right, negative = left)
        angle_rad = math.atan2(dx, dy)
        angle_deg = math.degrees(angle_rad)

        # Determine command based on angle
        command = self._angle_to_command(angle_deg)

        # Speak if different from last or enough time passed
        if command != self._last_command or now - self._last_speak_time > 3.0:
            self.tts.speak(command, priority="normal")
            self._last_speak_time = now
            self._last_command = command
            return command

        return None

    def _angle_to_command(self, angle_deg: float) -> str:
        """Convert angle to navigation command."""
        # Angle: 0 = forward, +ve = right, -ve = left
        abs_angle = abs(angle_deg)

        if abs_angle < 10:
            return "walk forward"
        elif abs_angle < 30:
            if angle_deg > 0:
                return "bear right"
            else:
                return "bear left"
        elif abs_angle < 60:
            if angle_deg > 0:
                return "turn right"
            else:
                return "turn left"
        else:
            if angle_deg > 0:
                return "sharp right"
            else:
                return "sharp left"


class FlowingCurve:
    """
    A smooth curve that flows/morphs towards target waypoints.

    The entire curve smoothly transitions rather than jumping,
    creating a fluid ribbon-like effect.
    """

    def __init__(self, num_points: int = 100, smoothing: float = 0.08):
        """
        Args:
            num_points: Number of points in the curve
            smoothing: How fast curve morphs to target (0.01-0.2, lower = smoother)
        """
        self.num_points = num_points
        self.smoothing = smoothing

        # Current curve as array of points
        self._curve: Optional[np.ndarray] = None  # Shape: (num_points, 2)

    def update(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        waypoints: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Update curve to flow towards new waypoints.

        Args:
            start: Start point (camera position)
            end: End point (target)
            waypoints: Intermediate waypoints

        Returns:
            Array of curve points (num_points, 2)
        """
        # Generate target curve through waypoints
        all_points = [start] + waypoints
        if waypoints and waypoints[-1] != end:
            all_points.append(end)

        target_curve = self._generate_spline(all_points)

        # Initialize curve if needed
        if self._curve is None or len(self._curve) != self.num_points:
            self._curve = target_curve.copy()
            return self._curve.astype(np.int32)

        # Smoothly morph entire curve towards target
        self._curve = self._curve + (target_curve - self._curve) * self.smoothing

        return self._curve.astype(np.int32)

    def _generate_spline(self, points: List[Tuple[int, int]]) -> np.ndarray:
        """Generate a smooth spline curve through points."""
        if len(points) < 2:
            return np.array([[points[0][0], points[0][1]]] * self.num_points, dtype=np.float32)

        points = np.array(points, dtype=np.float32)

        if len(points) == 2:
            # Simple linear interpolation
            t = np.linspace(0, 1, self.num_points).reshape(-1, 1)
            return points[0] + t * (points[1] - points[0])

        # Catmull-Rom spline for 3+ points
        # Pad start and end for full spline
        padded = np.vstack([points[0], points, points[-1]])

        curve_points = []
        num_segments = len(padded) - 3
        points_per_segment = self.num_points // num_segments

        for i in range(num_segments):
            p0, p1, p2, p3 = padded[i:i+4]

            for j in range(points_per_segment):
                t = j / points_per_segment
                t2 = t * t
                t3 = t2 * t

                # Catmull-Rom formula
                point = 0.5 * (
                    (2 * p1) +
                    (-p0 + p2) * t +
                    (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                    (-p0 + 3*p1 - 3*p2 + p3) * t3
                )
                curve_points.append(point)

        # Add final point
        curve_points.append(padded[-2])

        # Resample to exact num_points
        curve = np.array(curve_points, dtype=np.float32)
        if len(curve) != self.num_points:
            indices = np.linspace(0, len(curve) - 1, self.num_points).astype(int)
            curve = curve[indices]

        return curve

    def reset(self):
        """Reset curve state."""
        self._curve = None


class ProximityAnalyzer:
    """
    Analyzes nearby obstacles and computes curve deviations.

    - Soft deviation based on left/right proximity imbalance (under 3m)
    - Hard emergency turn when obstacle is under 1m and in path
    """

    # Thresholds
    SOFT_DEVIATION_RANGE = 3.0  # meters - consider objects within this range
    EMERGENCY_RANGE = 1.0  # meters - hard turn trigger
    CENTER_ANGLE_THRESHOLD = 0.2  # radians (~11 deg) - "in front" zone

    # Deviation strengths
    MAX_SOFT_DEVIATION = 80  # pixels - max soft curve offset
    EMERGENCY_TURN_OFFSET = 50  # pixels - emergency turn magnitude (subtle)

    def __init__(self):
        self._last_deviation = 0.0  # Smoothed deviation value
        self._emergency_direction = 0  # -1 left, 0 none, 1 right
        self._smoothing = 0.3  # How fast deviation changes

    def analyze(
        self,
        tracks: List[TrackedObject],
        frame_width: int,
        fov_rad: float
    ) -> Tuple[float, Optional[int]]:
        """
        Analyze proximity and compute deviation.

        Args:
            tracks: All tracked objects
            frame_width: Camera frame width
            fov_rad: Field of view in radians

        Returns:
            Tuple of (soft_deviation, emergency_direction)
            - soft_deviation: float, positive = deviate right, negative = deviate left
            - emergency_direction: None if no emergency, -1 for left, 1 for right
        """
        left_proximity = 0.0  # Sum of 1/depth for left side
        right_proximity = 0.0  # Sum of 1/depth for right side

        emergency_obstacle = None  # (depth, angle, class_name)

        for track in tracks:
            depth = track.get_current_depth()
            if depth <= 0 or depth > self.SOFT_DEVIATION_RANGE:
                continue

            # Skip floor/ground detections
            class_lower = track.class_name.lower()
            if class_lower in FLOOR_CLASSES:
                continue

            # Calculate horizontal angle of object
            x1, y1, x2, y2 = track.bbox
            bbox_center_x = (x1 + x2) / 2
            norm_x = bbox_center_x / frame_width
            angle = (norm_x - 0.5) * fov_rad  # negative = left, positive = right

            # ROADS are extremely dangerous - treat as emergency even at greater distance
            is_road = class_lower in ROAD_CLASSES
            if is_road:
                # Roads trigger emergency at 1m instead of 0.6m
                road_emergency_range = 1.0
                if depth < road_emergency_range and abs(angle) < self.CENTER_ANGLE_THRESHOLD * 1.2:
                    # Road is highest priority emergency
                    if emergency_obstacle is None or is_road:
                        emergency_obstacle = (depth, angle, class_lower)

            # Check for emergency obstacle (< 1m and in front)
            elif depth < self.EMERGENCY_RANGE:
                # Is it in front of us (within center cone)?
                if abs(angle) < self.CENTER_ANGLE_THRESHOLD:
                    # Skip walls unless directly centered
                    if class_lower in WALL_CLASSES and abs(angle) > 0.1:
                        continue

                    # This is an emergency obstacle
                    if emergency_obstacle is None or depth < emergency_obstacle[0]:
                        emergency_obstacle = (depth, angle, class_lower)

            # Accumulate proximity for soft deviation (inverse depth weighting)
            proximity_weight = 1.0 / depth

            # Roads get 5x weight - avoid them like the plague
            if is_road:
                proximity_weight *= 5.0

            # Weight more heavily for objects closer to center (more dangerous)
            center_weight = 1.0 - (abs(angle) / (fov_rad / 2)) * 0.5
            proximity_weight *= center_weight

            if angle < 0:
                left_proximity += proximity_weight
            else:
                right_proximity += proximity_weight

        # Compute soft deviation based on left/right imbalance
        total_proximity = left_proximity + right_proximity
        if total_proximity > 0.1:
            # Ratio: positive means more on left, so deviate right
            imbalance = (left_proximity - right_proximity) / total_proximity
            target_deviation = imbalance * self.MAX_SOFT_DEVIATION
        else:
            target_deviation = 0.0

        # Smooth the deviation
        self._last_deviation += (target_deviation - self._last_deviation) * self._smoothing

        # Handle emergency obstacle
        emergency_direction = None
        if emergency_obstacle:
            depth, angle, class_name = emergency_obstacle
            # Turn away from obstacle - if it's slightly left, turn right, etc.
            # But also consider current curve direction
            if angle <= 0:
                emergency_direction = 1  # Turn right (obstacle on left/center-left)
            else:
                emergency_direction = -1  # Turn left (obstacle on right/center-right)

        return self._last_deviation, emergency_direction

    def reset(self):
        """Reset state."""
        self._last_deviation = 0.0
        self._emergency_direction = 0


def _apply_proximity_deviation(
    curve: np.ndarray,
    center_x: int,
    center_y: int,
    scale: float,
    soft_deviation: float,
    emergency_direction: Optional[int]
) -> np.ndarray:
    """
    Apply proximity-based deviations to the curve.

    Args:
        curve: Original curve points (num_points, 2)
        center_x, center_y: Camera position
        scale: Bird's eye view scale
        soft_deviation: Soft horizontal deviation in pixels
        emergency_direction: Emergency turn direction (-1 left, 1 right, None = no emergency)

    Returns:
        Modified curve with deviations applied
    """
    if len(curve) < 2:
        return curve

    curve = curve.astype(np.float32).copy()
    num_points = len(curve)

    # Store original start point - we'll anchor the curve to start here
    start_x, start_y = center_x, center_y

    # Define the "near" portion of the curve (first ~50% corresponding to ~3m zone)
    near_portion = 0.5
    near_points = int(num_points * near_portion)

    # Apply soft deviation with falloff (skip first point to keep anchor)
    if abs(soft_deviation) > 0.5:
        for i in range(1, near_points):
            # Falloff: starts small, peaks around 20%, then fades
            t = i / near_points
            # Bell-shaped falloff: peaks early, then fades
            falloff = t * (1.0 - t) * 4  # Peaks at t=0.5 with value 1.0
            curve[i, 0] += soft_deviation * falloff

    # Apply emergency hard turn (skip first point, affects ~30% of curve)
    if emergency_direction is not None:
        emergency_portion = 0.3
        emergency_points = int(num_points * emergency_portion)
        emergency_offset = ProximityAnalyzer.EMERGENCY_TURN_OFFSET * emergency_direction

        for i in range(1, emergency_points):
            # Bell-shaped falloff for emergency turn
            t = i / emergency_points
            falloff = t * (1.0 - t) * 4
            curve[i, 0] += emergency_offset * falloff

    # Force first point to be exactly at camera position
    curve[0, 0] = start_x
    curve[0, 1] = start_y

    return curve.astype(np.int32)


# Global instances
_flowing_curve: Optional[FlowingCurve] = None
_nav_voice: Optional[NavigationVoice] = None
_proximity_analyzer: Optional[ProximityAnalyzer] = None


def draw_route(
    view: np.ndarray,
    route: PlannedRoute,
    tracks: List[TrackedObject],
    center_x: int,
    center_y: int,
    scale: float,
    fov_rad: float,
    frame_width: int
) -> np.ndarray:
    """
    Draw planned route on bird's eye view.
    """
    global _flowing_curve, _nav_voice, _proximity_analyzer

    if _flowing_curve is None:
        _flowing_curve = FlowingCurve(num_points=80, smoothing=0.18)

    if _nav_voice is None:
        _nav_voice = NavigationVoice(interval=1.0)

    if _proximity_analyzer is None:
        _proximity_analyzer = ProximityAnalyzer()

    if not route or not route.is_valid or not route.waypoints:
        return view

    # Get track positions
    track_positions = _compute_track_positions(
        tracks, center_x, center_y, scale, fov_rad, frame_width
    )

    # Collect waypoint screen positions
    waypoint_positions = []
    for wp in route.waypoints:
        if wp.track_id in track_positions:
            waypoint_positions.append(track_positions[wp.track_id])

    if not waypoint_positions:
        return view

    # Camera position (start)
    start = (center_x, center_y)
    # Target (end)
    end = waypoint_positions[-1]
    # Intermediate waypoints
    intermediates = waypoint_positions[:-1] if len(waypoint_positions) > 1 else []

    # Get smoothly flowing curve
    curve = _flowing_curve.update(start, end, intermediates)

    # Analyze proximity and get deviations
    soft_deviation, emergency_direction = _proximity_analyzer.analyze(
        tracks, frame_width, fov_rad
    )

    # Apply proximity-based deviations to curve
    curve = _apply_proximity_deviation(
        curve, center_x, center_y, scale,
        soft_deviation, emergency_direction
    )

    # Speak navigation direction (using deviated curve)
    _nav_voice.update(curve, center_x, center_y)

    # Draw the curve (with deviations applied)
    _draw_curve(view, curve, emergency_direction is not None)

    # Draw simple waypoint markers (at actual positions, not smoothed)
    _draw_waypoint_markers(view, route, track_positions)

    # Draw target marker
    if route.target_id and route.target_id in track_positions:
        _draw_target_marker(view, track_positions[route.target_id])

    # Status
    _draw_status(view, route)

    return view


def _compute_track_positions(
    tracks: List[TrackedObject],
    center_x: int,
    center_y: int,
    scale: float,
    fov_rad: float,
    frame_width: int
) -> Dict[int, Tuple[int, int]]:
    """Compute screen positions for all tracks."""
    positions = {}

    for track in tracks:
        depth = track.get_current_depth()
        if depth <= 0:
            continue

        x1, y1, x2, y2 = track.bbox
        bbox_center_x = (x1 + x2) / 2
        norm_x = bbox_center_x / frame_width
        angle = (norm_x - 0.5) * fov_rad

        obj_x = depth * math.sin(angle)
        obj_y = depth * math.cos(angle)

        view_x = int(center_x + obj_x * scale)
        view_y = int(center_y - obj_y * scale)

        positions[track.track_id] = (view_x, view_y)

    return positions


def _draw_curve(view: np.ndarray, curve: np.ndarray, emergency: bool = False) -> None:
    """Draw the flowing curve with gradient."""
    if len(curve) < 2:
        return

    # Draw as polyline with gradient coloring
    for i in range(len(curve) - 1):
        t = i / (len(curve) - 1)

        if emergency and t < 0.2:
            # Red/orange pulsing for emergency portion
            pulse = 0.5 + 0.5 * math.sin(time.time() * 10)
            color = (0, int(50 + 100 * pulse), 255)  # Red-orange
            thickness = 5
        else:
            # Green -> Yellow -> Orange gradient
            color = (0, int(255 - 100 * t), int(200 * t + 55))
            thickness = 3

        pt1 = tuple(curve[i])
        pt2 = tuple(curve[i + 1])

        cv2.line(view, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw arrows along curve
    for frac in [0.3, 0.6]:
        idx = int(frac * (len(curve) - 1))
        if idx < 2 or idx >= len(curve) - 2:
            continue

        p1 = curve[idx - 2]
        p2 = curve[idx + 2]

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1:
            continue
        dx, dy = dx/length, dy/length

        px, py = curve[idx]
        arrow_len = 10

        t = frac
        color = (0, int(255 - 100 * t), int(200 * t + 55))

        cv2.arrowedLine(
            view,
            (int(px - dx * arrow_len), int(py - dy * arrow_len)),
            (int(px + dx * arrow_len), int(py + dy * arrow_len)),
            color, 2, tipLength=0.5
        )


def _draw_waypoint_markers(
    view: np.ndarray,
    route: PlannedRoute,
    track_positions: Dict[int, Tuple[int, int]]
) -> None:
    """Draw numbered circle markers at waypoints."""
    route_color = (0, 255, 255)  # Yellow

    for wp in route.waypoints:
        if wp.track_id not in track_positions:
            continue

        pos = track_positions[wp.track_id]

        # Circle
        cv2.circle(view, pos, 18, route_color, 2)

        # Number
        label = str(wp.order)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            view, label,
            (pos[0] - tw // 2, pos[1] + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, route_color, 2
        )


def _draw_target_marker(view: np.ndarray, pos: Tuple[int, int]) -> None:
    """Draw star on target."""
    color = (0, 165, 255)  # Orange
    size = 22

    cx, cy = pos
    points = []
    for i in range(8):
        angle = i * math.pi / 4 - math.pi / 2
        r = size if i % 2 == 0 else size // 2
        points.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(view, [pts], True, color, 2)


def _draw_status(view: np.ndarray, route: PlannedRoute) -> None:
    """Draw status text."""
    if route.waypoints:
        depth = route.waypoints[-1].depth_m
        cv2.putText(view, f"Target: {depth:.0f}m", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)


def reset_smooth_path():
    """Reset curve state."""
    global _flowing_curve, _nav_voice, _proximity_analyzer
    if _flowing_curve:
        _flowing_curve.reset()
    if _proximity_analyzer:
        _proximity_analyzer.reset()
    _nav_voice = None
