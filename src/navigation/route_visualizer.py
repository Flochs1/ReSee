"""Route visualization on bird's eye view with smooth flowing curves."""

import math
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from .route_types import PlannedRoute
from src.detection.object_tracker import TrackedObject


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


# Global curve instance
_flowing_curve: Optional[FlowingCurve] = None


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
    global _flowing_curve

    if _flowing_curve is None:
        _flowing_curve = FlowingCurve(num_points=80, smoothing=0.06)

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

    # Draw the curve
    _draw_curve(view, curve)

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


def _draw_curve(view: np.ndarray, curve: np.ndarray) -> None:
    """Draw the flowing curve with gradient."""
    if len(curve) < 2:
        return

    # Draw as polyline with gradient coloring
    for i in range(len(curve) - 1):
        t = i / (len(curve) - 1)

        # Green -> Yellow -> Orange gradient
        color = (0, int(255 - 100 * t), int(200 * t + 55))

        pt1 = tuple(curve[i])
        pt2 = tuple(curve[i + 1])

        cv2.line(view, pt1, pt2, color, 3, cv2.LINE_AA)

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
    global _flowing_curve
    if _flowing_curve:
        _flowing_curve.reset()
