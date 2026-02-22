"""Route visualization on bird's eye view."""

import math
import cv2
import numpy as np
from typing import List, Dict, Tuple

from .route_types import PlannedRoute
from src.detection.object_tracker import TrackedObject


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

    Args:
        view: Bird's eye view image to draw on.
        route: Planned route to visualize.
        tracks: Current tracked objects (for position lookup).
        center_x, center_y: Camera position on view.
        scale: Pixels per meter.
        fov_rad: Field of view in radians.
        frame_width: Camera frame width.

    Returns:
        View with route overlay.
    """
    if not route or not route.is_valid:
        return view

    # Build track position lookup
    track_positions = _compute_track_positions(
        tracks, center_x, center_y, scale, fov_rad, frame_width
    )

    # Draw avoid indicators first (below route lines)
    for avoid_id in route.avoid_ids:
        if avoid_id in track_positions:
            _draw_avoid_marker(view, track_positions[avoid_id])

    # Draw route lines and waypoint markers
    if route.waypoints:
        _draw_route_path(view, route, track_positions, center_x, center_y)

    # Draw target star on furthest object
    if route.target_id and route.target_id in track_positions:
        _draw_target_star(view, track_positions[route.target_id])

    # Draw route status text
    _draw_route_status(view, route)

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

        # Same math as BirdsEyeView.render()
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


def _draw_route_path(
    view: np.ndarray,
    route: PlannedRoute,
    track_positions: Dict[int, Tuple[int, int]],
    center_x: int,
    center_y: int
) -> None:
    """Draw route lines connecting waypoints."""
    route_color = (0, 255, 255)  # Yellow (BGR)
    line_thickness = 2

    prev_pos = (center_x, center_y)  # Start from camera

    for wp in route.waypoints:
        if wp.track_id not in track_positions:
            continue

        curr_pos = track_positions[wp.track_id]

        # Draw line segment
        cv2.line(view, prev_pos, curr_pos, route_color, line_thickness)

        # Draw waypoint number circle
        cv2.circle(view, curr_pos, 18, route_color, 2)
        label = str(wp.order)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            view, label,
            (curr_pos[0] - tw // 2, curr_pos[1] + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, route_color, 2
        )

        prev_pos = curr_pos

    # Draw direction arrow on first segment
    if route.waypoints and route.waypoints[0].track_id in track_positions:
        first_pos = track_positions[route.waypoints[0].track_id]
        _draw_direction_arrow(view, (center_x, center_y), first_pos, route_color)


def _draw_direction_arrow(
    view: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int]
) -> None:
    """Draw arrow indicating direction at midpoint of line."""
    mid_x = (start[0] + end[0]) // 2
    mid_y = (start[1] + end[1]) // 2

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 20:
        return

    dx /= length
    dy /= length

    arrow_len = 12
    cv2.arrowedLine(
        view,
        (int(mid_x - dx * arrow_len), int(mid_y - dy * arrow_len)),
        (int(mid_x + dx * arrow_len), int(mid_y + dy * arrow_len)),
        color, 2, tipLength=0.5
    )


def _draw_avoid_marker(view: np.ndarray, pos: Tuple[int, int]) -> None:
    """Draw red X marker for objects to avoid."""
    color = (0, 0, 255)  # Red
    size = 12

    # Red circle
    cv2.circle(view, pos, size + 5, color, 2)

    # X through it
    cv2.line(view, (pos[0] - size, pos[1] - size),
             (pos[0] + size, pos[1] + size), color, 2)
    cv2.line(view, (pos[0] + size, pos[1] - size),
             (pos[0] - size, pos[1] + size), color, 2)


def _draw_target_star(view: np.ndarray, pos: Tuple[int, int]) -> None:
    """Draw star marker on final target."""
    color = (0, 200, 255)  # Orange-yellow
    size = 22

    # 8-point star
    cx, cy = pos
    points = []
    for i in range(8):
        angle = i * math.pi / 4 - math.pi / 2
        r = size if i % 2 == 0 else size // 2
        points.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(view, [pts], True, color, 2)


def _draw_route_status(view: np.ndarray, route: PlannedRoute) -> None:
    """Draw route status text overlay."""
    age = route.age_seconds()

    if route.waypoints:
        status = f"Route: {len(route.waypoints)} stops ({age:.1f}s ago)"
    else:
        status = f"No route ({age:.1f}s ago)"

    # Draw at top of view, below title
    cv2.putText(view, status, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
