"""Bird's eye view visualization of detected objects with visual odometry support."""

import math
import time
import cv2
import numpy as np
from typing import List, Optional
from collections import deque

from .object_tracker import TrackedObject


class BirdsEyeView:
    """
    Renders a 2D top-down bird's eye view of detected objects.

    Supports two modes:
    1. Camera-relative: Camera fixed at center, objects positioned relative to camera.
    2. World coordinates: Camera moves on map, objects at world positions, trajectory shown.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        max_depth_m: float = 15.0,
        fov_degrees: float = 75.0
    ):
        """
        Initialize bird's eye view renderer.

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            max_depth_m: Maximum depth/radius to display (meters).
            fov_degrees: Horizontal field of view.
        """
        self.width = width
        self.height = height
        self.max_depth_m = max_depth_m
        self.fov_degrees = fov_degrees
        self.fov_rad = math.radians(fov_degrees)

        # Pixels per meter (radius fills half the view)
        self.scale = min(width, height) / (2 * max_depth_m) * 0.9

        # Colors for different object classes (BGR)
        self.class_colors = {
            'person': (0, 165, 255),      # Orange
            'car': (255, 0, 0),           # Blue
            'truck': (255, 100, 0),       # Light blue
            'bicycle': (0, 255, 0),       # Green
            'motorcycle': (0, 200, 0),    # Dark green
            'dog': (180, 105, 255),       # Pink
            'cat': (203, 192, 255),       # Light pink
            'chair': (0, 255, 255),       # Yellow
            'bottle': (255, 255, 0),      # Cyan
            'couch': (128, 0, 128),       # Purple
            'tv': (255, 128, 0),          # Light blue
            'dining table': (0, 128, 128),  # Olive
            'potted plant': (0, 128, 0),  # Dark green
            'bed': (128, 128, 0),         # Teal
            'toilet': (192, 192, 192),    # Silver
            'refrigerator': (128, 128, 128),  # Gray
        }
        self.default_color = (128, 128, 128)  # Gray

        # Animation state for pulsing effects
        self._pulse_phase = 0.0

    def render(
        self,
        tracks: List[TrackedObject],
        frame_width: int,
        camera_pose=None,
        trajectory: Optional[deque] = None,
        tracking_state: str = "normal"
    ) -> np.ndarray:
        """
        Render bird's eye view of tracked objects.

        When camera_pose is provided, renders in world-coordinate mode with
        the camera at center, world rotated based on heading, and trajectory drawn.

        Args:
            tracks: List of tracked objects with depth info.
            frame_width: Width of the source camera frame (for X mapping).
            camera_pose: Optional CameraPose for world-coordinate rendering.
            trajectory: Optional deque of CameraPose for trajectory visualization.
            tracking_state: Tracking state ("normal", "degraded", "panic", "recovering").

        Returns:
            BGR image of the bird's eye view.
        """
        # Update pulse animation
        self._pulse_phase = (time.time() * 4) % (2 * math.pi)
        # Create dark background
        view = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        view[:] = (30, 30, 30)  # Dark gray background

        # Center of view (camera position on screen)
        center_x = self.width // 2
        center_y = self.height // 2

        # Get heading for rotation (0 if no pose)
        heading = camera_pose.heading if camera_pose else 0.0

        # Draw circular grid
        self._draw_circular_grid(view, center_x, center_y)

        # Draw trajectory if in world mode
        if camera_pose is not None and trajectory is not None:
            self._draw_trajectory(view, center_x, center_y, camera_pose, trajectory)

        # Draw FOV cone (rotates with heading in world mode)
        self._draw_fov_cone(view, center_x, center_y, heading=0.0)  # Cone always points up

        # Draw compass if in world mode
        if camera_pose is not None:
            self._draw_compass(view, heading)

        # Draw camera indicator with tracking state
        self._draw_camera_indicator(view, center_x, center_y, tracking_state)

        # Render each tracked object
        for track in tracks:
            depth = track.get_current_depth()

            # Skip objects without valid depth
            if depth <= 0 or depth > self.max_depth_m:
                continue

            # Get center X position of bounding box
            x1, y1, x2, y2 = track.bbox
            bbox_center_x = (x1 + x2) / 2

            # Calculate angle from center of frame
            # norm_x: 0 = left edge, 0.5 = center, 1 = right edge
            norm_x = bbox_center_x / frame_width
            angle = (norm_x - 0.5) * self.fov_rad  # -fov/2 to +fov/2

            # Convert polar (angle, depth) to cartesian (x, y)
            # Camera faces up (north), so:
            # - positive angle = right = positive x
            # - depth = distance forward = positive y (but screen y is inverted)
            obj_x = depth * math.sin(angle)
            obj_y = depth * math.cos(angle)

            # Convert to screen coordinates
            view_x = int(center_x + obj_x * self.scale)
            view_y = int(center_y - obj_y * self.scale)  # Invert Y for screen

            # Clamp to view bounds
            view_x = max(10, min(self.width - 10, view_x))
            view_y = max(10, min(self.height - 10, view_y))

            # Get color for this class
            color = self.class_colors.get(track.class_name, self.default_color)

            # Draw object circle (size based on confidence)
            radius = max(6, int(10 * track.confidence))
            cv2.circle(view, (view_x, view_y), radius, color, -1)
            cv2.circle(view, (view_x, view_y), radius, (255, 255, 255), 1)

            # Draw label
            label = f"#{track.track_id} {track.class_name} {depth:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label above or below the circle
            label_y = view_y - radius - 5 if view_y > center_y else view_y + radius + 12
            label_x = max(2, min(self.width - text_w - 2, view_x - text_w // 2))

            # Background for label
            cv2.rectangle(view,
                          (label_x - 2, label_y - text_h - 2),
                          (label_x + text_w + 2, label_y + 2),
                          (0, 0, 0), -1)
            cv2.putText(view, label, (label_x, label_y),
                        font, font_scale, (255, 255, 255), thickness)

        # Draw title and position info
        title = "Bird's Eye View"
        if camera_pose is not None:
            title += f" | Pos: ({camera_pose.x:.1f}, {camera_pose.y:.1f})m"
            title += f" | Hdg: {math.degrees(camera_pose.heading):.0f}deg"
        cv2.putText(view, title, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Instructions for reset
        if camera_pose is not None:
            cv2.putText(view, "Press 'R' to reset odometry", (10, self.height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        return view

    def _draw_circular_grid(self, view: np.ndarray, center_x: int, center_y: int) -> None:
        """Draw circular distance grid centered on camera."""
        # Draw concentric circles at distance intervals
        if self.max_depth_m <= 5:
            interval = 1
        elif self.max_depth_m <= 10:
            interval = 2
        else:
            interval = 5  # 5m intervals for 15m range

        for dist_m in range(interval, int(self.max_depth_m) + 1, interval):
            radius_px = int(dist_m * self.scale)
            cv2.circle(view, (center_x, center_y), radius_px, (60, 60, 60), 1)

            # Label at top of circle
            label_y = center_y - radius_px - 5
            if label_y > 10:
                cv2.putText(view, f"{dist_m}m",
                            (center_x - 10, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Draw outer boundary circle
        boundary_radius = int(self.max_depth_m * self.scale)
        cv2.circle(view, (center_x, center_y), boundary_radius, (80, 80, 80), 2)

    def _draw_fov_cone(self, view: np.ndarray, center_x: int, center_y: int, heading: float = 0.0) -> None:
        """Draw FOV cone lines from camera."""
        half_fov = self.fov_rad / 2
        line_length = int(self.max_depth_m * self.scale)

        # Left FOV boundary
        left_angle = -half_fov + heading
        left_x = int(center_x + line_length * math.sin(left_angle))
        left_y = int(center_y - line_length * math.cos(left_angle))
        cv2.line(view, (center_x, center_y), (left_x, left_y), (80, 80, 80), 1)

        # Right FOV boundary
        right_angle = half_fov + heading
        right_x = int(center_x + line_length * math.sin(right_angle))
        right_y = int(center_y - line_length * math.cos(right_angle))
        cv2.line(view, (center_x, center_y), (right_x, right_y), (80, 80, 80), 1)

    def _draw_camera_indicator(
        self,
        view: np.ndarray,
        center_x: int,
        center_y: int,
        tracking_state: str = "normal"
    ) -> None:
        """
        Draw camera position and heading indicator with tracking state visualization.

        States:
        - NORMAL: White circle + green arrow
        - PANIC: Pulsing red ring + "?" symbol + "LOST" label
        """
        # Calculate pulse intensity (0.5 to 1.0)
        pulse = 0.5 + 0.5 * abs(math.sin(self._pulse_phase))

        if tracking_state == "panic":
            # PANIC: Red pulsing ring with "?" and "LOST"
            ring_radius = int(15 + 5 * pulse)
            ring_color = (0, 0, int(255 * pulse))  # Pulsing red

            # Outer pulsing ring
            cv2.circle(view, (center_x, center_y), ring_radius, ring_color, 3)

            # Inner dark circle
            cv2.circle(view, (center_x, center_y), 10, (40, 40, 40), -1)
            cv2.circle(view, (center_x, center_y), 10, ring_color, 2)

            # Question mark symbol
            cv2.putText(
                view, "?",
                (center_x - 5, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

            # "LOST" label
            cv2.putText(
                view, "LOST",
                (center_x - 18, center_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        else:
            # NORMAL: White circle + green arrow
            cv2.circle(view, (center_x, center_y), 12, (100, 100, 100), 1)
            cv2.circle(view, (center_x, center_y), 10, (255, 255, 255), -1)

            # Green heading arrow
            arrow_length = 25
            cv2.arrowedLine(
                view,
                (center_x, center_y),
                (center_x, center_y - arrow_length),
                (0, 255, 0),  # Green arrow
                2,
                tipLength=0.4
            )

            # Label
            cv2.putText(
                view, "CAM",
                (center_x - 15, center_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

    def _draw_trajectory(
        self,
        view: np.ndarray,
        center_x: int,
        center_y: int,
        camera_pose,
        trajectory: deque
    ) -> None:
        """Draw camera trajectory (path through world)."""
        if len(trajectory) < 2:
            return

        # Convert trajectory to screen coordinates (camera-centered, rotated by -heading)
        cos_h = math.cos(-camera_pose.heading)
        sin_h = math.sin(-camera_pose.heading)

        points = []
        for pose in trajectory:
            # Relative to current camera position
            dx = pose.x - camera_pose.x
            dy = pose.y - camera_pose.y

            # Rotate by negative heading so camera's heading is always up
            screen_x = center_x + int((dx * cos_h - dy * sin_h) * self.scale)
            screen_y = center_y - int((dx * sin_h + dy * cos_h) * self.scale)

            points.append((screen_x, screen_y))

        # Draw trajectory line
        for i in range(1, len(points)):
            # Fade older parts of trajectory
            alpha = 0.3 + 0.7 * (i / len(points))
            color = (int(100 * alpha), int(100 * alpha), int(255 * alpha))  # Blue gradient
            cv2.line(view, points[i - 1], points[i], color, 1)

        # Draw start marker (oldest point)
        if points:
            cv2.circle(view, points[0], 4, (0, 100, 255), -1)  # Orange

    def _draw_compass(self, view: np.ndarray, heading: float) -> None:
        """Draw compass showing cardinal directions."""
        # Compass position (top-right corner)
        compass_x = self.width - 50
        compass_y = 50
        radius = 30

        # Draw compass circle
        cv2.circle(view, (compass_x, compass_y), radius, (60, 60, 60), 1)

        # Draw cardinal directions (rotated by negative heading)
        directions = [
            ('N', 0),
            ('E', math.pi / 2),
            ('S', math.pi),
            ('W', -math.pi / 2)
        ]

        for label, angle in directions:
            # Rotate direction by negative heading
            rotated_angle = angle - heading
            dx = int(radius * 0.8 * math.sin(rotated_angle))
            dy = int(-radius * 0.8 * math.cos(rotated_angle))

            # Position for label
            label_x = compass_x + dx - 4
            label_y = compass_y + dy + 4

            # Draw label (N is highlighted)
            color = (0, 200, 255) if label == 'N' else (150, 150, 150)
            cv2.putText(view, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw north arrow (always points to actual north)
        north_angle = -heading
        arrow_len = radius - 5
        north_dx = int(arrow_len * math.sin(north_angle))
        north_dy = int(-arrow_len * math.cos(north_angle))
        cv2.arrowedLine(
            view,
            (compass_x, compass_y),
            (compass_x + north_dx, compass_y + north_dy),
            (0, 200, 255),  # Yellow/orange
            1,
            tipLength=0.4
        )
