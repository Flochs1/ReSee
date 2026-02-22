"""Bird's eye view visualization of detected objects."""

import math
import cv2
import numpy as np
from typing import List

from .object_tracker import TrackedObject


class BirdsEyeView:
    """
    Renders a 2D top-down bird's eye view of detected objects.

    Camera is fixed at origin (0,0) facing north (up).
    Objects are positioned within the 75° FOV cone in front of the camera.
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

    def render(
        self,
        tracks: List[TrackedObject],
        frame_width: int
    ) -> np.ndarray:
        """
        Render bird's eye view of tracked objects.

        Camera is at center, facing up (north). Objects are positioned
        based on their depth and horizontal position in the frame.

        Args:
            tracks: List of tracked objects with depth info.
            frame_width: Width of the source camera frame (for X mapping).

        Returns:
            BGR image of the bird's eye view.
        """
        # Create dark background
        view = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        view[:] = (30, 30, 30)  # Dark gray background

        # Center of view (camera position)
        center_x = self.width // 2
        center_y = self.height // 2

        # Draw circular grid
        self._draw_circular_grid(view, center_x, center_y)

        # Draw FOV cone (facing up/north)
        self._draw_fov_cone(view, center_x, center_y)

        # Draw camera indicator
        self._draw_camera_indicator(view, center_x, center_y)

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

        # Draw title
        cv2.putText(view, "Bird's Eye View", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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

    def _draw_fov_cone(self, view: np.ndarray, center_x: int, center_y: int) -> None:
        """Draw FOV cone lines from camera (facing up/north)."""
        half_fov = self.fov_rad / 2
        line_length = int(self.max_depth_m * self.scale)

        # Left FOV boundary (camera facing up, so left is negative angle)
        left_x = int(center_x - line_length * math.sin(half_fov))
        left_y = int(center_y - line_length * math.cos(half_fov))
        cv2.line(view, (center_x, center_y), (left_x, left_y), (80, 80, 80), 1)

        # Right FOV boundary
        right_x = int(center_x + line_length * math.sin(half_fov))
        right_y = int(center_y - line_length * math.cos(half_fov))
        cv2.line(view, (center_x, center_y), (right_x, right_y), (80, 80, 80), 1)

    def _draw_camera_indicator(self, view: np.ndarray, center_x: int, center_y: int) -> None:
        """Draw camera position and heading indicator (facing up/north)."""
        # Camera body (circle)
        cv2.circle(view, (center_x, center_y), 10, (255, 255, 255), -1)

        # Heading arrow (points up)
        arrow_length = 25
        arrow_x = center_x
        arrow_y = center_y - arrow_length

        cv2.arrowedLine(
            view,
            (center_x, center_y),
            (arrow_x, arrow_y),
            (0, 255, 0),  # Green arrow
            2,
            tipLength=0.4
        )

        # Label
        cv2.putText(view, "CAM", (center_x - 15, center_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
