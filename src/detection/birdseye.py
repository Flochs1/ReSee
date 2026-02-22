"""Bird's eye view visualization of detected objects with world-fixed coordinates."""

import math
import time
import cv2
import numpy as np
from typing import List, Optional

from .object_tracker import TrackedObject
from .world_map import WorldObject, CameraState


class BirdsEyeView:
    """
    Renders a 2D top-down bird's eye view of detected objects.

    Supports two modes:
    - Legacy mode: Objects positioned relative to camera (camera-centric)
    - World mode: Objects positioned in world coordinates with heading indicator

    The view shows:
    - Circular 15m radius boundary centered on camera
    - Camera heading indicator
    - Objects as colored shapes (squares for anchors, circles for moving)
    - Fade effect for objects not recently seen
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
            height: Output image height in pixels (square for world view).
            max_depth_m: Maximum depth/radius to display (meters).
            fov_degrees: Approximate horizontal field of view.
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
        frame_width: int,
        depth_map: np.ndarray = None
    ) -> np.ndarray:
        """
        Render bird's eye view of tracked objects (legacy camera-centric mode).

        Args:
            tracks: List of tracked objects with depth info.
            frame_width: Width of the source camera frame (for X mapping).
            depth_map: Optional depth map to sample from if track has no depth.

        Returns:
            BGR image of the bird's eye view.
        """
        # Create dark background
        view = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        view[:] = (30, 30, 30)  # Dark gray background

        # Draw grid lines for depth
        self._draw_grid(view)

        # Draw camera position indicator at bottom center
        cam_x = self.width // 2
        cam_y = self.height - 20
        cv2.circle(view, (cam_x, cam_y), 8, (255, 255, 255), -1)
        cv2.putText(view, "CAM", (cam_x - 15, cam_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw FOV lines
        fov_rad = np.radians(self.fov_degrees / 2)
        line_len = self.height - 40
        left_x = int(cam_x - line_len * np.tan(fov_rad))
        right_x = int(cam_x + line_len * np.tan(fov_rad))
        cv2.line(view, (cam_x, cam_y), (left_x, 20), (80, 80, 80), 1)
        cv2.line(view, (cam_x, cam_y), (right_x, 20), (80, 80, 80), 1)

        # Render each tracked object
        for track in tracks:
            depth = track.get_current_depth()

            # Skip objects without valid depth
            if depth <= 0 or depth > self.max_depth_m:
                continue

            # Get center X position of bounding box
            x1, y1, x2, y2 = track.bbox
            center_x = (x1 + x2) / 2

            # Map depth: 0 = bottom (camera), max_depth = top
            usable_height = self.height - 50
            depth_ratio = depth / self.max_depth_m
            view_y = int(self.height - 30 - depth_ratio * usable_height)
            view_y = max(10, min(self.height - 30, view_y))

            # Horizontal position: linear interpolation within FOV triangle
            # norm_x: 0 = left edge, 0.5 = center, 1 = right edge
            norm_x = center_x / frame_width
            offset_from_center = norm_x - 0.5  # -0.5 to +0.5

            # Scale horizontal spread by depth ratio (closer = narrower spread)
            cam_x = self.width // 2
            max_spread = self.width // 2 - 20  # max horizontal spread at max depth
            view_x = int(cam_x + offset_from_center * 2 * max_spread * depth_ratio)
            view_x = max(10, min(self.width - 10, view_x))

            # Get color for this class
            color = self.class_colors.get(track.class_name, self.default_color)

            # Draw object circle (size based on confidence)
            radius = max(6, int(12 * track.confidence))
            cv2.circle(view, (view_x, view_y), radius, color, -1)
            cv2.circle(view, (view_x, view_y), radius, (255, 255, 255), 1)

            # Draw label
            label = f"#{track.track_id} {track.class_name} {depth:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label above or below the circle
            label_y = view_y - radius - 5 if view_y > 30 else view_y + radius + 12
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

    def render_world(
        self,
        world_objects: List[WorldObject],
        camera_state: CameraState,
        current_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Render bird's eye view with world-fixed coordinates.

        Camera is always centered, objects are positioned relative to camera
        in world coordinates. Heading indicator shows camera orientation.

        Args:
            world_objects: List of objects with world coordinates.
            camera_state: Current camera position and heading.
            current_time: Current timestamp for fade calculation.

        Returns:
            BGR image of the bird's eye view.
        """
        if current_time is None:
            current_time = time.monotonic()

        # Create dark background
        view = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        view[:] = (30, 30, 30)  # Dark gray background

        # Center of view (camera position)
        center_x = self.width // 2
        center_y = self.height // 2

        # Draw circular boundary
        self._draw_circular_grid(view, center_x, center_y)

        # Draw FOV cone (rotated by camera heading)
        self._draw_fov_cone(view, center_x, center_y, camera_state.heading)

        # Draw camera indicator
        self._draw_camera_indicator(view, center_x, center_y, camera_state.heading)

        # Render each world object
        for obj in world_objects:
            # Calculate relative position from camera
            rel_x = obj.world_x - camera_state.x
            rel_y = obj.world_y - camera_state.y

            # Skip if outside visible range
            distance = math.sqrt(rel_x**2 + rel_y**2)
            if distance > self.max_depth_m:
                continue

            # Rotate so camera heading points up
            # In our coordinate system, +Y is forward, +X is right
            # We want heading=0 to point up on screen
            cos_h = math.cos(-camera_state.heading)
            sin_h = math.sin(-camera_state.heading)
            rotated_x = rel_x * cos_h - rel_y * sin_h
            rotated_y = rel_x * sin_h + rel_y * cos_h

            # Convert to screen coordinates
            # +rotated_y points up (away from camera), -rotated_y points down (toward camera)
            # Camera is at center, forward is up
            view_x = int(center_x + rotated_x * self.scale)
            view_y = int(center_y - rotated_y * self.scale)  # Invert Y for screen

            # Clamp to view bounds
            view_x = max(10, min(self.width - 10, view_x))
            view_y = max(10, min(self.height - 10, view_y))

            # Calculate fade based on time since last seen
            age = current_time - obj.last_seen
            fade = max(0.3, 1.0 - (age / 2.0))  # Fade over 2 seconds, min 0.3

            # Get color for this class
            base_color = self.class_colors.get(obj.class_name, self.default_color)
            color = tuple(int(c * fade) for c in base_color)

            # Draw object (square for anchors, circle for others)
            radius = max(6, int(10 * obj.confidence))

            if obj.is_anchor:
                # Draw square for stationary objects
                half_size = radius
                cv2.rectangle(
                    view,
                    (view_x - half_size, view_y - half_size),
                    (view_x + half_size, view_y + half_size),
                    color, -1
                )
                cv2.rectangle(
                    view,
                    (view_x - half_size, view_y - half_size),
                    (view_x + half_size, view_y + half_size),
                    (255, 255, 255), 1
                )
            else:
                # Draw circle for moving objects
                cv2.circle(view, (view_x, view_y), radius, color, -1)
                cv2.circle(view, (view_x, view_y), radius, (255, 255, 255), 1)

            # Draw label
            label = f"#{obj.track_id} {obj.class_name} {obj.depth:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            label_color = tuple(int(255 * fade) for _ in range(3))
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label above or below
            label_y = view_y - radius - 5 if view_y > center_y else view_y + radius + 12
            label_x = max(2, min(self.width - text_w - 2, view_x - text_w // 2))

            # Background for label
            cv2.rectangle(view,
                          (label_x - 2, label_y - text_h - 2),
                          (label_x + text_w + 2, label_y + 2),
                          (0, 0, 0), -1)
            cv2.putText(view, label, (label_x, label_y),
                        font, font_scale, label_color, thickness)

        # Draw title and heading info
        heading_deg = math.degrees(camera_state.heading)
        cv2.putText(view, f"Bird's Eye View (Heading: {heading_deg:.1f}°)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return view

    def _draw_grid(self, view: np.ndarray) -> None:
        """Draw depth grid lines with labels (legacy mode)."""
        usable_height = self.height - 50

        # Draw horizontal lines at appropriate intervals based on max depth
        if self.max_depth_m <= 5:
            interval = 1
        elif self.max_depth_m <= 10:
            interval = 2
        else:
            interval = 5  # 5m intervals for 15m range

        for depth_m in range(interval, int(self.max_depth_m) + 1, interval):
            y = int(self.height - 30 - (depth_m / self.max_depth_m) * usable_height)
            cv2.line(view, (0, y), (self.width, y), (60, 60, 60), 1)
            cv2.putText(view, f"{depth_m}m", (self.width - 30, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

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

        # Draw outer boundary circle (15m)
        boundary_radius = int(self.max_depth_m * self.scale)
        cv2.circle(view, (center_x, center_y), boundary_radius, (80, 80, 80), 2)

    def _draw_fov_cone(
        self,
        view: np.ndarray,
        center_x: int,
        center_y: int,
        heading: float
    ) -> None:
        """Draw FOV cone lines from camera."""
        half_fov = self.fov_rad / 2
        line_length = int(self.max_depth_m * self.scale)

        # Left FOV boundary
        left_angle = heading + half_fov
        left_x = int(center_x + line_length * math.sin(left_angle))
        left_y = int(center_y - line_length * math.cos(left_angle))
        cv2.line(view, (center_x, center_y), (left_x, left_y), (80, 80, 80), 1)

        # Right FOV boundary
        right_angle = heading - half_fov
        right_x = int(center_x + line_length * math.sin(right_angle))
        right_y = int(center_y - line_length * math.cos(right_angle))
        cv2.line(view, (center_x, center_y), (right_x, right_y), (80, 80, 80), 1)

    def _draw_camera_indicator(
        self,
        view: np.ndarray,
        center_x: int,
        center_y: int,
        heading: float
    ) -> None:
        """Draw camera position and heading indicator."""
        # Camera body (circle)
        cv2.circle(view, (center_x, center_y), 10, (255, 255, 255), -1)

        # Heading arrow (points in direction camera is facing)
        arrow_length = 25
        arrow_x = int(center_x + arrow_length * math.sin(heading))
        arrow_y = int(center_y - arrow_length * math.cos(heading))

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
