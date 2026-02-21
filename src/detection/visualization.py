"""Visualization utilities for object detection overlays."""

import cv2
import numpy as np
from typing import List

from .object_tracker import TrackedObject


def draw_tracks(
    frame: np.ndarray,
    tracks: List[TrackedObject],
    show_depth: bool = True,
    show_speed: bool = True
) -> np.ndarray:
    """
    Draw tracked objects with depth and closing speed info.

    Args:
        frame: BGR image to draw on (modified in place).
        tracks: List of tracked objects.
        show_depth: Whether to show depth in labels.
        show_speed: Whether to show closing speed in labels.

    Returns:
        Frame with overlays drawn.
    """
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        depth = track.get_current_depth()
        closing_speed = track.closing_speed

        # Color based on depth (visual debugging)
        if depth < 0:
            color = (128, 128, 128)  # Gray - no depth
        elif depth < 1.0:
            color = (0, 0, 255)  # Red - very close
        elif depth < 2.0:
            color = (0, 128, 255)  # Orange - close
        elif depth < 3.0:
            color = (0, 200, 255)  # Yellow-orange - medium
        else:
            color = (0, 255, 0)  # Green - far

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build label
        label_parts = [f"#{track.track_id} {track.class_name}"]

        if show_depth and depth > 0:
            label_parts.append(f"{depth:.1f}m")

        if show_speed and abs(closing_speed) > 0.1:
            # Show + for approaching, - for receding
            label_parts.append(f"{closing_speed:+.1f}m/s")

        label = " ".join(label_parts)

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = y1 - 10 if y1 > 30 else y2 + 20
        label_x = x1

        # Background rectangle
        cv2.rectangle(
            frame,
            (label_x, label_y - text_h - 4),
            (label_x + text_w + 4, label_y + 4),
            color,
            -1
        )

        # Text (white for contrast)
        cv2.putText(
            frame,
            label,
            (label_x + 2, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame


def draw_detection_stats(
    frame: np.ndarray,
    tracks: List[TrackedObject],
    position: str = "top-right"
) -> np.ndarray:
    """
    Draw summary statistics overlay.

    Args:
        frame: BGR image to draw on.
        tracks: List of tracked objects.
        position: Where to place stats ("top-right", "top-left", "bottom-right", "bottom-left").

    Returns:
        Frame with stats overlay.
    """
    h, w = frame.shape[:2]

    # Count objects by class
    class_counts = {}
    for track in tracks:
        class_counts[track.class_name] = class_counts.get(track.class_name, 0) + 1

    # Build stats text
    lines = [f"Tracked: {len(tracks)}"]
    for cls_name, count in sorted(class_counts.items()):
        lines.append(f"  {cls_name}: {count}")

    # Calculate position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    padding = 10

    max_text_w = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    box_h = len(lines) * line_height + padding * 2
    box_w = max_text_w + padding * 2

    if "right" in position:
        box_x = w - box_w - 10
    else:
        box_x = 10

    if "bottom" in position:
        box_y = h - box_h - 10
    else:
        box_y = 10

    # Draw background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw text
    for i, line in enumerate(lines):
        y = box_y + padding + (i + 1) * line_height - 5
        cv2.putText(frame, line, (box_x + padding, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame
