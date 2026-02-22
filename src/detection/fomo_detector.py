"""FOMO-style object detection - center point detection with minimal overhead."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Represents a single object detection with center point."""
    cx: int  # Center x
    cy: int  # Center y
    confidence: float
    class_id: int
    class_name: str
    depth_m: Optional[float] = None


class FOMODetector:
    """
    FOMO-style detector using YOLO-nano for speed.

    Outputs center points of detected objects (not full bounding boxes)
    for minimal latency. Uses frame skipping to reduce compute load.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        input_size: int = 320
    ):
        """
        Initialize FOMO detector.

        Args:
            model_name: YOLO nano model name.
            confidence_threshold: Minimum confidence for detections.
            device: Device to use ("mps", "cuda", "cpu", or None for auto).
            input_size: Input image size for inference (smaller = faster).
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = device
        self.input_size = input_size
        self._last_detections: List[Detection] = []

        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load YOLO model with appropriate device."""
        try:
            from ultralytics import YOLO
            import torch

            # Determine device
            if self.device is None:
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    logger.info("FOMO: Using MPS (Metal) GPU")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info("FOMO: Using CUDA GPU")
                else:
                    self.device = "cpu"
                    logger.info("FOMO: Using CPU")

            logger.info(f"FOMO: Loading model {model_name} (input: {self.input_size}px)")
            self.model = YOLO(model_name)

            # Warmup with small image
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, imgsz=self.input_size, verbose=False)

            logger.info(f"FOMO: Model ready on {self.device}")

        except ImportError as e:
            logger.error(f"Failed to import ultralytics: {e}")
            logger.error("Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load FOMO model: {e}")
            raise

    def detect(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> List[Detection]:
        """
        Run detection and return center points.

        Args:
            frame: BGR image.
            depth_map: Optional depth map (meters) for depth lookup.

        Returns:
            List of Detection objects with center coordinates.
        """
        if self.model is None:
            return []

        h, w = frame.shape[:2]

        # Run inference with smaller input size for speed
        results = self.model.predict(
            frame,
            device=self.device,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            verbose=False,
            half=self.device in ("cuda", "mps")  # FP16 for GPU
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # Batch extract all data at once (faster than per-box)
            if len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Get depth at center point
                depth_m = None
                if depth_map is not None:
                    depth_m = self._get_center_depth(depth_map, cx, cy, w, h)

                detections.append(Detection(
                    cx=cx,
                    cy=cy,
                    confidence=float(confs[i]),
                    class_id=int(cls_ids[i]),
                    class_name=self.model.names[cls_ids[i]],
                    depth_m=depth_m
                ))

        self._last_detections = detections
        return detections

    def get_cached(self) -> List[Detection]:
        """Return last detections (for frame skipping)."""
        return self._last_detections

    def _get_center_depth(
        self,
        depth_map: np.ndarray,
        cx: int, cy: int,
        frame_w: int, frame_h: int
    ) -> Optional[float]:
        """Get depth at center point with small neighborhood sampling."""
        dh, dw = depth_map.shape[:2]

        # Scale center to depth map coordinates
        dx = int(cx * dw / frame_w)
        dy = int(cy * dh / frame_h)

        # Sample 3x3 neighborhood
        r = 2
        x1 = max(0, dx - r)
        y1 = max(0, dy - r)
        x2 = min(dw, dx + r + 1)
        y2 = min(dh, dy + r + 1)

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[(roi > 0.1) & (roi < 50)]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    def draw_centers(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        radius: int = 20,
        show_label: bool = True,
        show_depth: bool = True
    ) -> np.ndarray:
        """
        Draw center points on frame (FOMO-style visualization).

        Args:
            frame: BGR image to draw on (modified in place for speed).
            detections: List of Detection objects.
            radius: Circle radius for center points.
            show_label: Show class name.
            show_depth: Show depth value.

        Returns:
            Frame with drawn detections.
        """
        for det in detections:
            # Bright, high-contrast colors
            if det.depth_m is not None:
                t = np.clip((det.depth_m - 0.5) / 4.5, 0, 1)
                # Cyan (near) -> Magenta (far) - more visible than green/blue
                color = (int(255 * t), int(255 * (1 - t)), 255)
            else:
                color = (0, 255, 255)  # Bright cyan

            # Draw thick outer ring (black outline for contrast)
            cv2.circle(frame, (det.cx, det.cy), radius + 3, (0, 0, 0), 4)
            # Draw colored ring
            cv2.circle(frame, (det.cx, det.cy), radius, color, 4)
            # Draw crosshair
            cv2.line(frame, (det.cx - radius, det.cy), (det.cx + radius, det.cy), color, 2)
            cv2.line(frame, (det.cx, det.cy - radius), (det.cx, det.cy + radius), color, 2)

            # Draw label with large font
            if show_label:
                if det.depth_m is not None and show_depth:
                    label = f"{det.class_name} {det.depth_m:.1f}m"
                else:
                    label = det.class_name

                # Black outline for readability
                cv2.putText(
                    frame,
                    label,
                    (det.cx + radius + 8, det.cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    4
                )
                # Colored text
                cv2.putText(
                    frame,
                    label,
                    (det.cx + radius + 8, det.cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        return frame
