"""YOLO detector using ultralytics (requires PyTorch)."""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Single object detection result."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float


class YOLODetector:
    """
    YOLOv8 detector using ultralytics.

    WARNING: Loading PyTorch causes OpenCV StereoSGBM to slow down ~3x.
    Use CoreMLDetector instead for better performance with depth estimation.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "auto"
    ):
        self.conf_threshold = confidence_threshold

        try:
            from ultralytics import YOLO
            import torch
        except ImportError:
            raise ImportError("Requires: pip install ultralytics torch")

        # Select device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLO model on {self.device}")
        self.model = YOLO(model_path)

        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False)
        logger.info("YOLO detector ready")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on frame."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                class_name = result.names[class_id]

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence
                ))

        return detections
