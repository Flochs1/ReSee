"""CoreML-based YOLO detector - runs without PyTorch to avoid OpenCV conflicts."""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Single object detection result."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float


# COCO class names for YOLOv8
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class CoreMLDetector:
    """
    YOLOv8 detector using CoreML for inference.

    Does NOT require PyTorch at runtime, avoiding the OpenCV performance conflict.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.mlpackage",
        confidence_threshold: float = 0.5,
        input_size: int = 1280
    ):
        """
        Initialize CoreML detector.

        Args:
            model_path: Path to .mlpackage model file.
            confidence_threshold: Minimum confidence for detections.
            input_size: Model input size (default 640 for YOLOv8).
        """
        self.conf_threshold = confidence_threshold
        self.input_size = input_size

        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "CoreML detector requires coremltools. "
                "Install with: pip install coremltools"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"CoreML model not found: {model_path}\n"
                f"Export it first with: python scripts/export_coreml.py"
            )

        logger.info(f"Loading CoreML model: {model_path}")

        # Use Neural Engine + GPU for fast inference
        self.model = ct.models.MLModel(
            str(model_path),
            compute_units=ct.ComputeUnit.ALL  # Neural Engine + GPU + CPU
        )

        # Get model spec to understand input/output
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        logger.info(f"CoreML detector ready (input={self.input_name}, conf={confidence_threshold}, compute=ALL)")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of Detection objects.
        """
        import time
        orig_h, orig_w = frame.shape[:2]

        # Preprocess: resize and convert BGR->RGB
        t_pre = time.time()
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CoreML expects PIL Image or numpy array depending on model
        # YOLOv8 CoreML models typically expect normalized float32
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        pre_ms = (time.time() - t_pre) * 1000

        # Run inference
        t_infer = time.time()
        try:
            from PIL import Image
            # Some CoreML models prefer PIL input
            pil_img = Image.fromarray((img[0].transpose(1, 2, 0) * 255).astype(np.uint8))
            predictions = self.model.predict({self.input_name: pil_img})
        except Exception:
            # Try numpy input
            predictions = self.model.predict({self.input_name: img})
        infer_ms = (time.time() - t_infer) * 1000

        # Parse YOLO output
        t_post = time.time()
        # YOLOv8 CoreML output format varies - try common formats
        detections = self._parse_predictions(predictions, orig_w, orig_h)
        post_ms = (time.time() - t_post) * 1000

        # Log timing periodically
        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 50 == 0:
            logger.info(f"Detector timing: preprocess={pre_ms:.1f}ms, inference={infer_ms:.1f}ms, postprocess={post_ms:.1f}ms")

        return detections

    def _parse_predictions(
        self,
        predictions: dict,
        orig_w: int,
        orig_h: int
    ) -> List[Detection]:
        """Parse CoreML predictions into Detection objects."""
        detections = []

        # YOLOv8 CoreML with NMS typically outputs:
        # - 'coordinates': bounding boxes
        # - 'confidence': class confidences
        # Or combined output tensor

        # Try to find the output
        if 'coordinates' in predictions and 'confidence' in predictions:
            coords = np.array(predictions['coordinates'])
            confs = np.array(predictions['confidence'])

            for i in range(len(coords)):
                if len(confs[i]) == 0:
                    continue

                max_conf = np.max(confs[i])
                if max_conf < self.conf_threshold:
                    continue

                class_id = int(np.argmax(confs[i]))
                x_center, y_center, w, h = coords[i]

                # Convert to pixel coordinates
                x1 = int((x_center - w/2) * orig_w)
                y1 = int((y_center - h/2) * orig_h)
                x2 = int((x_center + w/2) * orig_w)
                y2 = int((y_center + h/2) * orig_h)

                # Clamp to image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(max_conf)
                ))
        else:
            # Handle raw YOLO output tensor
            for key, value in predictions.items():
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    detections = self._parse_yolo_tensor(value, orig_w, orig_h)
                    break

        return detections

    def _parse_yolo_tensor(
        self,
        output: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> List[Detection]:
        """Parse raw YOLO output tensor."""
        detections = []

        # YOLOv8 output: [batch, 84, num_detections] or [batch, num_detections, 84]
        # 84 = 4 (bbox) + 80 (classes)
        if output.ndim == 3:
            output = output[0]  # Remove batch dim

        if output.shape[0] == 84:
            output = output.T  # Transpose to [num_detections, 84]

        for det in output:
            if len(det) < 84:
                continue

            x_center, y_center, w, h = det[:4]
            class_scores = det[4:]

            max_score = np.max(class_scores)
            if max_score < self.conf_threshold:
                continue

            class_id = int(np.argmax(class_scores))

            # Scale to original image
            scale_x = orig_w / self.input_size
            scale_y = orig_h / self.input_size

            x1 = int((x_center - w/2) * scale_x)
            y1 = int((y_center - h/2) * scale_y)
            x2 = int((x_center + w/2) * scale_x)
            y2 = int((y_center + h/2) * scale_y)

            # Clamp
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                confidence=float(max_score)
            ))

        # Apply NMS
        detections = self._nms(detections, iou_threshold=0.5)

        return detections

    @staticmethod
    def _nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Non-maximum suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if d.class_id != best.class_id or
                CoreMLDetector._iou(best.bbox, d.bbox) < iou_threshold
            ]

        return keep

    @staticmethod
    def _iou(box1: tuple, box2: tuple) -> float:
        """Calculate IoU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
