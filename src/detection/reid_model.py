"""Lightweight Re-ID embedding model for appearance-based object tracking."""

import time
import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Timing stats for performance monitoring
_timing_stats = {
    "coreml_calls": 0,
    "coreml_total_ms": 0.0,
    "handcrafted_calls": 0,
    "handcrafted_total_ms": 0.0,
}


class ReIDEmbedder:
    """
    Appearance embedding model for Re-ID.

    Extracts fixed-size feature vectors from object crops for appearance matching.
    Supports multiple backends:
    - DINOv2: Best quality, 224x224 input
    - MobileNetV3: Faster, 128x128 input
    - Handcrafted: Fallback histogram features
    """

    EMBEDDING_DIM = 256
    # Default crop size - updated based on model
    CROP_SIZE = (224, 224)  # DINOv2 default

    def __init__(
        self,
        embedding_dim: int = 256,
        use_coreml: bool = False,
        model_path: Optional[str] = None
    ):
        """
        Initialize ReID embedder.

        Args:
            embedding_dim: Output embedding dimension.
            use_coreml: If True and model_path provided, use CoreML model.
            model_path: Path to optional CoreML embedding model.
        """
        self.embedding_dim = embedding_dim
        self.use_coreml = use_coreml
        self.model = None

        if use_coreml and model_path:
            self._load_coreml_model(model_path)
        else:
            logger.info(f"ReID embedder initialized (dim={embedding_dim}, handcrafted features)")

    def _load_coreml_model(self, model_path: str) -> None:
        """Load CoreML model if available."""
        try:
            import coremltools as ct
            path = Path(model_path)
            if path.exists():
                self.model = ct.models.MLModel(str(path))
                spec = self.model.get_spec()
                self.input_name = spec.description.input[0].name
                self.output_name = spec.description.output[0].name

                # Detect input size from model spec
                input_spec = spec.description.input[0]
                if hasattr(input_spec.type, 'imageType'):
                    img_type = input_spec.type.imageType
                    input_h = img_type.height
                    input_w = img_type.width
                    self.CROP_SIZE = (input_w, input_h)
                    logger.info(f"ReID model input size: {input_w}x{input_h}")

                model_name = "DINOv2" if "dinov2" in model_path.lower() else "MobileNet"
                logger.info(f"ReID {model_name} CoreML model loaded: {model_path}")
            else:
                logger.warning(f"ReID model not found: {model_path}, using handcrafted features")
        except ImportError:
            logger.warning("coremltools not available, using handcrafted features")
        except Exception as e:
            logger.warning(f"Failed to load ReID model: {e}, using handcrafted features")

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract embedding from object crop.

        Args:
            frame: Full BGR image.
            bbox: Bounding box (x1, y1, x2, y2).

        Returns:
            Normalized embedding vector of shape (embedding_dim,).
        """
        # Crop and resize
        t_start = time.perf_counter()
        crop = self._get_crop(frame, bbox)
        t_crop = (time.perf_counter() - t_start) * 1000

        if self.model is not None:
            t_extract_start = time.perf_counter()
            embedding = self._extract_coreml(crop)
            t_extract = (time.perf_counter() - t_extract_start) * 1000

            _timing_stats["coreml_calls"] += 1
            _timing_stats["coreml_total_ms"] += t_extract
            avg_ms = _timing_stats["coreml_total_ms"] / _timing_stats["coreml_calls"]

            # Log every extraction
            logger.info(
                f"ReID CoreML #{_timing_stats['coreml_calls']}: "
                f"crop={t_crop:.1f}ms, extract={t_extract:.1f}ms, "
                f"total={t_crop + t_extract:.1f}ms (avg={avg_ms:.1f}ms)"
            )

            return embedding
        else:
            t_extract_start = time.perf_counter()
            embedding = self._extract_handcrafted(crop)
            t_extract = (time.perf_counter() - t_extract_start) * 1000

            _timing_stats["handcrafted_calls"] += 1
            _timing_stats["handcrafted_total_ms"] += t_extract
            avg_ms = _timing_stats["handcrafted_total_ms"] / _timing_stats["handcrafted_calls"]

            # Log every extraction
            logger.info(
                f"ReID handcrafted #{_timing_stats['handcrafted_calls']}: "
                f"crop={t_crop:.1f}ms, extract={t_extract:.1f}ms, "
                f"total={t_crop + t_extract:.1f}ms (avg={avg_ms:.1f}ms)"
            )

            return embedding

    def _get_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract and resize object crop."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp to bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        # Extract crop
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            # Return black image if crop is empty
            return np.zeros((self.CROP_SIZE[1], self.CROP_SIZE[0], 3), dtype=np.uint8)

        # Resize to standard size
        crop = cv2.resize(crop, self.CROP_SIZE)
        return crop

    def _extract_handcrafted(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract handcrafted features for Re-ID.

        Combines:
        - Color histograms in HSV space (robust to lighting)
        - Spatial pyramid for location-aware features
        - Edge/texture features via HOG-lite
        """
        features = []

        # Convert to HSV for illumination-invariant color
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 1. Global color histogram (48 dims)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        features.extend([h_hist, s_hist, v_hist])

        # 2. Spatial pyramid color features (3 horizontal strips - typical for people)
        # Upper, middle, lower body regions
        h, w = crop.shape[:2]
        strip_h = h // 3
        for i in range(3):
            y1, y2 = i * strip_h, (i + 1) * strip_h
            strip = hsv[y1:y2, :]
            strip_h_hist = cv2.calcHist([strip], [0], None, [8], [0, 180]).flatten()
            strip_s_hist = cv2.calcHist([strip], [1], None, [8], [0, 256]).flatten()
            features.extend([strip_h_hist, strip_s_hist])

        # 3. Edge/texture features via gradient histogram
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx) * 180 / np.pi + 180  # 0-360

        # Gradient histogram (8 orientation bins)
        for i in range(3):
            y1, y2 = i * strip_h, (i + 1) * strip_h
            strip_mag = mag[y1:y2, :]
            strip_ang = ang[y1:y2, :]

            hist, _ = np.histogram(
                strip_ang.flatten(),
                bins=8,
                range=(0, 360),
                weights=strip_mag.flatten()
            )
            features.append(hist)

        # 4. Simple texture via LBP-like pattern
        # Compute local variance in 4x4 grid
        gh, gw = h // 4, w // 4
        variance_features = []
        for gi in range(4):
            for gj in range(4):
                cell = gray[gi*gh:(gi+1)*gh, gj*gw:(gj+1)*gw]
                variance_features.append(np.std(cell))
        features.append(np.array(variance_features))

        # Concatenate all features
        embedding = np.concatenate([f.flatten() for f in features])

        # Pad or truncate to target dimension
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def _extract_coreml(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract embedding using CoreML MobileNetV3-Small model.

        The model expects 128x128 RGB PIL Image input and outputs
        256-dim L2-normalized embedding.
        """
        try:
            from PIL import Image

            # Resize crop to model input size if needed
            h, w = crop.shape[:2]
            if h != self.CROP_SIZE[1] or w != self.CROP_SIZE[0]:
                crop = cv2.resize(crop, self.CROP_SIZE)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Run inference
            predictions = self.model.predict({self.input_name: pil_img})

            # Get embedding from known output name or first array output
            if hasattr(self, 'output_name') and self.output_name in predictions:
                embedding = np.asarray(predictions[self.output_name]).flatten()
            else:
                # Fallback: find first array output
                for value in predictions.values():
                    if isinstance(value, np.ndarray):
                        embedding = value.flatten()
                        break
                else:
                    return self._extract_handcrafted(crop)

            # L2 normalize (should already be normalized by model, but ensure)
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"CoreML extraction failed: {e}, using handcrafted")
            return self._extract_handcrafted(crop)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding (normalized).
            emb2: Second embedding (normalized).

        Returns:
            Similarity score in [-1, 1], higher = more similar.
        """
        # Embeddings are already L2 normalized, so dot product = cosine similarity
        return float(np.dot(emb1, emb2))

    @staticmethod
    def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine distance (1 - similarity).

        Returns:
            Distance in [0, 2], lower = more similar.
        """
        return 1.0 - ReIDEmbedder.cosine_similarity(emb1, emb2)
