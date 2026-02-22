"""HITNet ONNX backend for stereo depth estimation.

HITNet (Hierarchical Iterative Tile Refinement Network) is Google's
fast and accurate stereo matching model (CVPR 2021).

This backend uses ONNX Runtime with CoreML acceleration on Apple Silicon.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Any

import numpy as np

# Check ONNX Runtime availability
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    AVAILABLE_PROVIDERS = ort.get_available_providers()
except ImportError:
    ONNX_AVAILABLE = False
    ort = None  # type: ignore
    AVAILABLE_PROVIDERS = []

# Global flag for HITNet availability
HITNET_AVAILABLE = False
_hitnet_error_message = ""

# Check for CoreML (Apple Silicon GPU)
HAS_COREML = "CoreMLExecutionProvider" in AVAILABLE_PROVIDERS


def get_default_model_path() -> Path:
    """Get the default path to the HITNet ONNX model."""
    # Look in project root/models/hitnet/
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "models" / "hitnet" / "hitnet.onnx"


class HITNetBackend:
    """HITNet ONNX stereo matching backend.

    Uses ONNX Runtime with CoreML acceleration on Apple Silicon Macs.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        target_height: int = 480,
        target_width: int = 640,
        use_coreml: bool = False,  # Disabled by default - CoreML is very slow for this model
    ):
        """
        Initialize HITNet backend.

        Args:
            model_path: Path to HITNet ONNX model file.
                       If None, uses default location (models/hitnet/hitnet.onnx).
            target_height: Height to resize inputs to (must match model).
            target_width: Width to resize inputs to (must match model).
            use_coreml: Whether to use CoreML acceleration (Apple Silicon).
                       NOTE: CoreML is disabled by default because many HITNet
                       operations aren't well-supported, causing 50-100x slowdown.
                       CPU mode with ONNX Runtime is typically faster for this model.

        Raises:
            ImportError: If ONNX Runtime is not installed.
            FileNotFoundError: If model file not found.
        """
        global HITNET_AVAILABLE, _hitnet_error_message

        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime is required for HITNet. "
                "Run: pip install onnxruntime"
            )

        self.model_path = Path(model_path) if model_path else get_default_model_path()
        self.target_height = target_height
        self.target_width = target_width
        self.use_coreml = use_coreml and HAS_COREML
        self.session = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the HITNet ONNX model."""
        global HITNET_AVAILABLE, _hitnet_error_message

        if not self.model_path.exists():
            _hitnet_error_message = f"HITNet model not found at {self.model_path}"
            raise FileNotFoundError(
                f"HITNet model not found at {self.model_path}. "
                f"Run ./setup_depth.sh to download the model."
            )

        print(f"Loading HITNet model from {self.model_path}...")

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Select providers
        if self.use_coreml:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            print("ONNX Runtime: Using CoreML (Apple Silicon GPU)")
        else:
            providers = ["CPUExecutionProvider"]
            print("ONNX Runtime: Using CPU")

        # Create inference session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        print(f"Model input: {self.input_name} {self.input_shape}")
        print(f"Model output: {self.output_name}")

        HITNET_AVAILABLE = True
        print("HITNet model loaded successfully")

    def _preprocess(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Preprocess stereo pair for inference.

        Args:
            left: Left BGR image (H, W, 3)
            right: Right BGR image (H, W, 3)

        Returns:
            Tuple of (input_tensor, original_size)
        """
        import cv2

        original_size = (left.shape[0], left.shape[1])

        # Convert BGR to RGB
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        left_resized = cv2.resize(
            left_rgb, (self.target_width, self.target_height)
        )
        right_resized = cv2.resize(
            right_rgb, (self.target_width, self.target_height)
        )

        # Normalize to [0, 1]
        left_norm = left_resized.astype(np.float32) / 255.0
        right_norm = right_resized.astype(np.float32) / 255.0

        # Stack or concatenate based on model expectations
        # HITNet typically expects (B, 6, H, W) - concatenated RGB channels
        # or (B, 2, 3, H, W) - stacked stereo pair

        # Common format: (B, 6, H, W) - channels first, concatenated
        left_chw = np.transpose(left_norm, (2, 0, 1))  # (3, H, W)
        right_chw = np.transpose(right_norm, (2, 0, 1))  # (3, H, W)

        # Check expected input shape
        if self.input_shape and len(self.input_shape) == 4:
            expected_channels = self.input_shape[1]
            if expected_channels == 6:
                # Concatenate: (6, H, W)
                combined = np.concatenate([left_chw, right_chw], axis=0)
            elif expected_channels == 3:
                # Some models expect only left image initially
                combined = left_chw
            else:
                # Default to concatenation
                combined = np.concatenate([left_chw, right_chw], axis=0)
        else:
            # Default: concatenate
            combined = np.concatenate([left_chw, right_chw], axis=0)

        # Add batch dimension
        input_tensor = np.expand_dims(combined, axis=0).astype(np.float32)

        return input_tensor, original_size

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from stereo pair.

        Args:
            left: Left camera BGR image (H, W, 3)
            right: Right camera BGR image (H, W, 3)

        Returns:
            Disparity map as float32 array (H, W)
        """
        import cv2

        # Preprocess
        input_tensor, original_size = self._preprocess(left, right)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        disparity = outputs[0]

        # Remove batch dimension and channel dimension if present
        if disparity.ndim == 4:
            disparity = disparity[0, 0]  # (B, 1, H, W) -> (H, W)
        elif disparity.ndim == 3:
            disparity = disparity[0]  # (B, H, W) -> (H, W)

        # Ensure positive values
        disparity = np.abs(disparity).astype(np.float32)

        # Resize back to original size
        if disparity.shape[:2] != original_size:
            # Scale disparity values proportionally
            scale_x = original_size[1] / disparity.shape[1]
            disparity = cv2.resize(
                disparity,
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            disparity *= scale_x

        return disparity

    def shutdown(self) -> None:
        """Release resources."""
        global HITNET_AVAILABLE
        self.session = None
        HITNET_AVAILABLE = False

    @staticmethod
    def get_backend_name() -> str:
        """Return backend identifier."""
        return "hitnet"

    def get_backend_info(self) -> str:
        """Return human-readable backend description."""
        if not ONNX_AVAILABLE:
            return "HITNet (ONNX Runtime not installed)"

        if self.use_coreml:
            return "HITNet (ONNX + CoreML GPU)"
        else:
            return "HITNet (ONNX CPU)"


def check_hitnet_available() -> tuple[bool, str]:
    """
    Check if HITNet backend is available.

    Returns:
        Tuple of (available, message)
    """
    if not ONNX_AVAILABLE:
        return False, "ONNX Runtime not installed"

    model_path = get_default_model_path()
    if not model_path.exists():
        return False, f"Model not found at {model_path}"

    return True, "HITNet available"
