"""Stereo depth estimation with automatic backend selection.

Supports two backends:
1. HITNet (TensorFlow) - GPU accelerated via Metal on Apple Silicon
2. StereoSGBM (OpenCV) - CPU fallback that works everywhere
"""

import time
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

from .colormap import apply_colormap
from .backends.sgbm import SGBMBackend

# Try to import HITNet backend
try:
    from .backends.hitnet import HITNetBackend, check_hitnet_available
    HITNET_IMPORT_OK = True
except ImportError:
    HITNET_IMPORT_OK = False
    HITNetBackend = None

    def check_hitnet_available():
        return False, "TensorFlow not installed"


BackendType = Literal["auto", "hitnet", "sgbm"]


@dataclass
class DepthResult:
    """Result from stereo depth estimation."""

    depth_map: np.ndarray  # Float32 disparity values (H, W)
    computation_time_ms: float
    backend: str  # Which backend produced this result

    def to_colormap(self, cmap: str = "inferno") -> np.ndarray:
        """
        Convert to BGR colormap for display.

        Args:
            cmap: Colormap name (inferno, magma, plasma, viridis, jet, turbo, hot, bone)

        Returns:
            BGR image (H, W, 3) suitable for OpenCV display
        """
        return apply_colormap(self.depth_map, colormap=cmap)


class DepthProcessor:
    """Stereo depth estimation with automatic backend selection.

    Attempts to use HITNet (TensorFlow + Metal GPU) first, then falls
    back to OpenCV StereoSGBM if unavailable.

    Example:
        processor = DepthProcessor(backend="auto")
        result = processor.compute(left_frame, right_frame)
        colored = result.to_colormap("inferno")
    """

    def __init__(
        self,
        backend: BackendType = "auto",
        # SGBM parameters
        num_disparities: int = 128,
        block_size: int = 5,
        # HITNet parameters
        hitnet_height: int = 480,
        hitnet_width: int = 640,
    ):
        """
        Initialize the stereo depth processor.

        Args:
            backend: Backend to use - "auto", "hitnet", or "sgbm"
            num_disparities: SGBM max disparity (must be divisible by 16)
            block_size: SGBM block size (odd number, 3-11)
            hitnet_height: Input height for HITNet model
            hitnet_width: Input width for HITNet model

        Raises:
            RuntimeError: If requested backend is unavailable
        """
        self.requested_backend = backend
        self.active_backend_name: str = ""
        self._backend = None

        # Store parameters
        self._num_disparities = num_disparities
        self._block_size = block_size
        self._hitnet_height = hitnet_height
        self._hitnet_width = hitnet_width

        self._initialize_backend(backend)

    def _initialize_backend(self, backend: BackendType) -> None:
        """Initialize the appropriate backend."""
        if backend == "hitnet":
            self._init_hitnet(fallback=False)
        elif backend == "sgbm":
            self._init_sgbm()
        else:  # auto
            # Try HITNet first, fall back to SGBM
            if not self._init_hitnet(fallback=True):
                self._init_sgbm()

    def _init_hitnet(self, fallback: bool) -> bool:
        """
        Try to initialize HITNet backend.

        Args:
            fallback: If True, return False on failure instead of raising

        Returns:
            True if successful, False if failed and fallback=True

        Raises:
            RuntimeError: If fallback=False and initialization fails
        """
        if not HITNET_IMPORT_OK:
            msg = "TensorFlow not installed. Run: pip install tensorflow tensorflow-metal"
            if fallback:
                print(f"HITNet unavailable: {msg}")
                return False
            raise RuntimeError(msg)

        available, error_msg = check_hitnet_available()
        if not available:
            if fallback:
                print(f"HITNet unavailable: {error_msg}")
                return False
            raise RuntimeError(f"HITNet unavailable: {error_msg}")

        try:
            self._backend = HITNetBackend(
                target_height=self._hitnet_height,
                target_width=self._hitnet_width,
            )
            self.active_backend_name = "hitnet"
            print(f"Depth backend: {self._backend.get_backend_info()}")
            return True
        except Exception as e:
            if fallback:
                print(f"HITNet initialization failed: {e}")
                return False
            raise RuntimeError(f"HITNet initialization failed: {e}")

    def _init_sgbm(self) -> None:
        """Initialize SGBM backend."""
        self._backend = SGBMBackend(
            num_disparities=self._num_disparities,
            block_size=self._block_size,
        )
        self.active_backend_name = "sgbm"
        print(f"Depth backend: {self._backend.get_backend_info()}")

    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self.active_backend_name

    @property
    def backend_info(self) -> str:
        """Get human-readable info about the active backend."""
        if self._backend is not None:
            return self._backend.get_backend_info()
        return "No backend initialized"

    def compute(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> DepthResult:
        """
        Compute disparity map from stereo image pair.

        This uses true stereo matching - analyzing the pixel displacement
        between left and right images to determine depth.

        Args:
            left: Left camera BGR image (H, W, 3)
            right: Right camera BGR image (H, W, 3)

        Returns:
            DepthResult with disparity map, timing, and backend info
        """
        start_time = time.time()

        # Run stereo matching via active backend
        disparity_map = self._backend.compute(left, right)

        # Calculate computation time
        computation_time_ms = (time.time() - start_time) * 1000

        return DepthResult(
            depth_map=disparity_map,
            computation_time_ms=computation_time_ms,
            backend=self.active_backend_name,
        )

    def shutdown(self) -> None:
        """Release backend resources."""
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None
        print("Depth processor shutdown complete")


def check_backends_available() -> dict[str, tuple[bool, str]]:
    """
    Check availability of all backends.

    Returns:
        Dict mapping backend name to (available, message) tuples
    """
    results = {}

    # Check HITNet
    if HITNET_IMPORT_OK:
        results["hitnet"] = check_hitnet_available()
    else:
        results["hitnet"] = (False, "TensorFlow not installed")

    # SGBM is always available (only needs OpenCV)
    results["sgbm"] = (True, "OpenCV StereoSGBM ready")

    return results


def get_best_available_backend() -> str:
    """
    Get the name of the best available backend.

    Returns:
        "hitnet" if available, otherwise "sgbm"
    """
    backends = check_backends_available()

    if backends["hitnet"][0]:
        return "hitnet"
    return "sgbm"
