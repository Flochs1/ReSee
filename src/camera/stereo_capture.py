"""Stereo camera capture for ELP cameras."""

import cv2
import numpy as np
import threading
from queue import Queue, Empty, Full
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StereoCameraError(Exception):
    """Exception raised for stereo camera errors."""
    pass


class StereoCamera:
    """
    Manages ELP stereo camera capture with auto-detection.

    Supports multiple ELP camera configurations:
    - Single device with side-by-side output
    - Dual devices (separate left/right cameras)
    - Single device with retrieve() flags
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 10,
        device_mode: str = "auto",
        device_indices: Tuple[int, int] = (0, 1),
        buffer_size: int = 3
    ):
        """
        Initialize stereo camera.

        Args:
            width: Width per camera (not combined).
            height: Height per camera.
            fps: Target frames per second.
            device_mode: Camera mode (auto, single, dual, retrieve).
            device_indices: Device indices for dual mode (left, right).
            buffer_size: Frame buffer queue size.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.device_mode = device_mode
        self.device_indices = device_indices
        self.buffer_size = buffer_size

        # Camera capture objects
        self.cap_left: Optional[cv2.VideoCapture] = None
        self.cap_right: Optional[cv2.VideoCapture] = None
        self.detected_mode: Optional[str] = None

        # Threading
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

    def detect_camera_mode(self) -> str:
        """
        Auto-detect which camera mode is active.

        Returns:
            Detected mode: "single", "dual", or "retrieve".

        Raises:
            StereoCameraError: If no cameras are detected.
        """
        logger.info("Auto-detecting ELP camera configuration...")

        # Try dual mode first (most reliable)
        left_idx, right_idx = self.device_indices
        cap_left = cv2.VideoCapture(left_idx)
        cap_right = cv2.VideoCapture(right_idx)

        if cap_left.isOpened() and cap_right.isOpened():
            # Check if we can read from both
            ret_left, _ = cap_left.read()
            ret_right, _ = cap_right.read()

            if ret_left and ret_right:
                logger.info(f"Detected DUAL camera mode (indices {left_idx}, {right_idx})")
                cap_left.release()
                cap_right.release()
                return "dual"

        cap_left.release()
        cap_right.release()

        # Try single mode with side-by-side
        cap_single = cv2.VideoCapture(left_idx)

        if cap_single.isOpened():
            ret, frame = cap_single.read()

            if ret:
                # Check if frame width suggests side-by-side (approximately 2x expected width)
                frame_width = frame.shape[1]
                expected_combined_width = self.width * 2

                if abs(frame_width - expected_combined_width) < 100:
                    logger.info(f"Detected SINGLE camera mode with side-by-side output (width {frame_width})")
                    cap_single.release()
                    return "single"

                # Otherwise, try retrieve mode
                logger.info("Detected RETRIEVE mode (single device, separate retrieve)")
                cap_single.release()
                return "retrieve"

        cap_single.release()

        # No cameras detected
        raise StereoCameraError(
            "No cameras detected. Please check:\n"
            "  1. Camera is connected via USB\n"
            "  2. Camera permissions are granted\n"
            "  3. Device indices in config are correct"
        )

    def open(self) -> None:
        """
        Open camera devices and start capture.

        Raises:
            StereoCameraError: If cameras cannot be opened.
        """
        with self.lock:
            # Detect mode if auto
            if self.device_mode == "auto":
                self.detected_mode = self.detect_camera_mode()
            else:
                self.detected_mode = self.device_mode
                logger.info(f"Using manual camera mode: {self.detected_mode}")

            # Open cameras based on detected mode
            if self.detected_mode == "dual":
                self._open_dual_mode()
            elif self.detected_mode == "single":
                self._open_single_mode()
            elif self.detected_mode == "retrieve":
                self._open_retrieve_mode()
            else:
                raise StereoCameraError(f"Unknown camera mode: {self.detected_mode}")

            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            logger.info("Stereo camera opened successfully")

    def _open_dual_mode(self) -> None:
        """Open cameras in dual mode (two separate devices)."""
        left_idx, right_idx = self.device_indices

        self.cap_left = cv2.VideoCapture(left_idx)
        self.cap_right = cv2.VideoCapture(right_idx)

        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            raise StereoCameraError(f"Failed to open dual cameras (indices {left_idx}, {right_idx})")

        # Configure both cameras
        for cap in [self.cap_left, self.cap_right]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.debug(f"Dual mode: Left={left_idx}, Right={right_idx}")

    def _open_single_mode(self) -> None:
        """Open camera in single mode (side-by-side output)."""
        left_idx = self.device_indices[0]

        self.cap_left = cv2.VideoCapture(left_idx)

        if not self.cap_left.isOpened():
            raise StereoCameraError(f"Failed to open camera (index {left_idx})")

        # Configure for side-by-side resolution (2x width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap_left.set(cv2.CAP_PROP_FPS, self.fps)

        logger.debug(f"Single mode: Device={left_idx}")

    def _open_retrieve_mode(self) -> None:
        """Open camera in retrieve mode (single device, separate retrieve)."""
        left_idx = self.device_indices[0]

        self.cap_left = cv2.VideoCapture(left_idx)

        if not self.cap_left.isOpened():
            raise StereoCameraError(f"Failed to open camera (index {left_idx})")

        # Configure camera
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap_left.set(cv2.CAP_PROP_FPS, self.fps)

        logger.debug(f"Retrieve mode: Device={left_idx}")

    def _capture_loop(self) -> None:
        """Continuous capture loop (runs in background thread)."""
        logger.debug("Capture thread started")

        while self.running:
            try:
                # Capture based on mode
                if self.detected_mode == "dual":
                    left_frame, right_frame = self._capture_dual()
                elif self.detected_mode == "single":
                    left_frame, right_frame = self._capture_single()
                elif self.detected_mode == "retrieve":
                    left_frame, right_frame = self._capture_retrieve()
                else:
                    logger.error(f"Unknown capture mode: {self.detected_mode}")
                    break

                if left_frame is None or right_frame is None:
                    logger.warning("Failed to capture frames")
                    continue

                # Add to queue (drop old frames if full)
                try:
                    self.frame_queue.put((left_frame, right_frame), block=False)
                except Full:
                    # Queue is full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((left_frame, right_frame), block=False)
                    except (Empty, Full):
                        pass

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                break

        logger.debug("Capture thread stopped")

    def _capture_dual(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture frames in dual mode."""
        ret_left, left_frame = self.cap_left.read()
        ret_right, right_frame = self.cap_right.read()

        if not ret_left or not ret_right:
            return None, None

        return left_frame, right_frame

    def _capture_single(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture frames in single mode (split side-by-side)."""
        ret, combined_frame = self.cap_left.read()

        if not ret:
            return None, None

        # Split side-by-side frame
        height, width = combined_frame.shape[:2]
        mid = width // 2

        left_frame = combined_frame[:, :mid]
        right_frame = combined_frame[:, mid:]

        return left_frame, right_frame

    def _capture_retrieve(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture frames in retrieve mode."""
        # Grab both frames
        self.cap_left.grab()

        # Retrieve left frame
        ret_left, left_frame = self.cap_left.retrieve(0)

        # Retrieve right frame (if supported)
        ret_right, right_frame = self.cap_left.retrieve(1)

        if not ret_left:
            return None, None

        # If right frame not available, try grabbing again
        if not ret_right:
            self.cap_left.grab()
            ret_right, right_frame = self.cap_left.retrieve(0)

        if not ret_right:
            # Fallback: duplicate left frame
            right_frame = left_frame.copy()

        return left_frame, right_frame

    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get latest frame from queue.

        Args:
            timeout: Timeout in seconds.

        Returns:
            (left_frame, right_frame) tuple, or None if timeout.
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        """Close cameras and stop capture thread."""
        with self.lock:
            self.running = False

            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)

            # Release cameras
            if self.cap_left:
                self.cap_left.release()
                self.cap_left = None

            if self.cap_right:
                self.cap_right.release()
                self.cap_right = None

            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break

            logger.info("Stereo camera closed")

    def is_opened(self) -> bool:
        """Check if camera is opened and running."""
        with self.lock:
            return self.running and (
                (self.cap_left and self.cap_left.isOpened()) or
                (self.cap_right and self.cap_right.isOpened())
            )

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
