"""Frame processing for stereo camera streams."""

import cv2
import numpy as np
import base64
from typing import Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Processes stereo camera frames for transmission."""

    def __init__(
        self,
        jpeg_quality: int = 85,
        target_width: int = 1920,
        target_height: int = 1080
    ):
        """
        Initialize frame processor.

        Args:
            jpeg_quality: JPEG compression quality (1-100, higher = better quality).
            target_width: Target width for each frame.
            target_height: Target height for each frame.
        """
        self.jpeg_quality = jpeg_quality
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self.target_width = target_width
        self.target_height = target_height

    def combine_frames(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        target_height: int = 1080,
        target_width: int = 1920
    ) -> np.ndarray:
        """
        Combine left and right frames side-by-side.

        If frames have different dimensions, they will be resized to match
        the target resolution before combining.

        Args:
            left_frame: Left camera frame.
            right_frame: Right camera frame.
            target_height: Target height for each frame.
            target_width: Target width for each frame.

        Returns:
            Combined frame (left | right) at target_width*2 x target_height.
        """
        # Resize frames if they don't match target dimensions
        left_h, left_w = left_frame.shape[:2]
        right_h, right_w = right_frame.shape[:2]

        if left_h != target_height or left_w != target_width:
            logger.debug(f"Resizing left frame from {left_w}x{left_h} to {target_width}x{target_height}")
            left_frame = cv2.resize(left_frame, (target_width, target_height))

        if right_h != target_height or right_w != target_width:
            logger.debug(f"Resizing right frame from {right_w}x{right_h} to {target_width}x{target_height}")
            right_frame = cv2.resize(right_frame, (target_width, target_height))

        # Combine horizontally
        combined = np.hstack((left_frame, right_frame))

        return combined

    def encode_jpeg(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Encode frame to JPEG format.

        Args:
            frame: Input frame (BGR or grayscale).

        Returns:
            JPEG-encoded bytes, or None if encoding fails.
        """
        try:
            success, encoded = cv2.imencode('.jpg', frame, self.encode_params)

            if not success:
                logger.error("Failed to encode frame to JPEG")
                return None

            return encoded.tobytes()

        except Exception as e:
            logger.error(f"Error encoding JPEG: {e}")
            return None

    def encode_base64(self, jpeg_data: bytes) -> str:
        """
        Encode JPEG data to base64 string.

        Args:
            jpeg_data: JPEG-encoded bytes.

        Returns:
            Base64-encoded string.
        """
        return base64.b64encode(jpeg_data).decode('utf-8')

    def process_frame_pair(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray
    ) -> Optional[str]:
        """
        Complete processing pipeline: combine -> JPEG -> base64.

        Args:
            left_frame: Left camera frame.
            right_frame: Right camera frame.

        Returns:
            Base64-encoded JPEG string, or None if processing fails.
        """
        try:
            # Combine frames side-by-side (with automatic resizing if needed)
            combined = self.combine_frames(
                left_frame,
                right_frame,
                target_height=self.target_height,
                target_width=self.target_width
            )

            # Encode to JPEG
            jpeg_data = self.encode_jpeg(combined)

            if jpeg_data is None:
                return None

            # Encode to base64
            base64_data = self.encode_base64(jpeg_data)

            return base64_data

        except Exception as e:
            logger.error(f"Error in frame processing pipeline: {e}")
            return None

    def resize_frame(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Resize frame to target dimensions.

        Args:
            frame: Input frame.
            width: Target width.
            height: Target height.
            interpolation: Interpolation method.

        Returns:
            Resized frame.
        """
        return cv2.resize(frame, (width, height), interpolation=interpolation)

    def get_frame_info(self, frame: np.ndarray) -> dict:
        """
        Get information about a frame.

        Args:
            frame: Input frame.

        Returns:
            Dictionary with frame information.
        """
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1

        return {
            'width': width,
            'height': height,
            'channels': channels,
            'dtype': str(frame.dtype),
            'size_mb': frame.nbytes / (1024 * 1024)
        }

    def validate_frame(
        self,
        frame: np.ndarray,
        expected_width: Optional[int] = None,
        expected_height: Optional[int] = None
    ) -> bool:
        """
        Validate frame dimensions and format.

        Args:
            frame: Frame to validate.
            expected_width: Expected width (optional).
            expected_height: Expected height (optional).

        Returns:
            True if valid, False otherwise.
        """
        if frame is None or frame.size == 0:
            logger.warning("Frame is None or empty")
            return False

        if len(frame.shape) < 2:
            logger.warning(f"Invalid frame shape: {frame.shape}")
            return False

        height, width = frame.shape[:2]

        if expected_width and width != expected_width:
            logger.warning(f"Width mismatch: expected {expected_width}, got {width}")
            return False

        if expected_height and height != expected_height:
            logger.warning(f"Height mismatch: expected {expected_height}, got {height}")
            return False

        return True


def create_test_pattern(
    width: int,
    height: int,
    label: str = "TEST"
) -> np.ndarray:
    """
    Create a test pattern frame for debugging.

    Args:
        width: Frame width.
        height: Frame height.
        label: Label to display on frame.

    Returns:
        Test pattern frame (BGR).
    """
    # Create gradient background
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add gradient
    for y in range(height):
        intensity = int((y / height) * 255)
        frame[y, :] = [intensity // 3, intensity // 2, intensity]

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 2, 3)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        2,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    return frame
