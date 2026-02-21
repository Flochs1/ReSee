"""Video display module with headless fallback."""

import cv2
import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoDisplay:
    """Manages video preview window with optional headless mode."""

    def __init__(
        self,
        window_name: str = "Video Preview",
        scale: float = 1.0,
        show_fps: bool = True,
        show_status: bool = True
    ):
        """
        Initialize video display.

        Args:
            window_name: Name of the display window.
            scale: Scale factor for display (1.0 = full size).
            show_fps: Whether to show FPS counter.
            show_status: Whether to show status text.
        """
        self.window_name = window_name
        self.scale = scale
        self.show_fps = show_fps
        self.show_status = show_status

        self.available = False
        self.window_created = False
        self.fps_text = ""
        self.status_text = "Initializing..."

        # Try to detect if display is available
        self._check_display_available()

    def _check_display_available(self) -> None:
        """Check if display/GUI is available."""
        try:
            # Try to create a test window
            test_window = "test_display_check"
            cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(test_window)
            cv2.waitKey(1)

            self.available = True
            logger.info("Display available - video preview enabled")

        except (cv2.error, Exception) as e:
            self.available = False
            logger.info(f"Display not available - running in headless mode: {e}")

    def create_window(self) -> bool:
        """
        Create display window.

        Returns:
            True if window created successfully, False otherwise.
        """
        if not self.available:
            logger.debug("Display not available, skipping window creation")
            return False

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.window_created = True
            logger.debug(f"Created window: {self.window_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to create window: {e}")
            self.available = False
            self.window_created = False
            return False

    def show_frame(
        self,
        frame: np.ndarray,
        fps: Optional[float] = None,
        status: Optional[str] = None
    ) -> bool:
        """
        Display frame in window with optional overlays.

        Args:
            frame: Frame to display (BGR format).
            fps: Current FPS (optional).
            status: Status message (optional).

        Returns:
            True if displayed successfully, False if headless or error.
        """
        if not self.available:
            return False

        if not self.window_created:
            if not self.create_window():
                return False

        try:
            # Scale frame if needed
            if self.scale != 1.0:
                height, width = frame.shape[:2]
                new_width = int(width * self.scale)
                new_height = int(height * self.scale)
                display_frame = cv2.resize(frame, (new_width, new_height))
            else:
                display_frame = frame.copy()

            # Add FPS overlay
            if self.show_fps and fps is not None:
                self.fps_text = f"FPS: {fps:.1f}"
                cv2.putText(
                    display_frame,
                    self.fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # Add status overlay
            if self.show_status and status is not None:
                self.status_text = status
                cv2.putText(
                    display_frame,
                    self.status_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # Display frame
            cv2.imshow(self.window_name, display_frame)

            # Note: waitKey is called separately via check_key_press()
            # to avoid consuming key events here

            return True

        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
            self.available = False
            return False

    def check_key_press(self, timeout: int = 1) -> int:
        """
        Check for key press and process window events.

        This MUST be called regularly for the display to update.

        Args:
            timeout: Wait timeout in milliseconds.

        Returns:
            Key code, or -1 if no key pressed.
        """
        if not self.available or not self.window_created:
            return -1

        try:
            key = cv2.waitKey(timeout) & 0xFF
            # Also check if window was closed via X button
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return 27  # Return ESC to signal close
            return key
        except Exception:
            return -1

    def close(self) -> None:
        """Close display window."""
        if self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
                self.window_created = False
                logger.debug(f"Closed window: {self.window_name}")
            except Exception as e:
                logger.warning(f"Error closing window: {e}")

    def is_available(self) -> bool:
        """
        Check if display is available.

        Returns:
            True if display available, False if headless.
        """
        return self.available

    def is_open(self) -> bool:
        """
        Check if window is currently open.

        Returns:
            True if window is open, False otherwise.
        """
        return self.window_created

    def __enter__(self):
        """Context manager entry."""
        self.create_window()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def add_overlay_text(
    frame: np.ndarray,
    text: str,
    position: tuple = (10, 30),
    font_scale: float = 0.7,
    color: tuple = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Add text overlay to frame.

    Args:
        frame: Input frame.
        text: Text to display.
        position: (x, y) position for text.
        font_scale: Font scale.
        color: Text color (BGR).
        thickness: Text thickness.

    Returns:
        Frame with text overlay.
    """
    frame_with_text = frame.copy()

    cv2.putText(
        frame_with_text,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return frame_with_text


def add_status_bar(
    frame: np.ndarray,
    status_items: dict,
    bar_height: int = 40,
    bg_color: tuple = (50, 50, 50),
    text_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Add status bar to bottom of frame.

    Args:
        frame: Input frame.
        status_items: Dictionary of status key-value pairs.
        bar_height: Height of status bar.
        bg_color: Background color (BGR).
        text_color: Text color (BGR).

    Returns:
        Frame with status bar.
    """
    height, width = frame.shape[:2]

    # Create frame with status bar
    output = np.zeros((height + bar_height, width, 3), dtype=frame.dtype)
    output[:height] = frame

    # Fill status bar background
    output[height:] = bg_color

    # Add status text
    status_text = " | ".join([f"{k}: {v}" for k, v in status_items.items()])

    cv2.putText(
        output,
        status_text,
        (10, height + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA
    )

    return output
