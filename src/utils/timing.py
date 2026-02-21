"""Frame rate control and timing utilities."""

import time
from typing import Optional
from collections import deque


class FPSController:
    """Controller for maintaining target frame rate."""

    def __init__(self, target_fps: float):
        """
        Initialize FPS controller.

        Args:
            target_fps: Target frames per second.
        """
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.last_frame_time: Optional[float] = None

    def wait(self) -> None:
        """
        Wait until it's time for the next frame.

        Uses sleep with compensation for processing time to maintain
        consistent frame rate.
        """
        current_time = time.monotonic()

        if self.last_frame_time is not None:
            elapsed = current_time - self.last_frame_time
            sleep_time = max(0, self.target_interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

        self.last_frame_time = time.monotonic()

    def reset(self) -> None:
        """Reset the controller (e.g., after a pause)."""
        self.last_frame_time = None

    def get_actual_fps(self) -> Optional[float]:
        """
        Get the actual FPS based on last frame interval.

        Returns:
            Actual FPS, or None if not enough data.
        """
        if self.last_frame_time is None:
            return None

        current_time = time.monotonic()
        elapsed = current_time - self.last_frame_time

        if elapsed > 0:
            return 1.0 / elapsed

        return None


class FrameTimer:
    """Track frame timing statistics over a window."""

    def __init__(self, window_size: int = 30):
        """
        Initialize frame timer.

        Args:
            window_size: Number of frames to track for statistics.
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time: Optional[float] = None

    def tick(self) -> None:
        """Record a frame tick."""
        current_time = time.monotonic()

        if self.last_time is not None:
            interval = current_time - self.last_time
            self.frame_times.append(interval)

        self.last_time = current_time

    def get_fps(self) -> Optional[float]:
        """
        Get average FPS over the window.

        Returns:
            Average FPS, or None if not enough data.
        """
        if len(self.frame_times) == 0:
            return None

        avg_interval = sum(self.frame_times) / len(self.frame_times)

        if avg_interval > 0:
            return 1.0 / avg_interval

        return None

    def get_min_fps(self) -> Optional[float]:
        """
        Get minimum FPS over the window.

        Returns:
            Minimum FPS, or None if not enough data.
        """
        if len(self.frame_times) == 0:
            return None

        max_interval = max(self.frame_times)

        if max_interval > 0:
            return 1.0 / max_interval

        return None

    def get_max_fps(self) -> Optional[float]:
        """
        Get maximum FPS over the window.

        Returns:
            Maximum FPS, or None if not enough data.
        """
        if len(self.frame_times) == 0:
            return None

        min_interval = min(self.frame_times)

        if min_interval > 0:
            return 1.0 / min_interval

        return None

    def reset(self) -> None:
        """Reset all statistics."""
        self.frame_times.clear()
        self.last_time = None


def timestamp() -> float:
    """
    Get current timestamp using monotonic clock.

    Returns:
        Timestamp in seconds.
    """
    return time.monotonic()


def timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds using monotonic clock.

    Returns:
        Timestamp in milliseconds.
    """
    return int(time.monotonic() * 1000)
