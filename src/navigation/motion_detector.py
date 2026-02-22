"""Motion detection using visual odometry data."""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MotionSample:
    """A single motion sample."""
    timestamp: float
    x: float
    y: float
    heading: float


class MotionDetector:
    """
    Detects motion state (stationary, walking, turning) from visual odometry.

    Uses position and heading history to classify the user's motion state.
    """

    def __init__(
        self,
        stationary_threshold_mps: float = 0.05,
        walking_threshold_mps: float = 0.1,
        turning_threshold_dps: float = 15.0,
        window_seconds: float = 0.5
    ):
        """
        Initialize motion detector.

        Args:
            stationary_threshold_mps: Speed below which user is stationary (m/s).
            walking_threshold_mps: Speed above which user is walking (m/s).
            turning_threshold_dps: Rotation rate for turning detection (deg/s).
            window_seconds: Time window for motion analysis.
        """
        self.stationary_threshold = stationary_threshold_mps
        self.walking_threshold = walking_threshold_mps
        self.turning_threshold = math.radians(turning_threshold_dps)
        self.window_seconds = window_seconds

        # History buffer (keep enough for the window)
        max_samples = int(window_seconds * 60)  # Assume up to 60 fps
        self._history: Deque[MotionSample] = deque(maxlen=max(30, max_samples))

        self._state = "stationary"
        self._speed = 0.0
        self._rotation_rate = 0.0

        logger.info(
            f"Motion detector initialized (stationary<{stationary_threshold_mps}m/s, "
            f"walking>{walking_threshold_mps}m/s, turning>{turning_threshold_dps}deg/s)"
        )

    def update(
        self,
        camera_pose,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Update motion state from camera pose.

        Args:
            camera_pose: CameraPose object with x, y, heading attributes.
            timestamp: Current timestamp (uses monotonic time if not provided).

        Returns:
            Motion state: "stationary", "walking", or "turning".
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # Handle None pose (VO not available)
        if camera_pose is None:
            return self._state

        # Add sample to history
        sample = MotionSample(
            timestamp=timestamp,
            x=camera_pose.x,
            y=camera_pose.y,
            heading=camera_pose.heading
        )
        self._history.append(sample)

        # Need at least 2 samples
        if len(self._history) < 2:
            return self._state

        # Analyze motion over window
        self._analyze_motion()

        return self._state

    def _analyze_motion(self) -> None:
        """Analyze motion from position history."""
        if len(self._history) < 2:
            return

        # Find samples within window
        current_time = self._history[-1].timestamp
        window_start = current_time - self.window_seconds

        # Get samples in window
        window_samples = [
            s for s in self._history
            if s.timestamp >= window_start
        ]

        if len(window_samples) < 2:
            # Not enough recent samples, use last two
            window_samples = [self._history[-2], self._history[-1]]

        # Calculate displacement and rotation
        first = window_samples[0]
        last = window_samples[-1]
        dt = last.timestamp - first.timestamp

        if dt < 0.01:  # Need some time span
            return

        # Calculate speed
        dx = last.x - first.x
        dy = last.y - first.y
        distance = math.sqrt(dx * dx + dy * dy)
        self._speed = distance / dt

        # Calculate rotation rate
        dheading = self._normalize_angle(last.heading - first.heading)
        self._rotation_rate = abs(dheading) / dt

        # Classify motion state
        if self._speed < self.stationary_threshold:
            if self._rotation_rate > self.turning_threshold:
                self._state = "turning"
            else:
                self._state = "stationary"
        elif self._speed >= self.walking_threshold:
            if self._rotation_rate > self.turning_threshold:
                self._state = "turning"
            else:
                self._state = "walking"
        else:
            # Between thresholds - keep previous state
            pass

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @property
    def is_moving(self) -> bool:
        """Whether the user is currently moving (walking or turning)."""
        return self._state in ("walking", "turning")

    @property
    def state(self) -> str:
        """Current motion state."""
        return self._state

    @property
    def speed(self) -> float:
        """Current estimated speed in m/s."""
        return self._speed

    @property
    def rotation_rate(self) -> float:
        """Current rotation rate in rad/s."""
        return self._rotation_rate

    def reset(self) -> None:
        """Reset motion detector state."""
        self._history.clear()
        self._state = "stationary"
        self._speed = 0.0
        self._rotation_rate = 0.0
