"""Camera pose representation for visual odometry."""

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraPose:
    """
    Represents the camera's pose in world coordinates.

    Uses a right-handed coordinate system:
    - X: East (positive = right)
    - Y: North (positive = forward)
    - heading: Yaw angle in radians (0 = North, positive = clockwise)
    """
    x: float = 0.0       # World X position (meters, East-West)
    y: float = 0.0       # World Y position (meters, North-South)
    heading: float = 0.0  # Yaw angle in radians (0 = North)
    timestamp: float = 0.0

    def copy(self) -> 'CameraPose':
        """Create a copy of this pose."""
        return CameraPose(
            x=self.x,
            y=self.y,
            heading=self.heading,
            timestamp=self.timestamp
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, heading]."""
        return np.array([self.x, self.y, self.heading])

    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: float = 0.0) -> 'CameraPose':
        """Create from numpy array [x, y, heading]."""
        return cls(x=arr[0], y=arr[1], heading=arr[2], timestamp=timestamp)
