"""Navigation pipeline for visually impaired assistance."""

import time
from pathlib import Path
from typing import List, Optional

import yaml

from .obstacle_advisor import ObstacleAdvisor, Obstacle
from .tts_output import TTSOutput
from .motion_detector import MotionDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NavigationPipeline:
    """
    Orchestrates navigation assistance for visually impaired users.

    Combines obstacle detection, motion state, and TTS output to provide
    real-time navigation advice.
    """

    def __init__(
        self,
        enabled: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Initialize navigation pipeline.

        Args:
            enabled: Whether navigation assistance is enabled.
            config_path: Path to navigation config file.
        """
        self.enabled = enabled

        if not enabled:
            logger.info("Navigation pipeline disabled")
            self.obstacle_advisor = None
            self.tts = None
            self.motion_detector = None
            return

        # Load configuration
        config = self._load_config(config_path)

        # Initialize components
        nav_config = config.get("navigation", {})
        zones = nav_config.get("danger_zones", {})
        tts_config = nav_config.get("tts", {})
        motion_config = nav_config.get("motion", {})

        self.obstacle_advisor = ObstacleAdvisor(
            fov_degrees=nav_config.get("fov_degrees", 75.0),
            center_corridor_degrees=nav_config.get("center_corridor_degrees", 30.0),
            danger_m=zones.get("danger_m", 1.5),
            warning_m=zones.get("warning_m", 3.0),
            caution_m=zones.get("caution_m", 5.0)
        )

        self.tts = TTSOutput(
            voice=tts_config.get("voice", "Samantha"),
            rate=tts_config.get("rate", 200),
            enabled=True
        )

        self.motion_detector = MotionDetector(
            stationary_threshold_mps=motion_config.get("stationary_threshold_mps", 0.05),
            walking_threshold_mps=motion_config.get("walking_threshold_mps", 0.1)
        )

        # Timing for continuous feedback
        self._last_advice_time = 0.0
        self._continuous_interval = tts_config.get("continuous_interval_seconds", 3.0)
        self._last_advice_text = ""

        logger.info("Navigation pipeline initialized")

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load navigation configuration."""
        if config_path is None:
            config_path = "config/navigation_config.yaml"

        path = Path(config_path)
        if path.exists():
            try:
                with open(path) as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded navigation config from {config_path}")
                return config or {}
            except Exception as e:
                logger.warning(f"Failed to load navigation config: {e}")

        logger.info("Using default navigation configuration")
        return {}

    def update(
        self,
        tracks: List,
        frame_width: int,
        camera_pose=None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Process frame and provide navigation advice.

        Args:
            tracks: List of TrackedObject instances from object tracker.
            frame_width: Width of the camera frame in pixels.
            camera_pose: CameraPose object from visual odometry (optional).
            timestamp: Current timestamp (uses monotonic time if not provided).
        """
        if not self.enabled:
            return

        if timestamp is None:
            timestamp = time.monotonic()

        # Update motion state
        motion_state = self.motion_detector.update(camera_pose, timestamp)

        # Analyze obstacles
        obstacles = self.obstacle_advisor.analyze_path(tracks, frame_width)

        # Generate advice
        advice, priority = self.obstacle_advisor.get_advice(
            obstacles,
            is_moving=self.motion_detector.is_moving
        )

        # Speak advice
        self._speak_advice(advice, priority, obstacles, timestamp)

    def _speak_advice(
        self,
        advice: str,
        priority: str,
        obstacles: List[Obstacle],
        timestamp: float
    ) -> None:
        """
        Speak navigation advice with appropriate timing.

        Args:
            advice: Advice text to speak.
            priority: Priority level ("urgent", "high", "normal", "low").
            obstacles: Current obstacles for context.
            timestamp: Current timestamp.
        """
        if not advice:
            return

        # For low priority (path clear), enforce continuous interval
        if priority == "low":
            time_since_last = timestamp - self._last_advice_time
            if time_since_last < self._continuous_interval:
                return

        # Speak the advice
        spoken = self.tts.speak(advice, priority)

        if spoken:
            self._last_advice_time = timestamp
            self._last_advice_text = advice

            # Log for debugging
            danger_count = sum(1 for o in obstacles if o.danger_level == "danger")
            warning_count = sum(1 for o in obstacles if o.danger_level == "warning")
            motion = self.motion_detector.state
            logger.info(
                f"Navigation advice: '{advice}' [{priority}] "
                f"(motion={motion}, dangers={danger_count}, warnings={warning_count})"
            )

    def get_status(self) -> dict:
        """
        Get current navigation status for display.

        Returns:
            Dictionary with motion state and last advice.
        """
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "motion_state": self.motion_detector.state,
            "speed_mps": self.motion_detector.speed,
            "last_advice": self._last_advice_text
        }

    def close(self) -> None:
        """Clean up resources."""
        if self.tts:
            self.tts.close()
