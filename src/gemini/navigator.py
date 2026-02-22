"""Gemini-powered navigation assistant with smart triggering."""

import base64
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .gemini_client import GeminiClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TriggerType(Enum):
    """Type of trigger that caused a Gemini call."""
    ROUTINE = "ROUTINE"
    DANGER_CLOSE = "DANGER_CLOSE"
    DANGER_APPROACHING = "DANGER_APPROACHING"


@dataclass
class TriggerInfo:
    """Information about what triggered a Gemini call."""
    trigger_type: TriggerType
    reason: str
    object_name: Optional[str] = None
    distance: Optional[float] = None
    speed: Optional[float] = None


@dataclass
class NavigationContext:
    """Context about the current navigation state."""
    is_moving: bool
    motion_score: float
    left_clear: bool
    right_clear: bool
    center_clear: bool


class MotionDetector:
    """Detects camera/user motion using frame differencing."""

    def __init__(self, history_size: int = 10, motion_threshold: float = 5.0):
        """
        Initialize motion detector.

        Args:
            history_size: Number of frames to track.
            motion_threshold: Pixel movement threshold to consider "moving".
        """
        self.history_size = history_size
        self.motion_threshold = motion_threshold
        self.prev_gray: Optional[np.ndarray] = None
        self.motion_history: deque = deque(maxlen=history_size)

    def update(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Update motion detector with new frame.

        Args:
            frame: BGR frame from camera.

        Returns:
            Tuple of (is_moving, average_motion).
        """
        # Convert to grayscale and downsample for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120))
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        # Calculate frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        motion_score = np.mean(diff)

        self.motion_history.append(motion_score)
        self.prev_gray = gray

        # Average motion over history
        avg_motion = np.mean(self.motion_history) if self.motion_history else 0.0
        is_moving = avg_motion > self.motion_threshold

        return is_moving, avg_motion


class GeminiNavigator:
    """
    Gemini-powered navigation assistant with smart triggering.

    Provides real-time navigation assistance for visually impaired users:
    - Immediate alerts when objects approach fast or are too close
    - Routine situational updates every ~1 second
    - Only warns about obstacles in the path (center of view)

    Uses ThreadPoolExecutor for non-blocking API calls.
    """

    def __init__(
        self,
        api_key: str,
        user_goal: str = "moving forward",
        routine_interval: float = 1.0,
        danger_zone_m: float = 1.0,
        closing_speed_threshold: float = 0.5
    ):
        """
        Initialize Gemini Navigator.

        Args:
            api_key: Gemini API key.
            user_goal: Description of user's current navigation goal.
            routine_interval: Seconds between routine updates.
            danger_zone_m: Distance threshold for immediate alerts (meters).
            closing_speed_threshold: Approaching speed for immediate alerts (m/s).
        """
        self.client = GeminiClient(api_key=api_key, max_history=2)
        self.user_goal = user_goal
        self.routine_interval = routine_interval

        # Trigger thresholds
        self.danger_zone_m = danger_zone_m
        self.closing_speed_threshold = closing_speed_threshold
        self.center_zone = (0.25, 0.75)  # Middle 50% of frame = "in path"
        self.left_zone = (0.0, 0.33)
        self.right_zone = (0.67, 1.0)

        # Motion detection
        self.motion_detector = MotionDetector()
        self.nav_context = NavigationContext(
            is_moving=False,
            motion_score=0.0,
            left_clear=True,
            right_clear=True,
            center_clear=True
        )

        # Timing
        self.last_call_time = 0.0

        # Background execution (non-blocking)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future: Optional[Future] = None
        self.pending_trigger: Optional[TriggerInfo] = None

        logger.info(
            f"Gemini Navigator initialized "
            f"(interval={routine_interval}s, danger={danger_zone_m}m, "
            f"speed_threshold={closing_speed_threshold}m/s)"
        )

    def _get_zone(self, bbox: Tuple[int, int, int, int], frame_width: int) -> str:
        """Get which zone an object is in: left, center, or right."""
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        relative_x = center_x / frame_width

        if relative_x < 0.33:
            return "left"
        elif relative_x > 0.67:
            return "right"
        else:
            return "center"

    def is_in_path(self, bbox: Tuple[int, int, int, int], frame_width: int) -> bool:
        """Check if object is in the navigation path (center of frame)."""
        return self._get_zone(bbox, frame_width) == "center"

    def _analyze_clearance(self, tracks: List, frame_width: int) -> Tuple[bool, bool, bool]:
        """
        Analyze which zones are clear for dodging.

        Returns:
            Tuple of (left_clear, right_clear, center_clear).
        """
        left_clear = True
        right_clear = True
        center_clear = True

        for track in tracks:
            zone = self._get_zone(track.bbox, frame_width)
            depth = track.get_current_depth()

            # Only consider objects within 3m as blocking
            if depth > 0 and depth < 3.0:
                if zone == "left":
                    left_clear = False
                elif zone == "right":
                    right_clear = False
                else:
                    center_clear = False

        return left_clear, right_clear, center_clear

    def check_trigger(
        self,
        tracks: List,
        frame_width: int,
        current_time: float
    ) -> Tuple[bool, Optional[TriggerInfo]]:
        """
        Check if we should trigger a Gemini call.

        Args:
            tracks: List of TrackedObject from detection pipeline.
            frame_width: Width of the frame for path calculation.
            current_time: Current timestamp.

        Returns:
            Tuple of (should_trigger, TriggerInfo or None).
        """
        # Check for immediate danger triggers (objects in path)
        for track in tracks:
            if not self.is_in_path(track.bbox, frame_width):
                continue

            depth = track.get_current_depth()

            # Object too close
            if depth > 0 and depth < self.danger_zone_m:
                return True, TriggerInfo(
                    trigger_type=TriggerType.DANGER_CLOSE,
                    reason=f"{track.class_name} too close",
                    object_name=track.class_name,
                    distance=depth
                )

            # Object approaching fast
            if track.closing_speed > self.closing_speed_threshold:
                return True, TriggerInfo(
                    trigger_type=TriggerType.DANGER_APPROACHING,
                    reason=f"{track.class_name} approaching fast",
                    object_name=track.class_name,
                    distance=depth,
                    speed=track.closing_speed
                )

        # Routine trigger
        if current_time - self.last_call_time >= self.routine_interval:
            return True, TriggerInfo(
                trigger_type=TriggerType.ROUTINE,
                reason="Scheduled update"
            )

        return False, None

    def build_prompt(
        self,
        trigger: TriggerInfo,
        tracks: List,
        frame_width: int
    ) -> str:
        """
        Build context-aware prompt for Gemini.

        Args:
            trigger: Trigger information.
            tracks: List of tracked objects.
            frame_width: Frame width for position calculation.

        Returns:
            Formatted prompt string.
        """
        ctx = self.nav_context

        # Build object descriptions
        object_lines = []
        for track in tracks:
            zone = self._get_zone(track.bbox, frame_width)
            in_path = "BLOCKING PATH" if zone == "center" else f"on {zone}"
            depth = track.get_current_depth()
            depth_str = f"{depth:.1f}m" if depth > 0 else "unknown"

            speed_str = ""
            if track.closing_speed > 0.1:
                speed_str = f", approaching {track.closing_speed:.1f}m/s"
            elif track.closing_speed < -0.1:
                speed_str = f", moving away"

            object_lines.append(
                f"- {track.class_name}: {depth_str}, {in_path}{speed_str}"
            )

        objects_text = "\n".join(object_lines) if object_lines else "- None detected"

        # Motion and clearance status
        motion_status = "MOVING" if ctx.is_moving else "STATIONARY"

        clearance_parts = []
        if ctx.left_clear:
            clearance_parts.append("LEFT CLEAR")
        else:
            clearance_parts.append("LEFT BLOCKED")
        if ctx.right_clear:
            clearance_parts.append("RIGHT CLEAR")
        else:
            clearance_parts.append("RIGHT BLOCKED")
        if ctx.center_clear:
            clearance_parts.append("PATH CLEAR")
        else:
            clearance_parts.append("PATH BLOCKED")

        clearance_status = " | ".join(clearance_parts)

        # Trigger description
        if trigger.trigger_type == TriggerType.DANGER_CLOSE:
            trigger_desc = f"DANGER: {trigger.object_name} at {trigger.distance:.1f}m - IMMEDIATE THREAT"
        elif trigger.trigger_type == TriggerType.DANGER_APPROACHING:
            trigger_desc = f"DANGER: {trigger.object_name} approaching at {trigger.speed:.1f}m/s from {trigger.distance:.1f}m"
        else:
            trigger_desc = "ROUTINE: Scheduled situational update"

        prompt = f"""You are a real-time navigation assistant for a visually impaired person. You have context from previous frames.

CURRENT STATE:
- User: {motion_status}
- Clearance: {clearance_status}
- Trigger: {trigger_desc}

DETECTED OBJECTS:
{objects_text}

IMAGE: Camera view. Depth info provided in DETECTED OBJECTS above.

RESPOND IN THIS EXACT FORMAT:
SITUATION: [Describe surroundings. Note if user is moving or still. What's ahead, left, right? Any changes from before? Max 70 words]
TRIGGER: [Why this alert. Empty if routine with clear path]
ACTION: [Specific instruction. Empty if none needed]

ACTION DECISION RULES:
1. User is {motion_status}
2. If user is STATIONARY and something approaches them → must DODGE (they cannot "stop")
3. If user is MOVING and approaches something stationary → can STOP or DODGE
4. If something approaches user while MOVING → prefer DODGE over STOP (faster reaction)
5. DODGE direction: {"DODGE LEFT is safe" if ctx.left_clear else "LEFT BLOCKED"}, {"DODGE RIGHT is safe" if ctx.right_clear else "RIGHT BLOCKED"}
6. If both sides blocked and path blocked → STOP and WAIT
7. If path is clear → CONTINUE (no action needed)

Be decisive. One clear action only."""

        return prompt

    def process_frame(
        self,
        left_frame: np.ndarray,
        depth_colored: Optional[np.ndarray],
        tracks: List,
        timestamp: float
    ) -> None:
        """
        Process frame for navigation assistance (non-blocking).

        This method never blocks the main loop:
        1. Checks for completed responses from previous calls
        2. Starts new calls only if no pending call and trigger conditions met

        Args:
            left_frame: Left camera frame (BGR).
            depth_colored: Colored depth visualization, or None.
            tracks: List of tracked objects.
            timestamp: Current timestamp.
        """
        frame_width = left_frame.shape[1]

        # Update motion detection
        is_moving, motion_score = self.motion_detector.update(left_frame)

        # Update clearance analysis
        left_clear, right_clear, center_clear = self._analyze_clearance(tracks, frame_width)

        # Update navigation context
        self.nav_context = NavigationContext(
            is_moving=is_moving,
            motion_score=motion_score,
            left_clear=left_clear,
            right_clear=right_clear,
            center_clear=center_clear
        )

        # Check for completed response
        if self.pending_future is not None and self.pending_future.done():
            try:
                response_text, elapsed_ms = self.pending_future.result()
                self._print_response(self.pending_trigger, response_text, elapsed_ms)
            except Exception as e:
                logger.error(f"Gemini call failed: {e}")
            finally:
                self.pending_future = None
                self.pending_trigger = None

        # Check if we should start a new call
        if self.pending_future is not None:
            # Already have a call in flight, skip
            return

        should_trigger, trigger = self.check_trigger(
            tracks, frame_width, timestamp
        )

        if not should_trigger or trigger is None:
            return

        # Prepare image
        image_b64 = self._prepare_image(left_frame, depth_colored)
        if image_b64 is None:
            return

        # Build prompt
        prompt = self.build_prompt(trigger, tracks, frame_width)

        # Submit to background thread
        self.pending_future = self.executor.submit(
            self._call_gemini, image_b64, prompt
        )
        self.pending_trigger = trigger
        self.last_call_time = timestamp

        trigger_label = trigger.trigger_type.value
        logger.debug(f"Gemini call submitted ({trigger_label}: {trigger.reason})")

    def _prepare_image(
        self,
        left_frame: np.ndarray,
        depth_colored: Optional[np.ndarray]
    ) -> Optional[str]:
        """
        Prepare image for Gemini (camera only, depth info in text prompt).

        Args:
            left_frame: Left camera frame (BGR).
            depth_colored: Colored depth visualization (unused, depth in prompt).

        Returns:
            Base64-encoded JPEG, or None on error.
        """
        try:
            # Send only camera image (depth info already in text prompt)
            image = left_frame

            # Resize for API (max 384px height to reduce latency)
            h, w = image.shape[:2]
            if h > 384:
                scale = 384 / h
                image = cv2.resize(image, (int(w * scale), 384))

            # Encode as JPEG (quality 60 for faster transmission)
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            return base64.b64encode(buffer).decode('utf-8')

        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            return None

    def _call_gemini(self, image_b64: str, prompt: str) -> Tuple[Optional[str], float]:
        """
        Call Gemini API (runs in background thread).

        Args:
            image_b64: Base64-encoded image.
            prompt: Analysis prompt.

        Returns:
            Tuple of (response text, elapsed_ms).
        """
        return self.client.analyze_image_with_retry(image_b64, prompt, use_history=True)

    def _print_response(
        self,
        trigger: Optional[TriggerInfo],
        response: Optional[str],
        elapsed_ms: float
    ) -> None:
        """Print formatted response to console."""
        separator = "\u2501" * 60

        # Determine trigger label and icon
        if trigger is None:
            trigger_label = "UNKNOWN"
            icon = "\U0001F514"  # bell
        elif trigger.trigger_type == TriggerType.DANGER_CLOSE:
            trigger_label = f"DANGER - {trigger.object_name} at {trigger.distance:.1f}m"
            icon = "\U0001F6A8"  # rotating light
        elif trigger.trigger_type == TriggerType.DANGER_APPROACHING:
            trigger_label = f"DANGER - {trigger.object_name} approaching {trigger.speed:.1f}m/s"
            icon = "\U0001F6A8"  # rotating light
        else:
            trigger_label = "ROUTINE"
            icon = "\U0001F514"  # bell

        # Format timing
        timing_str = f"[{elapsed_ms:.0f}ms]"

        if response:
            print(f"\n{separator}")
            print(f"{icon} [{trigger_label}] {timing_str}")
            print(separator)
            print(response.strip())
            print(f"{separator}\n")
        else:
            print(f"\n{icon} [{trigger_label}] {timing_str}: No response")

    def shutdown(self) -> None:
        """Shutdown background executor."""
        logger.info("Shutting down Gemini navigator...")
        self.executor.shutdown(wait=False)
