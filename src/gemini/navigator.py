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
class ZoneInfo:
    """Zone clearance information."""
    clear: bool = True
    nearest_dist: float = float('inf')
    nearest_obj: Optional[str] = None


# Stationary objects that can be used to detect user motion (cannot move on their own)
STATIONARY_CLASSES = {
    # Street furniture
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    # Indoor furniture
    'chair', 'couch', 'bed', 'dining table', 'desk', 'toilet', 'sink',
    # Appliances
    'tv', 'refrigerator', 'oven', 'microwave', 'toaster',
    # Fixed structures
    'door', 'window', 'building', 'wall', 'fence', 'pole',
    # Nature (rooted)
    'tree', 'plant', 'potted plant',
    # Other immobile objects
    'clock', 'vase', 'laptop', 'keyboard', 'mouse', 'book', 'backpack',
    'suitcase', 'handbag', 'umbrella', 'bottle', 'cup', 'bowl'
}

# Mobile objects that CAN move towards us (potential collision hazards)
MOBILE_CLASSES = {
    # People
    'person',
    # Vehicles
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane', 'boat',
    # Animals
    'dog', 'cat', 'horse', 'cow', 'sheep', 'bird', 'bear', 'zebra', 'giraffe', 'elephant',
    # Sports equipment (in motion)
    'sports ball', 'frisbee', 'skateboard', 'skis', 'snowboard', 'surfboard'
}


@dataclass
class NavigationContext:
    """Context about the current navigation state."""
    is_moving: bool
    motion_magnitude: float
    motion_direction: str  # 'forward', 'backward', 'stationary'
    user_speed: float  # User's movement speed in m/s (positive = moving forward)
    left_zone: ZoneInfo
    center_zone: ZoneInfo
    right_zone: ZoneInfo


class StationaryObjectMotionDetector:
    """
    Detects user motion by tracking depth changes in stationary objects.

    If multiple stationary objects show consistent depth changes (all getting
    closer or farther), it indicates the user is moving.
    """

    def __init__(self, history_size: int = 10, depth_change_threshold: float = 0.1):
        """
        Initialize stationary object motion detector.

        Args:
            history_size: Number of depth samples to track per object.
            depth_change_threshold: Minimum depth change (m/s) to consider motion.
        """
        self.history_size = history_size
        self.depth_change_threshold = depth_change_threshold
        # track_id -> deque of (timestamp, depth)
        self.depth_history: dict = {}
        self.last_timestamp: float = 0.0

    def update(
        self,
        tracks: List,
        timestamp: float
    ) -> Tuple[bool, float, str, float]:
        """
        Update motion detector with tracked objects.

        Args:
            tracks: List of TrackedObject from detection pipeline.
            timestamp: Current timestamp.

        Returns:
            Tuple of (is_moving, motion_magnitude, direction, user_speed).
            direction is 'forward', 'backward', or 'stationary'.
            user_speed is in m/s (positive = moving forward).
        """
        dt = timestamp - self.last_timestamp if self.last_timestamp > 0 else 0.1
        self.last_timestamp = timestamp

        # Track depth changes for stationary objects
        depth_changes = []

        for track in tracks:
            # Only use stationary objects
            if track.class_name.lower() not in STATIONARY_CLASSES:
                continue

            depth = track.get_current_depth()
            if depth <= 0 or depth > 20.0:  # Invalid or too far
                continue

            track_id = track.track_id

            # Initialize history for new tracks
            if track_id not in self.depth_history:
                self.depth_history[track_id] = deque(maxlen=self.history_size)

            history = self.depth_history[track_id]

            # Calculate depth change rate if we have history
            if len(history) >= 2:
                old_time, old_depth = history[0]
                time_diff = timestamp - old_time
                if time_diff > 0.05:  # At least 50ms
                    depth_change_rate = (depth - old_depth) / time_diff
                    depth_changes.append(depth_change_rate)

            history.append((timestamp, depth))

        # Clean up old tracks
        active_ids = {t.track_id for t in tracks}
        self.depth_history = {
            tid: hist for tid, hist in self.depth_history.items()
            if tid in active_ids
        }

        if len(depth_changes) < 2:
            return False, 0.0, 'stationary', 0.0

        # Check if majority of stationary objects show consistent motion
        avg_change = np.mean(depth_changes)
        std_change = np.std(depth_changes)

        # Motion is detected if changes are consistent (low std) and significant
        is_consistent = std_change < abs(avg_change) * 0.5 + 0.1
        is_significant = abs(avg_change) > self.depth_change_threshold

        is_moving = is_consistent and is_significant

        # User speed: negative depth change = objects getting closer = user moving forward
        # So user_speed = -avg_change (positive when moving forward)
        user_speed = -avg_change if is_moving else 0.0

        if not is_moving:
            return False, 0.0, 'stationary', 0.0

        # Negative change = objects getting closer = user moving forward
        if avg_change < 0:
            direction = 'forward'
        else:
            direction = 'backward'

        return is_moving, abs(avg_change), direction, user_speed


class RelativeApproachTracker:
    """
    Tracks relative approaching behavior of mobile objects.

    Calculates relative speed = object closing speed - user speed.
    Only mobile objects (people, vehicles, animals) are considered threats.
    Objects that haven't changed in 3 frames are ignored.
    """

    def __init__(self, history_size: int = 10):
        """
        Initialize relative approach tracker.

        Args:
            history_size: Number of samples to track per object.
        """
        self.history_size = history_size
        # track_id -> deque of (closing_speed, depth, user_speed)
        self.track_history: dict = {}

    def update(self, tracks: List, user_speed: float) -> None:
        """
        Update approach tracker with current tracks and user speed.

        Args:
            tracks: List of TrackedObject from detection pipeline.
            user_speed: User's movement speed in m/s (positive = forward).
        """
        active_ids = set()

        for track in tracks:
            track_id = track.track_id
            active_ids.add(track_id)

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.history_size)

            depth = track.get_current_depth()
            self.track_history[track_id].append({
                'closing_speed': track.closing_speed,
                'depth': depth,
                'user_speed': user_speed,
                'class_name': track.class_name
            })

        # Clean up old tracks
        self.track_history = {
            tid: hist for tid, hist in self.track_history.items()
            if tid in active_ids
        }

    def get_relative_speed(self, track_id: int) -> float:
        """
        Get the relative approach speed of an object.

        Relative speed = object closing speed - user speed.
        Positive = object moving towards us faster than we're moving towards it.

        Args:
            track_id: The track ID to check.

        Returns:
            Relative approach speed in m/s, or 0.0 if unknown.
        """
        if track_id not in self.track_history:
            return 0.0

        history = self.track_history[track_id]
        if not history:
            return 0.0

        # Use most recent sample
        latest = history[-1]
        closing_speed = latest['closing_speed']
        user_speed = latest['user_speed']

        # Relative speed: how fast object approaches us beyond our own movement
        # closing_speed is positive when object gets closer
        # user_speed is positive when we move forward (towards objects)
        # If we're moving at 0.5 m/s and object closes at 0.5 m/s, relative = 0
        # If we're moving at 0.3 m/s and object closes at 0.8 m/s, relative = 0.5 (it's moving towards us)
        relative_speed = closing_speed - user_speed

        return relative_speed

    def has_changed_recently(self, track_id: int, min_change: float = 0.1) -> bool:
        """
        Check if object's depth has changed meaningfully in last 3 samples.

        Args:
            track_id: The track ID to check.
            min_change: Minimum depth change to consider "changed" (meters).

        Returns:
            True if depth has changed, False if stable.
        """
        if track_id not in self.track_history:
            return False

        history = list(self.track_history[track_id])
        if len(history) < 3:
            return True  # Not enough data, assume it could be changing

        # Check last 3 samples
        recent = history[-3:]
        depths = [s['depth'] for s in recent if s['depth'] > 0]

        if len(depths) < 2:
            return True  # Not enough valid depth data

        # Check if depth has changed
        depth_range = max(depths) - min(depths)
        return depth_range > min_change

    def is_mobile_object(self, track_id: int) -> bool:
        """
        Check if the tracked object is a mobile class (can move on its own).

        Args:
            track_id: The track ID to check.

        Returns:
            True if object is mobile (person, vehicle, animal).
        """
        if track_id not in self.track_history:
            return False

        history = self.track_history[track_id]
        if not history:
            return False

        class_name = history[-1]['class_name'].lower()
        return class_name in MOBILE_CLASSES

    def is_actively_approaching(
        self,
        track_id: int,
        threshold: float = 0.2,
        min_samples: int = 3
    ) -> bool:
        """
        Check if a mobile object is actively approaching us.

        Only returns True if:
        1. Object is a mobile class (can move on its own)
        2. Object has changed position in last 3 frames
        3. Relative speed is above threshold for recent samples

        Args:
            track_id: The track ID to check.
            threshold: Minimum relative speed to consider approaching (m/s).
            min_samples: Minimum number of samples required.

        Returns:
            True if object is actively approaching.
        """
        # Must be a mobile object
        if not self.is_mobile_object(track_id):
            return False

        # Must have changed recently (not static)
        if not self.has_changed_recently(track_id):
            return False

        if track_id not in self.track_history:
            return False

        history = list(self.track_history[track_id])
        if len(history) < min_samples:
            return False

        # Check relative speed for recent samples
        recent = history[-min_samples:]
        for sample in recent:
            closing_speed = sample['closing_speed']
            user_speed = sample['user_speed']
            relative_speed = closing_speed - user_speed

            if relative_speed < threshold:
                return False  # Not consistently approaching

        return True


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
        danger_zone_m: float = 5.0,
        closing_speed_threshold: float = 0.5
    ):
        """
        Initialize Gemini Navigator.

        Args:
            api_key: Gemini API key.
            user_goal: Description of user's current navigation goal.
            routine_interval: Seconds between routine updates.
            danger_zone_m: Distance threshold for alerts when moving (meters).
            closing_speed_threshold: Approaching speed for immediate alerts (m/s).
        """
        self.client = GeminiClient(api_key=api_key)
        self.user_goal = user_goal
        self.routine_interval = routine_interval

        # Trigger thresholds
        self.danger_zone_m = danger_zone_m
        self.closing_speed_threshold = closing_speed_threshold
        self.center_zone = (0.25, 0.75)  # Middle 50% of frame = "in path"
        self.left_zone = (0.0, 0.33)
        self.right_zone = (0.67, 1.0)

        # Motion detection using stationary objects
        self.motion_detector = StationaryObjectMotionDetector()
        self.approach_tracker = RelativeApproachTracker()
        self.nav_context = NavigationContext(
            is_moving=False,
            motion_magnitude=0.0,
            motion_direction='stationary',
            user_speed=0.0,
            left_zone=ZoneInfo(),
            center_zone=ZoneInfo(),
            right_zone=ZoneInfo()
        )

        # Timing
        self.last_call_time = 0.0

        # History tracking (last 3 frames/responses)
        self.object_history: deque = deque(maxlen=3)  # List of object descriptions per frame
        self.situation_history: deque = deque(maxlen=3)  # Previous Gemini responses

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

    def _analyze_clearance(
        self,
        tracks: List,
        frame_width: int
    ) -> Tuple[ZoneInfo, ZoneInfo, ZoneInfo]:
        """
        Analyze zone clearance with nearest object information.

        Returns:
            Tuple of (left_zone, center_zone, right_zone) ZoneInfo objects.
        """
        left_zone = ZoneInfo()
        center_zone = ZoneInfo()
        right_zone = ZoneInfo()

        for track in tracks:
            zone_name = self._get_zone(track.bbox, frame_width)
            depth = track.get_current_depth()

            if depth <= 0:
                continue

            # Select the appropriate zone
            if zone_name == "left":
                zone = left_zone
            elif zone_name == "right":
                zone = right_zone
            else:
                zone = center_zone

            # Update zone with nearest object
            if depth < zone.nearest_dist:
                zone.nearest_dist = depth
                zone.nearest_obj = track.class_name

            # Consider objects within danger zone as blocking
            if depth < self.danger_zone_m:
                zone.clear = False

        return left_zone, center_zone, right_zone

    def check_trigger(
        self,
        tracks: List,
        frame_width: int,
        current_time: float
    ) -> Tuple[bool, Optional[TriggerInfo]]:
        """
        Check if we should trigger a Gemini call.

        Trigger conditions:
        1. DANGER_CLOSE: User is moving AND object in center zone < danger_zone_m
        2. DANGER_APPROACHING: Object is persistently approaching AND in center zone
        3. ROUTINE: 1-second interval for regular updates

        Args:
            tracks: List of TrackedObject from detection pipeline.
            frame_width: Width of the frame for path calculation.
            current_time: Current timestamp.

        Returns:
            Tuple of (should_trigger, TriggerInfo or None).
        """
        ctx = self.nav_context

        # Check for DANGER_CLOSE: User moving + obstacle in path within danger zone
        if ctx.is_moving and ctx.motion_direction == 'forward':
            if not ctx.center_zone.clear:
                return True, TriggerInfo(
                    trigger_type=TriggerType.DANGER_CLOSE,
                    reason=f"{ctx.center_zone.nearest_obj} in path while moving",
                    object_name=ctx.center_zone.nearest_obj,
                    distance=ctx.center_zone.nearest_dist
                )

        # Check for DANGER_APPROACHING: Mobile object actively approaching in center
        for track in tracks:
            if not self.is_in_path(track.bbox, frame_width):
                continue

            depth = track.get_current_depth()
            if depth <= 0:
                continue

            # Only consider mobile objects (people, vehicles, animals)
            if track.class_name.lower() not in MOBILE_CLASSES:
                continue

            # Get relative approach speed (accounts for our own movement)
            relative_speed = self.approach_tracker.get_relative_speed(track.track_id)

            # Check if object is actively approaching (mobile, changed recently, consistent)
            if self.approach_tracker.is_actively_approaching(track.track_id):
                return True, TriggerInfo(
                    trigger_type=TriggerType.DANGER_APPROACHING,
                    reason=f"{track.class_name} actively approaching",
                    object_name=track.class_name,
                    distance=depth,
                    speed=relative_speed
                )

            # Also trigger for fast relative approach (even if not persistent yet)
            if relative_speed > self.closing_speed_threshold:
                return True, TriggerInfo(
                    trigger_type=TriggerType.DANGER_APPROACHING,
                    reason=f"{track.class_name} approaching fast",
                    object_name=track.class_name,
                    distance=depth,
                    speed=relative_speed
                )

        # Routine trigger
        if current_time - self.last_call_time >= self.routine_interval:
            return True, TriggerInfo(
                trigger_type=TriggerType.ROUTINE,
                reason="Scheduled update"
            )

        return False, None

    def _format_zone_status(self, zone: ZoneInfo, zone_name: str) -> str:
        """Format zone status for prompt."""
        if zone.clear:
            if zone.nearest_dist < float('inf'):
                return f"{zone_name}: clear (nearest: {zone.nearest_obj} at {zone.nearest_dist:.1f}m)"
            return f"{zone_name}: clear"
        else:
            return f"{zone_name}: {zone.nearest_obj} at {zone.nearest_dist:.1f}m"

    def _format_objects_for_history(self, tracks: List, frame_width: int) -> str:
        """Format current objects for history storage."""
        if not tracks:
            return "No objects detected"

        lines = []
        for track in tracks:
            zone = self._get_zone(track.bbox, frame_width)
            depth = track.get_current_depth()
            depth_str = f"{depth:.1f}m" if depth > 0 else "?"

            speed_str = ""
            if track.closing_speed > 0.1:
                speed_str = f" approaching"
            elif track.closing_speed < -0.1:
                speed_str = f" receding"

            lines.append(f"{track.class_name} ({zone}, {depth_str}{speed_str})")

        return ", ".join(lines)

    def build_prompt(
        self,
        trigger: TriggerInfo,
        tracks: List,
        frame_width: int
    ) -> str:
        """
        Build context-aware prompt for Gemini (text only, no image).

        Args:
            trigger: Trigger information.
            tracks: List of tracked objects.
            frame_width: Frame width for position calculation.

        Returns:
            Formatted prompt string.
        """
        ctx = self.nav_context

        # Motion status
        if ctx.is_moving:
            motion_status = f"Moving {ctx.motion_direction}"
        else:
            motion_status = "Stationary"

        # Reason for call
        if trigger.trigger_type == TriggerType.DANGER_CLOSE:
            reason = f"Warning: {trigger.object_name} at {trigger.distance:.1f}m ahead"
        elif trigger.trigger_type == TriggerType.DANGER_APPROACHING:
            reason = f"Warning: {trigger.object_name} approaching"
        else:
            reason = "Routine update"

        # Current zone status
        zone_status = f"Left: {self._format_zone_status(ctx.left_zone, 'L')}, Center: {self._format_zone_status(ctx.center_zone, 'C')}, Right: {self._format_zone_status(ctx.right_zone, 'R')}"

        # Build history context
        history_text = ""
        if self.object_history:
            history_lines = []
            for i, obj_desc in enumerate(self.object_history):
                history_lines.append(f"  Frame -{len(self.object_history) - i}: {obj_desc}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "  No previous frames"

        # Previous situations
        prev_situations = ""
        if self.situation_history:
            prev_situations = "\n".join([f"  - {s}" for s in self.situation_history])
        else:
            prev_situations = "  No previous updates"

        # Current objects
        current_objects = self._format_objects_for_history(tracks, frame_width)

        prompt = f"""You are a calm navigation assistant for a visually impaired person walking outdoors.

CURRENT STATE:
- State: {motion_status}
- Reason: {reason}
- Zones: {zone_status}

OBJECT HISTORY (last 3 frames):
{history_text}

CURRENT OBJECTS:
  {current_objects}

PREVIOUS RESPONSES:
{prev_situations}

RESPOND IN THIS EXACT FORMAT (3 lines only):
STATE: {motion_status}
REASON: [Brief reason - "Routine" or "Warning: what's happening"]
ACTION: [Only if truly needed: "Step left/right" or "Slow down" or "Stop". Otherwise leave empty or say "Continue"]

GUIDELINES:
- Be calm and reassuring, not alarming
- Only suggest action if object is <2m AND in center path
- Objects >3m away generally don't need action yet
- Prefer gentle guidance ("step slightly left") over urgent commands
- If path ahead is reasonably clear, no action needed
- Consider if situation is actually dangerous before warning"""

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

        # Update motion detection using stationary objects
        is_moving, motion_magnitude, motion_direction, user_speed = self.motion_detector.update(
            tracks, timestamp
        )

        # Update approach tracker with user speed for relative calculations
        self.approach_tracker.update(tracks, user_speed)

        # Update clearance analysis with zone info
        left_zone, center_zone, right_zone = self._analyze_clearance(tracks, frame_width)

        # Update navigation context
        self.nav_context = NavigationContext(
            is_moving=is_moving,
            motion_magnitude=motion_magnitude,
            motion_direction=motion_direction,
            user_speed=user_speed,
            left_zone=left_zone,
            center_zone=center_zone,
            right_zone=right_zone
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

        # Store current objects in history before making call
        current_objects = self._format_objects_for_history(tracks, frame_width)
        self.object_history.append(current_objects)

        # Build prompt (text only, no image)
        prompt = self.build_prompt(trigger, tracks, frame_width)

        # Submit to background thread (text-only call)
        self.pending_future = self.executor.submit(
            self._call_gemini_text, prompt
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

    def _call_gemini_text(self, prompt: str) -> Tuple[Optional[str], float]:
        """
        Call Gemini API with text only (runs in background thread).

        Args:
            prompt: Text prompt.

        Returns:
            Tuple of (response text, elapsed_ms).
        """
        return self.client.generate_text_with_retry(prompt)

    def _print_response(
        self,
        trigger: Optional[TriggerInfo],
        response: Optional[str],
        elapsed_ms: float
    ) -> None:
        """Print formatted response to console and store in history."""
        separator = "\u2501" * 60

        # Determine trigger label and icon
        if trigger is None:
            trigger_label = "UNKNOWN"
            icon = "\U0001F514"  # bell
        elif trigger.trigger_type == TriggerType.DANGER_CLOSE:
            trigger_label = f"WARNING - {trigger.object_name} at {trigger.distance:.1f}m"
            icon = "\u26A0\uFE0F"  # warning sign
        elif trigger.trigger_type == TriggerType.DANGER_APPROACHING:
            speed_str = f" {trigger.speed:.1f}m/s" if trigger.speed else ""
            trigger_label = f"WARNING - {trigger.object_name} approaching{speed_str}"
            icon = "\u26A0\uFE0F"  # warning sign
        else:
            trigger_label = "ROUTINE"
            icon = "\U0001F7E2"  # green circle

        # Format timing
        timing_str = f"[{elapsed_ms:.0f}ms]"

        if response:
            print(f"\n{separator}")
            print(f"{icon} [{trigger_label}] {timing_str}")
            print(separator)
            print(response.strip())
            print(f"{separator}\n")

            # Store abbreviated response in history (just the ACTION line if present)
            lines = response.strip().split('\n')
            summary = ""
            for line in lines:
                if line.startswith('ACTION:'):
                    action = line.replace('ACTION:', '').strip()
                    if action and action.lower() not in ['', 'continue', 'none']:
                        summary = action
                    break
            if not summary:
                summary = f"{trigger_label}"
            self.situation_history.append(summary)
        else:
            print(f"\n{icon} [{trigger_label}] {timing_str}: No response")

    def shutdown(self) -> None:
        """Shutdown background executor."""
        logger.info("Shutting down Gemini navigator...")
        self.executor.shutdown(wait=False)
