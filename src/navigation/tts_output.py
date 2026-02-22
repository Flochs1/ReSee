"""Text-to-speech output using macOS say command."""

import subprocess
import time
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TTSOutput:
    """
    Text-to-speech output using macOS say command.

    Manages speech priority and rate limiting to avoid overwhelming
    the user with too many audio messages.
    """

    # Priority levels and their minimum intervals
    PRIORITY_INTERVALS = {
        "urgent": 0.0,    # Immediate, can interrupt
        "high": 1.0,      # 1 second minimum
        "normal": 2.0,    # 2 seconds minimum
        "low": 5.0,       # 5 seconds minimum (e.g., "path clear")
    }

    def __init__(
        self,
        voice: str = "Samantha",
        rate: int = 200,
        enabled: bool = True
    ):
        """
        Initialize TTS output.

        Args:
            voice: macOS voice name (e.g., "Samantha", "Alex").
            rate: Speech rate in words per minute.
            enabled: Whether TTS is enabled.
        """
        self.voice = voice
        self.rate = rate
        self.enabled = enabled

        self._last_message = ""
        self._last_time = 0.0
        self._current_process: Optional[subprocess.Popen] = None

        if enabled:
            logger.info(f"TTS initialized (voice={voice}, rate={rate})")
        else:
            logger.info("TTS disabled")

    def speak(self, message: str, priority: str = "normal") -> bool:
        """
        Speak a message with given priority.

        Args:
            message: Text to speak.
            priority: "urgent", "high", "normal", or "low".

        Returns:
            True if message was spoken, False if skipped.
        """
        if not self.enabled or not message:
            return False

        current_time = time.monotonic()
        min_interval = self.PRIORITY_INTERVALS.get(priority, 2.0)

        # Check rate limiting
        time_since_last = current_time - self._last_time

        if time_since_last < min_interval:
            # Allow urgent messages to interrupt
            if priority != "urgent":
                return False

        # Don't repeat the same message too quickly
        if message == self._last_message and time_since_last < 3.0:
            return False

        # For urgent messages, stop any current speech
        if priority == "urgent":
            self._stop_current()

        # Speak asynchronously
        self._speak_async(message)

        self._last_message = message
        self._last_time = current_time

        return True

    def _speak_async(self, message: str) -> None:
        """
        Speak message asynchronously using subprocess.

        Args:
            message: Text to speak.
        """
        try:
            # Clean the message for shell safety
            clean_message = message.replace('"', "'").replace("\\", "")

            # Use macOS say command
            cmd = ["say", "-v", self.voice, "-r", str(self.rate), clean_message]

            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            logger.debug(f"TTS: {message}")

        except FileNotFoundError:
            logger.warning("'say' command not found (not macOS?)")
            self.enabled = False
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def _stop_current(self) -> None:
        """Stop any currently speaking message."""
        if self._current_process is not None:
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=0.1)
            except Exception:
                pass
            self._current_process = None

    def is_speaking(self) -> bool:
        """
        Check if currently speaking.

        Returns:
            True if speech is in progress.
        """
        if self._current_process is None:
            return False

        # Check if process is still running
        return self._current_process.poll() is None

    def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """
        Wait for current speech to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if completed, False if timed out.
        """
        if self._current_process is None:
            return True

        try:
            self._current_process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def close(self) -> None:
        """Clean up resources."""
        self._stop_current()
