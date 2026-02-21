"""Audio capture from microphone for streaming."""

import pyaudio
import threading
import base64
from queue import Queue, Empty, Full
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioCaptureError(Exception):
    """Exception raised for audio capture errors."""
    pass


class AudioCapture:
    """
    Captures audio from system microphone for streaming.

    Supports 16-bit PCM audio at 16kHz mono (Gemini Live API requirement).
    """

    # Audio format constants
    FORMAT_MAP = {
        'int16': pyaudio.paInt16,
        'int32': pyaudio.paInt32,
        'float32': pyaudio.paFloat32,
    }

    def __init__(
        self,
        sample_rate: int = 16000,
        format: str = 'int16',
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
        buffer_size: int = 10
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz (16000 for Gemini).
            format: Audio format (int16, int32, float32).
            channels: Number of channels (1 = mono, 2 = stereo).
            chunk_size: Number of samples per chunk.
            device_index: Audio device index (None = system default).
            buffer_size: Audio chunk queue size.
        """
        self.sample_rate = sample_rate
        self.format = format
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.buffer_size = buffer_size

        # PyAudio objects
        self.pyaudio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

        # Threading
        self.audio_queue: Queue = Queue(maxsize=buffer_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

    def list_devices(self) -> list:
        """
        List available audio input devices.

        Returns:
            List of device information dictionaries.
        """
        if self.pyaudio is None:
            temp_pyaudio = pyaudio.PyAudio()
        else:
            temp_pyaudio = self.pyaudio

        devices = []

        try:
            for i in range(temp_pyaudio.get_device_count()):
                info = temp_pyaudio.get_device_info_by_index(i)

                # Only include input devices
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate'])
                    })

        finally:
            if self.pyaudio is None:
                temp_pyaudio.terminate()

        return devices

    def get_default_device(self) -> Optional[dict]:
        """
        Get default input device information.

        Returns:
            Device information dictionary, or None if not found.
        """
        if self.pyaudio is None:
            temp_pyaudio = pyaudio.PyAudio()
        else:
            temp_pyaudio = self.pyaudio

        try:
            default_info = temp_pyaudio.get_default_input_device_info()

            return {
                'index': default_info['index'],
                'name': default_info['name'],
                'channels': default_info['maxInputChannels'],
                'sample_rate': int(default_info['defaultSampleRate'])
            }

        except Exception as e:
            logger.error(f"Failed to get default device: {e}")
            return None

        finally:
            if self.pyaudio is None:
                temp_pyaudio.terminate()

    def open(self) -> None:
        """
        Open audio stream and start capture.

        Raises:
            AudioCaptureError: If audio stream cannot be opened.
        """
        with self.lock:
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()

            # Log device information
            if self.device_index is None:
                default_device = self.get_default_device()
                if default_device:
                    logger.info(f"Using default audio device: {default_device['name']}")
                else:
                    logger.warning("Could not get default device info")
            else:
                logger.info(f"Using audio device index: {self.device_index}")

            # Get PyAudio format
            pa_format = self.FORMAT_MAP.get(self.format)
            if pa_format is None:
                raise AudioCaptureError(f"Unsupported audio format: {self.format}")

            try:
                # Open audio stream
                self.stream = self.pyaudio.open(
                    format=pa_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=None  # We'll use blocking mode
                )

                logger.info(
                    f"Audio stream opened: {self.sample_rate}Hz, "
                    f"{self.format}, {self.channels} channel(s)"
                )

            except Exception as e:
                self.pyaudio.terminate()
                self.pyaudio = None
                raise AudioCaptureError(f"Failed to open audio stream: {e}")

            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            logger.info("Audio capture started")

    def _capture_loop(self) -> None:
        """Continuous audio capture loop (runs in background thread)."""
        logger.debug("Audio capture thread started")

        while self.running:
            try:
                # Read audio chunk (blocking)
                audio_data = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False
                )

                # Add to queue (drop old chunks if full)
                try:
                    self.audio_queue.put(audio_data, block=False)
                except Full:
                    # Queue is full, drop oldest chunk
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put(audio_data, block=False)
                    except (Empty, Full):
                        pass

            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    logger.error(f"Error in audio capture loop: {e}")
                break

        logger.debug("Audio capture thread stopped")

    def get_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Get latest audio chunk from queue.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Audio chunk bytes (PCM format), or None if timeout.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_chunk_base64(self, timeout: float = 1.0) -> Optional[str]:
        """
        Get latest audio chunk as base64 string.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Base64-encoded audio chunk, or None if timeout.
        """
        chunk = self.get_chunk(timeout)

        if chunk is None:
            return None

        return base64.b64encode(chunk).decode('utf-8')

    def close(self) -> None:
        """Close audio stream and stop capture."""
        with self.lock:
            self.running = False

            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)

            # Close stream
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Error closing audio stream: {e}")
                finally:
                    self.stream = None

            # Terminate PyAudio
            if self.pyaudio:
                try:
                    self.pyaudio.terminate()
                except Exception as e:
                    logger.warning(f"Error terminating PyAudio: {e}")
                finally:
                    self.pyaudio = None

            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except Empty:
                    break

            logger.info("Audio capture closed")

    def is_active(self) -> bool:
        """
        Check if audio capture is active.

        Returns:
            True if capturing, False otherwise.
        """
        with self.lock:
            return self.running and self.stream is not None and self.stream.is_active()

    def get_audio_level(self, chunk: bytes) -> float:
        """
        Calculate audio level (volume) from chunk.

        Args:
            chunk: Audio chunk bytes.

        Returns:
            Audio level (0.0 to 1.0).
        """
        import struct
        import numpy as np

        # Convert bytes to numpy array
        if self.format == 'int16':
            samples = np.frombuffer(chunk, dtype=np.int16)
            max_value = 32768.0
        elif self.format == 'int32':
            samples = np.frombuffer(chunk, dtype=np.int32)
            max_value = 2147483648.0
        elif self.format == 'float32':
            samples = np.frombuffer(chunk, dtype=np.float32)
            max_value = 1.0
        else:
            return 0.0

        # Calculate RMS (root mean square) as audio level
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

        # Normalize to 0-1 range
        level = min(1.0, rms / max_value)

        return level

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
