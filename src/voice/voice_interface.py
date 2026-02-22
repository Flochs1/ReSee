"""Voice interface with wake word detection, TTS and speech recognition."""

import subprocess
import tempfile
import wave
import io
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional

import speech_recognition as sr
import pyaudio
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import Whisper for better accuracy
WHISPER_AVAILABLE = False
whisper_model = None
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("Whisper available for speech recognition")
except ImportError:
    logger.info("Whisper not installed, using Google Speech API (install with: pip install openai-whisper)")

# Wake word variations (lowercase for matching)
# Include common misrecognitions from various speech-to-text engines
WAKE_WORDS = [
    # Correct spellings
    'resee', 're-see', 're see', 'ree see', 'ri see',
    # Common phonetic misrecognitions
    'reese', 'reece', 'reis', 'rees', 'riis',
    'recei', 'receive', 'receipt',
    'we see', 'he see', 'she see', 'i see',
    'heresy', 'her see', 'hear see',
    'razee', 'risi', 'rishi', 'risee', 'risey', 'rizzy',
    'racy', 'racey', 'rase', 'raise',
    'rc', 're c', 're z', 'rez',
    'easy', 're easy', 'uneasy',
    'recede', 'recipe', 'recent',
    'recy', 'recie', 'reecy',
    'precede', 'preceed',
    'greasy', 'breezy', 'freezy',
    # Whisper-specific
    'résée', 'rése', 'risée',
    # Single word matches (be careful - might false trigger)
    'percy', 'mercy',
]


class VoiceInterface:
    """
    Voice interface with wake word detection, TTS and speech recognition.

    Continuously listens for "Resee" wake word, then captures the query.
    Uses macOS 'say' command for TTS (fast, no dependencies).
    Uses SpeechRecognition library for speech-to-text.
    """

    def __init__(self, speech_rate: int = 200, device_index: Optional[int] = None, use_whisper: bool = True):
        """
        Initialize voice interface.

        Args:
            speech_rate: TTS speech rate (words per minute).
            device_index: Audio input device index (None = default).
            use_whisper: Use Whisper for better accuracy (if available).
        """
        self.speech_rate = speech_rate
        self.device_index = device_index
        self.use_whisper = use_whisper and WHISPER_AVAILABLE
        self.tts_executor = ThreadPoolExecutor(max_workers=1)
        self.voice_queue: Queue = Queue()  # Incoming voice queries
        self.running = False
        self.stop_event = Event()
        self._listen_thread: Optional[Thread] = None
        self._audio_monitor_thread: Optional[Thread] = None

        # Monitoring state (for live display)
        self.last_heard: str = ""  # Last transcribed text
        self.status: str = "Starting..."  # Current status
        self.is_listening: bool = False  # Currently capturing audio
        self.audio_energy: float = 0.0  # Current audio energy level (0-1 normalized)
        self.energy_threshold: float = 0.0  # Speech recognition threshold

        # Audio monitoring
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._audio_stream = None

        # Whisper model (loaded lazily)
        self._whisper_model = None

    def start(self) -> None:
        """Start the voice interface with continuous wake word detection."""
        self.running = True
        self.stop_event.clear()

        # List available microphones
        self._list_microphones()

        # Start audio level monitoring thread
        self._audio_monitor_thread = Thread(target=self._monitor_audio_level, daemon=True)
        self._audio_monitor_thread.start()

        # Start speech recognition thread
        self._listen_thread = Thread(target=self._continuous_listen, daemon=True)
        self._listen_thread.start()
        logger.info("Voice interface started (wake word: 'Resee')")

    def _list_microphones(self) -> None:
        """List available microphones for debugging."""
        try:
            p = pyaudio.PyAudio()
            logger.info("Available audio input devices:")
            found_any = False
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    logger.info(f"  [{i}] {info['name']} (inputs: {info['maxInputChannels']})")
                    found_any = True
            if not found_any:
                logger.warning("  No input devices found!")
            try:
                default = p.get_default_input_device_info()
                logger.info(f"  Default: [{default['index']}] {default['name']}")
            except IOError:
                logger.warning("  No default input device set!")
                logger.warning("  Check: System Settings > Privacy & Security > Microphone")
            p.terminate()
        except Exception as e:
            logger.warning(f"Could not list microphones: {e}")

    def _monitor_audio_level(self) -> None:
        """Monitor real-time audio levels in background."""
        CHUNK = 1024
        RATE = 16000

        try:
            self._pyaudio = pyaudio.PyAudio()

            # Get device info
            if self.device_index is not None:
                device_info = self._pyaudio.get_device_info_by_index(self.device_index)
            else:
                try:
                    device_info = self._pyaudio.get_default_input_device_info()
                except IOError:
                    logger.error("=" * 60)
                    logger.error("NO MICROPHONE ACCESS!")
                    logger.error("On macOS, grant microphone permission:")
                    logger.error("  System Settings > Privacy & Security > Microphone")
                    logger.error("  Enable access for Terminal (or your IDE)")
                    logger.error("=" * 60)
                    self.status = "No mic access - check permissions"
                    return

            logger.info(f"Using audio device: {device_info['name']}")

            self._audio_stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )

            logger.info("Audio monitoring started")
            self.status = "Listening for 'Resee'..."

            while not self.stop_event.is_set():
                try:
                    data = self._audio_stream.read(CHUNK, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16)

                    # Calculate RMS energy
                    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

                    # Normalize to 0-1 range (32768 is max for int16)
                    # Using 5000 as typical speech level for better visualization
                    self.audio_energy = min(1.0, rms / 5000.0)

                except Exception as e:
                    if not self.stop_event.is_set():
                        logger.warning(f"Audio read error: {e}")
                    break

        except OSError as e:
            if "Input overflowed" in str(e):
                pass  # Ignore overflow
            else:
                logger.error(f"Audio monitoring failed: {e}")
                self.status = f"Mic error: check permissions"
        except Exception as e:
            logger.error(f"Audio monitoring failed: {e}")
            self.status = f"Audio error: {e}"
        finally:
            if self._audio_stream:
                try:
                    self._audio_stream.stop_stream()
                    self._audio_stream.close()
                except Exception:
                    pass
            if self._pyaudio:
                try:
                    self._pyaudio.terminate()
                except Exception:
                    pass

    def speak(self, text: str) -> None:
        """
        Speak text asynchronously (non-blocking).

        Args:
            text: Text to speak aloud.
        """
        if not text or not text.strip():
            return
        self.tts_executor.submit(self._speak_sync, text)

    def _speak_sync(self, text: str) -> None:
        """
        Speak text synchronously (blocking).

        Args:
            text: Text to speak.
        """
        try:
            # Use macOS 'say' with Samantha voice (warm American female)
            subprocess.run(
                ['say', '-v', 'Samantha', '-r', str(self.speech_rate), text],
                check=False,
                capture_output=True
            )
        except FileNotFoundError:
            logger.warning("'say' command not found (not on macOS?)")
        except Exception as e:
            logger.warning(f"TTS failed: {e}")

    def _load_whisper_model(self) -> bool:
        """Load Whisper model (lazy loading)."""
        global whisper_model
        if self._whisper_model is not None:
            return True
        if not WHISPER_AVAILABLE:
            return False

        try:
            import whisper
            self.status = "Loading Whisper model..."
            logger.info("Loading Whisper 'base' model (first time may download ~150MB)...")
            # Use 'base' model - good balance of speed and accuracy
            # Options: tiny, base, small, medium, large
            self._whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.use_whisper = False
            return False

    def _transcribe_with_whisper(self, audio: sr.AudioData) -> Optional[str]:
        """
        Transcribe audio using Whisper.

        Args:
            audio: AudioData from speech_recognition.

        Returns:
            Transcribed text or None on failure.
        """
        if not self._whisper_model:
            if not self._load_whisper_model():
                return None

        try:
            # Convert AudioData to numpy array
            # speech_recognition AudioData is raw PCM
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe with Whisper
            result = self._whisper_model.transcribe(
                audio_array,
                language="en",
                fp16=False,  # Use FP32 for CPU compatibility
                task="transcribe"
            )

            text = result.get("text", "").strip()
            return text if text else None

        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
            return None

    def _transcribe_audio(self, recognizer: sr.Recognizer, audio: sr.AudioData) -> Optional[str]:
        """
        Transcribe audio using best available method.

        Args:
            recognizer: SpeechRecognition recognizer.
            audio: AudioData to transcribe.

        Returns:
            Transcribed text or None.
        """
        # Try Whisper first (better accuracy)
        if self.use_whisper:
            text = self._transcribe_with_whisper(audio)
            if text:
                return text.lower()

        # Fall back to Google Speech API
        try:
            text = recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.warning(f"Google Speech API error: {e}")
            return None

    def _continuous_listen(self) -> None:
        """Continuously listen for wake word, then capture query."""
        # Wait a moment for audio monitor to initialize first
        import time
        time.sleep(0.5)

        try:
            recognizer = sr.Recognizer()
            mic = sr.Microphone(device_index=self.device_index)
        except OSError as e:
            logger.error(f"Cannot access microphone: {e}")
            logger.error("Grant microphone permission in System Settings > Privacy & Security > Microphone")
            self.status = "No mic access"
            return
        except Exception as e:
            logger.error(f"Microphone init failed: {e}")
            self.status = f"Mic error: {e}"
            return

        # Configure pause detection - stop after 4 seconds of silence
        recognizer.pause_threshold = 4.0  # 4 seconds of silence ends phrase
        recognizer.non_speaking_duration = 0.5  # Keep 0.5s buffer

        # Adjust for ambient noise once at startup
        self.status = "Calibrating mic..."
        logger.info("Adjusting for ambient noise...")
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                self.energy_threshold = recognizer.energy_threshold
                logger.info(f"Energy threshold set to: {self.energy_threshold:.0f}")
        except Exception as e:
            logger.error(f"Mic calibration failed: {e}")
            self.status = "Mic calibration failed"
            return

        self.status = "Listening for 'Resee'..."
        logger.info("Listening for wake word 'Resee'...")

        # Load Whisper model upfront if enabled
        if self.use_whisper:
            self._load_whisper_model()

        while not self.stop_event.is_set():
            try:
                self.is_listening = True
                with mic as source:
                    # Store threshold for reference (audio_energy is updated by monitor thread)
                    self.energy_threshold = recognizer.energy_threshold

                    # Listen for speech (short timeout to check stop_event)
                    try:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                    except sr.WaitTimeoutError:
                        self.last_heard = "(silence)"
                        continue

                self.is_listening = False
                self.status = "Transcribing..."

                # Transcribe using best available method
                text = self._transcribe_audio(recognizer, audio)

                if text:
                    self.last_heard = text
                    logger.debug(f"Heard: {text}")

                    # Check for wake word
                    if self._contains_wake_word(text):
                        self.status = "Wake word detected!"
                        logger.info(f"Wake word detected in: '{text}'")

                        # Extract query after wake word, or listen for it
                        query = self._extract_query_after_wake_word(text)

                        if query:
                            self.status = f"Query: {query[:30]}..."
                            logger.info(f"Query (inline): {query}")
                            self.voice_queue.put(query)
                        else:
                            # Wake word alone - listen for the actual query
                            self.status = "Listening for query..."
                            self._speak_sync("Yes?")
                            query = self._listen_for_query(recognizer, mic)
                            if query:
                                self.last_heard = query
                                self.status = f"Query: {query[:30]}..."
                                logger.info(f"Query: {query}")
                                self.voice_queue.put(query)
                            else:
                                self.status = "No query heard"
                    else:
                        self.status = "Listening for 'Resee'..."
                else:
                    self.last_heard = "(unclear)"
                    self.status = "Listening for 'Resee'..."

            except Exception as e:
                if not self.stop_event.is_set():
                    self.status = f"Error: {e}"
                    logger.warning(f"Listen error: {e}")

            self.status = "Listening for 'Resee'..."

        self.status = "Stopped"
        logger.debug("Continuous listen stopped")

    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains the wake word."""
        text_lower = text.lower()
        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                return True
        return False

    def _extract_query_after_wake_word(self, text: str) -> Optional[str]:
        """Extract query text after the wake word, if present."""
        text_lower = text.lower()

        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                # Find position after wake word
                idx = text_lower.find(wake_word)
                after = text[idx + len(wake_word):].strip()

                # Remove common filler words at the start
                for filler in [',', '.', '!', '?']:
                    after = after.lstrip(filler).strip()

                if len(after) > 3:  # Meaningful query
                    return after

        return None

    def _listen_for_query(self, recognizer: sr.Recognizer, mic: sr.Microphone) -> Optional[str]:
        """Listen for the actual query after wake word (stops after 4s silence)."""
        try:
            with mic as source:
                # timeout=8 gives time to start speaking, pause_threshold=4 handles end
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=30)

            # Use best available transcription
            text = self._transcribe_audio(recognizer, audio)
            if text:
                return text

            self._speak_sync("Sorry, I didn't catch that")
            return None

        except sr.WaitTimeoutError:
            logger.debug("Query timeout")
            return None
        except Exception as e:
            logger.warning(f"Query listen error: {e}")
            return None

    def listen_async(self) -> None:
        """
        Trigger manual listen (legacy support for V key).

        With continuous listening, this just provides audio feedback.
        """
        self.speak("I'm always listening. Just say Resee.")

    def get_voice_query(self) -> Optional[str]:
        """
        Non-blocking check for voice input.

        Returns:
            Voice query string if available, None otherwise.
        """
        try:
            return self.voice_queue.get_nowait()
        except Empty:
            return None

    def shutdown(self) -> None:
        """Shutdown the voice interface."""
        self.running = False
        self.stop_event.set()

        if self._audio_monitor_thread and self._audio_monitor_thread.is_alive():
            self._audio_monitor_thread.join(timeout=2.0)

        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=3.0)

        self.tts_executor.shutdown(wait=False)
        logger.info("Voice interface shutdown")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
