"""Gemini Live API voice interface with real-time STT and TTS."""

import asyncio
import base64
import os
import struct
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional

import pyaudio
import numpy as np
from google import genai
from google.genai import types

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Wake word variations
WAKE_WORDS = [
    'resee', 're-see', 're see', 'ree see', 'ri see',
    'reese', 'reece', 'reis', 'rees', 'riis',
    'recei', 'receive', 'receipt',
    'we see', 'he see', 'she see', 'i see',
    'heresy', 'her see', 'hear see',
    'razee', 'risi', 'risee', 'risey', 'rizzy',
    'racy', 'racey', 'rase', 'raise',
    'rc', 're c', 're z', 'rez',
    'easy', 're easy', 'uneasy',
    'recede', 'recipe', 'recent',
    'recy', 'recie', 'reecy',
    'precede', 'preceed',
    'greasy', 'breezy', 'freezy',
    'résée', 'rése', 'risée',
    'percy', 'mercy',
]


class GeminiVoiceInterface:
    """Voice interface using Gemini Live API for real-time STT and TTS."""

    # Audio format constants
    INPUT_SAMPLE_RATE = 16000  # 16kHz for input
    OUTPUT_SAMPLE_RATE = 24000  # 24kHz for output
    CHANNELS = 1
    CHUNK_SIZE = 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
        device_index: Optional[int] = None
    ):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.model = model
        self.device_index = device_index

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        # State
        self.running = False
        self.stop_event = Event()
        self._main_thread: Optional[Thread] = None

        # Voice query queue
        self.voice_queue: Queue = Queue()

        # Audio playback queue
        self._audio_out_queue: Queue = Queue()

        # Monitoring state (for UI display)
        self.last_heard: str = ""
        self.status: str = "Starting..."
        self.is_listening: bool = False
        self.audio_energy: float = 0.0
        self.energy_threshold: float = 300.0

        logger.info(f"Gemini Voice Interface initialized (model: {model})")

    def start(self) -> None:
        """Start the voice interface."""
        self.running = True
        self.stop_event.clear()

        # Start main async loop in thread
        self._main_thread = Thread(target=self._run_async_loop, daemon=True)
        self._main_thread.start()

        logger.info("Gemini voice interface started (wake word: 'Resee')")

    def _run_async_loop(self) -> None:
        """Run the async event loop."""
        try:
            asyncio.run(self._main_async())
        except Exception as e:
            logger.error(f"Async loop error: {e}")
            self.status = f"Error: {e}"

    async def _main_async(self) -> None:
        """Main async function running all tasks."""
        self.status = "Connecting to Gemini..."

        # Configure Live API with female voice
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO", "TEXT"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede"  # Warm female voice
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text="""You are ReSee, a kind navigation assistant for visually impaired users.
Listen carefully and transcribe what you hear.
When users say your name (ReSee, or similar like "we see", "reese"), acknowledge them warmly.
Keep responses brief and helpful.
Speak in a warm, friendly tone like a caring friend.""")]
            )
        )

        # Correct model names for Gemini 2.5 Flash Live API
        models = [
            "gemini-2.5-flash-native-audio-preview-12-2025",
            "gemini-2.5-flash-preview-tts",
        ]

        last_error = None
        for model_name in models:
            try:
                logger.info(f"Trying model: {model_name}")
                async with self.client.aio.live.connect(
                    model=model_name,
                    config=config
                ) as session:
                    self.model = model_name
                    logger.info(f"Connected to Gemini Live: {model_name}")
                    self.status = "Listening for 'Resee'..."

                    # Run all tasks concurrently
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self._capture_audio(session))
                        tg.create_task(self._receive_responses(session))
                        tg.create_task(self._play_audio())
                        tg.create_task(self._monitor_stop())

                    return  # Success

            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                last_error = e

        logger.error(f"All models failed. Last error: {last_error}")
        self.status = "Connection failed"

    async def _monitor_stop(self) -> None:
        """Monitor stop event and cancel tasks."""
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)
        raise asyncio.CancelledError("Stop requested")

    async def _capture_audio(self, session) -> None:
        """Capture microphone audio and send to Gemini."""
        p = pyaudio.PyAudio()

        try:
            # List devices
            logger.info("Audio input devices:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    logger.info(f"  [{i}] {info['name']}")

            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.INPUT_SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK_SIZE
            )

            self.is_listening = True
            logger.info("Microphone capture started")

            while not self.stop_event.is_set():
                try:
                    # Read audio chunk
                    data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                    # Calculate energy for UI
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                    self.audio_energy = min(1.0, rms / 5000.0)

                    # Send to Gemini
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=data,
                            mime_type="audio/pcm;rate=16000"
                        )
                    )

                    await asyncio.sleep(0.01)

                except Exception as e:
                    if not self.stop_event.is_set():
                        logger.warning(f"Audio capture error: {e}")
                    break

        finally:
            self.is_listening = False
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def _receive_responses(self, session) -> None:
        """Receive and process responses from Gemini."""
        while not self.stop_event.is_set():
            try:
                async for response in session.receive():
                    if self.stop_event.is_set():
                        break

                    # Handle server content
                    server_content = response.server_content
                    if server_content:
                        # Check for turn complete
                        if server_content.turn_complete:
                            logger.debug("Turn complete")

                        # Process model turn
                        model_turn = server_content.model_turn
                        if model_turn and model_turn.parts:
                            for part in model_turn.parts:
                                # Text response
                                if part.text:
                                    text = part.text.strip().lower()
                                    self.last_heard = text
                                    logger.debug(f"Gemini: {text}")

                                    # Check for wake word
                                    if self._contains_wake_word(text):
                                        self.status = "Wake word detected!"
                                        logger.info(f"Wake word in: {text}")
                                        query = self._extract_query_after_wake_word(text)
                                        if query and len(query) > 3:
                                            self.voice_queue.put(query)
                                            self.status = f"Query: {query[:30]}..."

                                # Audio response
                                if part.inline_data:
                                    audio_data = part.inline_data.data
                                    if audio_data:
                                        self._audio_out_queue.put(audio_data)

            except Exception as e:
                if not self.stop_event.is_set():
                    logger.warning(f"Receive error: {e}")
                break

    async def _play_audio(self) -> None:
        """Play audio responses."""
        p = pyaudio.PyAudio()

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.OUTPUT_SAMPLE_RATE,
                output=True
            )

            while not self.stop_event.is_set():
                try:
                    audio_data = self._audio_out_queue.get(timeout=0.1)
                    stream.write(audio_data)
                except Empty:
                    continue
                except Exception as e:
                    logger.warning(f"Playback error: {e}")

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains wake word."""
        text_lower = text.lower()
        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                return True
        return False

    def _extract_query_after_wake_word(self, text: str) -> Optional[str]:
        """Extract query after wake word."""
        text_lower = text.lower()
        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                idx = text_lower.find(wake_word)
                after = text[idx + len(wake_word):].strip()
                for filler in [',', '.', '!', '?']:
                    after = after.lstrip(filler).strip()
                if len(after) > 3:
                    return after
        return None

    def speak(self, text: str) -> None:
        """Speak text (uses macOS say as fallback)."""
        if not text or not text.strip():
            return
        import subprocess
        try:
            subprocess.Popen(
                ['say', '-v', 'Samantha', '-r', '180', text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.warning(f"TTS failed: {e}")

    def get_voice_query(self) -> Optional[str]:
        """Get pending voice query."""
        try:
            return self.voice_queue.get_nowait()
        except Empty:
            return None

    def listen_async(self) -> None:
        """Legacy support."""
        self.speak("I'm listening. Just say Resee.")

    def shutdown(self) -> None:
        """Shutdown the interface."""
        self.running = False
        self.stop_event.set()

        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=3.0)

        logger.info("Gemini voice interface shutdown")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
