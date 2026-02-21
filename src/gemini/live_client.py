"""WebSocket client for Gemini Live API."""

import asyncio
import websockets
import json
import time
from typing import Optional, Callable, Dict, Any
from src.gemini.message_handler import GeminiMessage
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLiveClientError(Exception):
    """Exception raised for Gemini Live API client errors."""
    pass


class GeminiLiveClient:
    """
    WebSocket client for Gemini Live API.

    Handles bidirectional communication with Gemini for real-time
    multimodal streaming (video + audio).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-native-audio-preview-12-2025",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        timeout: int = 30
    ):
        """
        Initialize Gemini Live API client.

        Args:
            api_key: Gemini API key.
            model: Model name to use.
            max_retries: Maximum reconnection attempts.
            retry_delay: Initial retry delay in seconds (exponential backoff).
            timeout: WebSocket timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # WebSocket connection
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False

        # Message handler
        self.message_handler = GeminiMessage()

        # Response callback
        self.response_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Retry tracking
        self.retry_count = 0

    @property
    def websocket_url(self) -> str:
        """
        Get WebSocket URL with API key.

        Returns:
            WebSocket URL string.
        """
        base_url = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
        return f"{base_url}?key={self.api_key}"

    async def connect(
        self,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Connect to Gemini Live API and send setup message.

        Args:
            system_instruction: Optional system instruction for the model.
            generation_config: Optional generation configuration.

        Raises:
            GeminiLiveClientError: If connection fails.
        """
        try:
            logger.info(f"Connecting to Gemini Live API (model: {self.model})...")

            # Connect to WebSocket (wrapped in wait_for for timeout)
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.websocket_url,
                    max_size=10 * 1024 * 1024  # 10MB max message size
                ),
                timeout=self.timeout
            )

            self.connected = True
            self.retry_count = 0

            logger.info("WebSocket connected")

            # Send setup message
            setup_msg = self.message_handler.create_setup(
                model=self.model,
                generation_config=generation_config,
                system_instruction=system_instruction
            )

            await self.send_message(setup_msg)

            logger.info("Setup message sent, waiting for confirmation...")

            # Wait for setup complete response
            response = await self.receive_message()

            if response and response.get("type") == "setup_complete":
                logger.info("Gemini Live API session established")
            else:
                logger.warning(f"Unexpected setup response: {response}")

        except Exception as e:
            self.connected = False
            raise GeminiLiveClientError(f"Failed to connect: {e}")

    async def send_message(self, message: Dict[str, Any]) -> None:
        """
        Send message to Gemini.

        Args:
            message: Message dictionary to send.

        Raises:
            GeminiLiveClientError: If send fails.
        """
        if not self.connected or not self.websocket:
            raise GeminiLiveClientError("Not connected")

        try:
            message_str = self.message_handler.serialize(message)
            await self.websocket.send(message_str)

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise GeminiLiveClientError(f"Send failed: {e}")

    async def send_video_frame(self, video_data_base64: str) -> None:
        """
        Send video frame to Gemini.

        Args:
            video_data_base64: Base64-encoded JPEG frame.
        """
        message = self.message_handler.create_realtime_input_video(video_data_base64)
        await self.send_message(message)

    async def send_audio_chunk(self, audio_data_base64: str) -> None:
        """
        Send audio chunk to Gemini.

        Args:
            audio_data_base64: Base64-encoded PCM audio.
        """
        message = self.message_handler.create_realtime_input_audio(audio_data_base64)
        await self.send_message(message)

    async def send_combined(
        self,
        video_data_base64: Optional[str] = None,
        audio_data_base64: Optional[str] = None
    ) -> None:
        """
        Send video and/or audio in a single message.

        Args:
            video_data_base64: Base64-encoded JPEG frame (optional).
            audio_data_base64: Base64-encoded PCM audio (optional).
        """
        message = self.message_handler.create_realtime_input_combined(
            video_data_base64, audio_data_base64
        )
        await self.send_message(message)

    async def send_text(self, text: str) -> None:
        """
        Send text message to Gemini.

        Args:
            text: Text to send.
        """
        message = self.message_handler.create_client_content_text(text)
        await self.send_message(message)

    async def receive_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receive and parse message from Gemini.

        Args:
            timeout: Receive timeout in seconds (None = use default).

        Returns:
            Parsed message dictionary, or None if timeout/error.
        """
        if not self.connected or not self.websocket:
            return None

        try:
            # Receive raw message
            message_str = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=timeout or self.timeout
            )

            # Deserialize
            message = self.message_handler.deserialize(message_str)

            if message is None:
                return None

            # Parse response
            parsed = self.message_handler.parse_response(message)

            return parsed

        except asyncio.TimeoutError:
            return None

        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    async def receive_loop(self) -> None:
        """
        Continuous receive loop for handling incoming messages.

        Calls response_callback for each received message if set.
        """
        logger.debug("Starting receive loop")

        while self.connected:
            try:
                response = await self.receive_message(timeout=1.0)

                if response and self.response_callback:
                    self.response_callback(response)

            except Exception as e:
                if self.connected:
                    logger.error(f"Error in receive loop: {e}")
                break

        logger.debug("Receive loop stopped")

    def set_response_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Set callback function for receiving responses.

        Args:
            callback: Function that takes parsed response dict as argument.
        """
        self.response_callback = callback

    async def disconnect(self) -> None:
        """Disconnect from Gemini Live API."""
        self.connected = False

        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnected successfully, False otherwise.
        """
        if self.retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded")
            return False

        # Disconnect first
        await self.disconnect()

        # Calculate backoff delay
        delay = self.retry_delay * (2 ** self.retry_count)
        self.retry_count += 1

        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.retry_count}/{self.max_retries})...")
        await asyncio.sleep(delay)

        try:
            await self.connect()
            return True

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    def is_connected(self) -> bool:
        """
        Check if client is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self.connected and self.websocket is not None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
