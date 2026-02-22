"""Simple REST API client for Gemini 2.5 Flash Lite."""

import base64
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from google import genai
from google.genai import types
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClientError(Exception):
    """Exception raised for Gemini client errors."""
    pass


@dataclass
class GeminiResponse:
    """Response from Gemini API with timing info."""
    text: Optional[str]
    elapsed_ms: float
    success: bool


class GeminiClient:
    """
    Simple Gemini client using REST API.

    Uses Gemini 2.5 Flash Lite for fast, cost-effective vision analysis.
    Supports multi-turn conversations with image history.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        timeout: int = 30,
        max_history: int = 3
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key.
            model: Model name to use.
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
            max_history: Maximum conversation turns to keep.
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_history = max_history

        # Conversation history for multi-turn context
        self.conversation_history: List[types.Content] = []

        # Initialize Google GenAI client
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Gemini client initialized (model: {model})")
        except Exception as e:
            raise GeminiClientError(f"Failed to initialize Gemini client: {e}")

    def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        mime_type: str = "image/jpeg",
        use_history: bool = True
    ) -> GeminiResponse:
        """
        Analyze an image with Gemini.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt for analysis.
            mime_type: MIME type of image.
            use_history: Whether to include conversation history (text only, no images).

        Returns:
            GeminiResponse with text, timing, and success status.
        """
        start_time = time.time()

        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)

            # Create blob for image
            image_blob = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )

            # Create current user content with image
            user_content = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    image_blob
                ]
            )

            # Build contents with text-only history if enabled
            if use_history and self.conversation_history:
                contents = self.conversation_history + [user_content]
            else:
                contents = [user_content]

            # Generate response
            history_turns = len(self.conversation_history) // 2
            logger.debug(f"Sending image to Gemini ({len(image_bytes)} bytes, history={history_turns} turns)")

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            elapsed_ms = (time.time() - start_time) * 1000

            # Extract text from response
            if response and response.text:
                # Add to conversation history (text only - no image)
                self._add_to_history(prompt, response.text)

                return GeminiResponse(
                    text=response.text,
                    elapsed_ms=elapsed_ms,
                    success=True
                )
            else:
                return GeminiResponse(
                    text=None,
                    elapsed_ms=elapsed_ms,
                    success=False
                )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Error analyzing image ({elapsed_ms:.0f}ms): {e}")
            return GeminiResponse(
                text=None,
                elapsed_ms=elapsed_ms,
                success=False
            )

    def _add_to_history(self, user_prompt: str, response_text: str) -> None:
        """Add a turn to conversation history (text only, no images)."""
        # Extract only the key context from prompt (skip the rules)
        # Look for CURRENT STATE and DETECTED OBJECTS sections
        context_summary = self._extract_context(user_prompt)

        # Add user message (minimal context only)
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"[Previous frame]\n{context_summary}")]
        )
        self.conversation_history.append(user_content)

        # Add assistant response
        assistant_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=response_text)]
        )
        self.conversation_history.append(assistant_content)

        # Trim history if too long (keep pairs)
        max_items = self.max_history * 2
        if len(self.conversation_history) > max_items:
            self.conversation_history = self.conversation_history[-max_items:]

    def _extract_context(self, prompt: str) -> str:
        """Extract only essential context from prompt for history."""
        lines = prompt.split('\n')
        context_lines = []
        in_section = False

        for line in lines:
            # Capture CURRENT STATE and DETECTED OBJECTS sections
            if line.startswith('CURRENT STATE:') or line.startswith('DETECTED OBJECTS:'):
                in_section = True
                context_lines.append(line)
            elif line.startswith('IMAGE:') or line.startswith('RESPOND'):
                in_section = False
            elif in_section and line.strip():
                context_lines.append(line)

        return '\n'.join(context_lines) if context_lines else "No context"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def analyze_image_with_retry(
        self,
        image_base64: str,
        prompt: str,
        mime_type: str = "image/jpeg",
        use_history: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Analyze image with automatic retry on failure.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt for analysis.
            mime_type: MIME type of image.
            use_history: Whether to include conversation history.

        Returns:
            Tuple of (response text or None, elapsed_ms).
        """
        total_elapsed = 0.0

        for attempt in range(self.max_retries):
            result = self.analyze_image(image_base64, prompt, mime_type, use_history)
            total_elapsed += result.elapsed_ms

            if result.success and result.text:
                return result.text, total_elapsed

            logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed ({result.elapsed_ms:.0f}ms)")

            if attempt >= self.max_retries - 1:
                logger.error("All retry attempts exhausted")

        return None, total_elapsed

    def is_available(self) -> bool:
        """
        Check if Gemini API is available.

        Returns:
            True if client is initialized, False otherwise.
        """
        return self.client is not None


def create_vision_prompt(context: str = "") -> str:
    """
    Create a prompt for vision analysis.

    Args:
        context: Additional context to include in prompt.

    Returns:
        Formatted prompt string.
    """
    base_prompt = """You are viewing a live stereo camera feed showing a real-world scene.
Analyze what you see and provide helpful, concise insights about:
- Objects and people in the scene
- Activities or actions happening
- Spatial relationships and depth (this is a stereo camera)
- Any notable or interesting elements

Be specific and descriptive."""

    if context:
        return f"{base_prompt}\n\nAdditional context: {context}"

    return base_prompt
