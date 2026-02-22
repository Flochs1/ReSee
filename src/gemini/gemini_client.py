"""Simple REST API client for Gemini 2.5 Flash Lite."""

import base64
from typing import Optional, Tuple
from google import genai
from google.genai import types
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClientError(Exception):
    """Exception raised for Gemini client errors."""
    pass


class GeminiClient:
    """
    Simple Gemini client using REST API.

    Uses Gemini 2.5 Flash Lite for fast, cost-effective vision analysis.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key.
            model: Model name to use.
            max_retries: Maximum retry attempts.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize Google GenAI client
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Gemini client initialized (model: {model})")
        except Exception as e:
            raise GeminiClientError(f"Failed to initialize Gemini client: {e}")

    def analyze_image(
        self,
        image_base64: str,
        prompt: str = "What do you see in this stereo camera image? Describe the scene.",
        mime_type: str = "image/jpeg"
    ) -> Optional[str]:
        """
        Analyze an image with Gemini.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt for analysis.
            mime_type: MIME type of image.

        Returns:
            Text response from Gemini, or None if failed.
        """
        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)

            # Create blob for image
            image_blob = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )

            # Create content with text and image
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        image_blob
                    ]
                )
            ]

            # Generate response
            logger.debug(f"Sending image to Gemini ({len(image_bytes)} bytes)")

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )

            # Extract text from response
            if response and response.text:
                logger.debug(f"Received response ({len(response.text)} chars)")
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None

    def analyze_image_with_retry(
        self,
        image_base64: str,
        prompt: str = "What do you see in this stereo camera image?",
        mime_type: str = "image/jpeg"
    ) -> Optional[str]:
        """
        Analyze image with automatic retry on failure.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt for analysis.
            mime_type: MIME type of image.

        Returns:
            Text response from Gemini, or None if all retries failed.
        """
        for attempt in range(self.max_retries):
            try:
                result = self.analyze_image(image_base64, prompt, mime_type)
                if result:
                    return result

                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} returned empty result")

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    logger.info("Retrying...")
                    continue
                else:
                    logger.error("All retry attempts exhausted")
                    return None

        return None

    def generate_text(self, prompt: str) -> Optional[str]:
        """
        Generate text response (no image).

        Args:
            prompt: Text prompt.

        Returns:
            Text response from Gemini, or None if failed.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            if response and response.text:
                logger.debug(f"Received response ({len(response.text)} chars)")
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None

    def generate_text_with_retry(self, prompt: str) -> Tuple[Optional[str], float]:
        """
        Generate text with automatic retry on failure.

        Args:
            prompt: Text prompt.

        Returns:
            Tuple of (response text, elapsed_ms).
        """
        import time
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                result = self.generate_text(prompt)
                elapsed_ms = (time.time() - start_time) * 1000

                if result:
                    return result, elapsed_ms

                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} returned empty result")

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    logger.info("Retrying...")
                    continue

        elapsed_ms = (time.time() - start_time) * 1000
        return None, elapsed_ms

    def analyze_image_with_timing(
        self,
        image_base64: str,
        prompt: str,
        mime_type: str = "image/jpeg"
    ) -> Tuple[Optional[str], float]:
        """
        Analyze image with timing info and retry on failure.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Text prompt for analysis.
            mime_type: MIME type of image.

        Returns:
            Tuple of (response text, elapsed_ms).
        """
        import time
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                result = self.analyze_image(image_base64, prompt, mime_type)
                elapsed_ms = (time.time() - start_time) * 1000

                if result:
                    return result, elapsed_ms

                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} returned empty result")

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    logger.info("Retrying...")
                    continue

        elapsed_ms = (time.time() - start_time) * 1000
        return None, elapsed_ms

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
