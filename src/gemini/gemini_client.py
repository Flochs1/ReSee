"""Simple REST API client for Gemini 2.5 Flash Lite."""

import base64
import cv2
import numpy as np
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

    MODEL = "gemini-2.5-flash-lite"

    def __init__(self, api_key: str):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key.
        """
        self.api_key = api_key

        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Gemini client initialized (model: {self.MODEL})")
        except Exception as e:
            raise GeminiClientError(f"Failed to initialize Gemini client: {e}")

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode OpenCV frame to base64 JPEG."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_obstacles(
        self,
        camera_frame: np.ndarray,
        depth_frame: np.ndarray
    ) -> Optional[str]:
        """
        Analyze camera and depth images for obstacles.

        Args:
            camera_frame: RGB camera image (numpy array from OpenCV).
            depth_frame: Colorized depth map (numpy array from OpenCV).

        Returns:
            Text response describing obstacles, or None if failed.
        """
        try:
            # Encode both frames
            camera_b64 = self._encode_frame(camera_frame)
            depth_b64 = self._encode_frame(depth_frame)

            camera_bytes = base64.b64decode(camera_b64)
            depth_bytes = base64.b64decode(depth_b64)

            # Create image parts
            camera_part = types.Part.from_bytes(
                data=camera_bytes,
                mime_type="image/jpeg"
            )
            depth_part = types.Part.from_bytes(
                data=depth_bytes,
                mime_type="image/jpeg"
            )

            prompt = """You are analyzing images from a stereo camera system to help with navigation.

Image 1 is the camera view.
Image 2 is the depth map where:
- RED = very close (0-1m)
- YELLOW = close (1-2m)
- GREEN = medium distance (2-3m)
- CYAN = further (3-4m)
- BLUE = far (4-5m+)

What are the major obstacles in the way? Be concise and focus on:
1. What objects are blocking the path
2. How close they are (use the depth colors)
3. Which direction they are (left, center, right)

Keep your response brief and actionable."""

            # Pass prompt string directly, images as Parts
            contents = [
                prompt,
                camera_part,
                depth_part
            ]

            logger.debug("Sending images to Gemini...")

            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=contents
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error analyzing obstacles: {e}")
            return None
