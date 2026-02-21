"""Helper functions for formatting Gemini responses."""

from typing import Optional
from colorama import Fore, Style


def format_console_output(response_text: str, prefix: str = "Gemini") -> str:
    """
    Format Gemini response for console display.

    Args:
        response_text: Text response from Gemini.
        prefix: Prefix label for the output.

    Returns:
        Formatted string for console output with color coding.
    """
    if not response_text:
        return ""

    # Add color and formatting
    return f"{Fore.CYAN}[{prefix}]{Style.RESET_ALL} {response_text}"


def create_analysis_prompt(
    frame_number: Optional[int] = None,
    previous_context: Optional[str] = None
) -> str:
    """
    Create a prompt for image analysis.

    Args:
        frame_number: Optional frame number for context.
        previous_context: Optional context from previous analysis.

    Returns:
        Formatted prompt string.
    """
    base_prompt = """Analyze this stereo camera image and describe what you see.
Focus on:
- Main objects and their spatial positions
- Any people or activities
- Notable features or changes from before"""

    if frame_number:
        base_prompt = f"Frame {frame_number}: {base_prompt}"

    if previous_context:
        base_prompt += f"\n\nPrevious observation: {previous_context}"

    return base_prompt
