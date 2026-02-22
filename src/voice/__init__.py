"""Voice interaction module for TTS and speech recognition."""

from .voice_interface import VoiceInterface

# Try to import Gemini voice interface
try:
    from .gemini_voice import GeminiVoiceInterface
    GEMINI_VOICE_AVAILABLE = True
except ImportError:
    GeminiVoiceInterface = None
    GEMINI_VOICE_AVAILABLE = False

__all__ = ['VoiceInterface', 'GeminiVoiceInterface', 'GEMINI_VOICE_AVAILABLE']
