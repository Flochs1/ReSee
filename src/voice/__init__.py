"""Voice interaction module for TTS and speech recognition."""

# Try to import voice interface (requires pyaudio, SpeechRecognition)
VOICE_AVAILABLE = False
VoiceInterface = None

try:
    from .voice_interface import VoiceInterface
    VOICE_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Voice interface not available: {e}")
    logging.getLogger(__name__).warning("Install with: pip install pyaudio SpeechRecognition openai-whisper")

__all__ = ['VoiceInterface', 'VOICE_AVAILABLE']
