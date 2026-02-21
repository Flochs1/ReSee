"""Configuration management for ReSee application."""

import os
import yaml
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class ResolutionConfig(BaseModel):
    """Video resolution configuration."""
    width: int = Field(default=1920, ge=640, le=3840)
    height: int = Field(default=1080, ge=480, le=2160)


class CameraConfig(BaseModel):
    """Camera configuration."""
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    fps: int = Field(default=10, ge=1, le=30)
    device_mode: str = Field(default="auto")
    device_indices: List[int] = Field(default=[0, 1])
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    buffer_size: int = Field(default=3, ge=1, le=10)

    @validator('device_mode')
    def validate_device_mode(cls, v):
        """Validate device mode."""
        allowed = ['auto', 'single', 'dual', 'retrieve']
        if v not in allowed:
            raise ValueError(f"device_mode must be one of {allowed}")
        return v


class AudioConfig(BaseModel):
    """Audio configuration."""
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    format: str = Field(default="int16")
    channels: int = Field(default=1, ge=1, le=2)
    chunk_size: int = Field(default=1024, ge=256, le=4096)
    device_index: Optional[int] = Field(default=None)

    @validator('format')
    def validate_format(cls, v):
        """Validate audio format."""
        allowed = ['int16', 'int32', 'float32']
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v


class GeminiConfig(BaseModel):
    """Gemini API configuration."""
    api_key_env: str = Field(default="GEMINI_API_KEY")
    model: str = Field(default="gemini-2.5-flash-lite")
    max_retries: int = Field(default=3, ge=0, le=20)
    timeout: int = Field(default=30, ge=5, le=300)
    frame_interval: float = Field(default=2.0, ge=0.5, le=10.0)

    @property
    def api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{self.api_key_env}'. "
                f"Please set it in your .env file or environment."
            )
        return api_key


class DisplayConfig(BaseModel):
    """Display configuration."""
    preview_enabled: bool = Field(default=True)
    window_name: str = Field(default="ReSee - Stereo Camera Preview")
    show_fps: bool = Field(default=True)
    show_status: bool = Field(default=True)
    scale: float = Field(default=0.5, ge=0.1, le=2.0)


class DepthConfig(BaseModel):
    """Depth estimation configuration."""
    enabled: bool = Field(default=True)
    baseline_mm: float = Field(default=60.0, ge=10.0, le=500.0)
    calibration_file: str = Field(default="config/calibration/stereo_calib.npz")
    num_disparities: int = Field(default=64, ge=16, le=256)
    block_size: int = Field(default=9, ge=3, le=21)
    min_depth_m: float = Field(default=1.0, ge=0.1, le=5.0)
    max_depth_m: float = Field(default=5.0, ge=1.0, le=20.0)

    @validator('num_disparities')
    def validate_num_disparities(cls, v):
        """Ensure num_disparities is divisible by 16."""
        if v % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16")
        return v

    @validator('block_size')
    def validate_block_size(cls, v):
        """Ensure block_size is odd."""
        if v % 2 == 0:
            raise ValueError("block_size must be odd")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    console_colors: bool = Field(default=True)
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @validator('level')
    def validate_level(cls, v):
        """Validate logging level."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"level must be one of {allowed}")
        return v


class Settings(BaseModel):
    """Main application settings."""
    camera: CameraConfig = Field(default_factory=CameraConfig)
    audio: Optional[AudioConfig] = None  # Optional - audio support disabled
    gemini: Optional[GeminiConfig] = None  # Optional - Gemini support disabled
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """
        Load settings from YAML file and environment variables.

        Args:
            config_path: Path to config.yaml file. If None, uses default location.

        Returns:
            Settings instance with loaded configuration.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Default config path
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        # Load YAML config if it exists
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            # Use defaults if no config file found
            config_data = {}

        # Create Settings instance (Pydantic will validate)
        return cls(**config_data)

    def get_combined_resolution(self) -> tuple[int, int]:
        """
        Get the combined resolution for side-by-side stereo frames.

        Returns:
            (width, height) tuple for combined frame.
        """
        return (self.camera.resolution.width * 2, self.camera.resolution.height)


# Singleton instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[Path] = None, reload: bool = False) -> Settings:
    """
    Get application settings (singleton pattern).

    Args:
        config_path: Path to config.yaml file. Only used on first call or when reload=True.
        reload: Force reload of settings.

    Returns:
        Settings instance.
    """
    global _settings

    if _settings is None or reload:
        _settings = Settings.load(config_path)

    return _settings
