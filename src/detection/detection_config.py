"""Detection configuration."""

import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class DetectionConfig(BaseModel):
    """Object detection settings."""
    enabled: bool = True
    model_path: str = "models/yolov8n.mlpackage"  # CoreML model
    confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    use_coreml: bool = True  # Use CoreML (no PyTorch) vs ultralytics (requires PyTorch)


class TrackingConfig(BaseModel):
    """Object tracking settings."""
    iou_threshold: float = Field(default=0.3, ge=0.1, le=0.9)
    max_age_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    depth_history_frames: int = Field(default=30, ge=5, le=100)


class DetectionSettings(BaseModel):
    """Combined detection settings."""
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "DetectionSettings":
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "detection_config.yaml"

        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        return cls(**data)


_settings: Optional[DetectionSettings] = None


def get_detection_settings(reload: bool = False) -> DetectionSettings:
    global _settings
    if _settings is None or reload:
        _settings = DetectionSettings.load()
    return _settings
