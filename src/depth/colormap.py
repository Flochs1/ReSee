"""Depth visualization utilities."""

import numpy as np
import cv2

# Available colormaps for depth visualization
COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
}


def apply_colormap(
    depth_map: np.ndarray,
    colormap: str = "inferno",
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert depth map to BGR colormap for display.

    Args:
        depth_map: Float32 depth values (H, W)
        colormap: Name of colormap to use (inferno, magma, plasma, viridis, jet, turbo, hot, bone)
        normalize: Whether to normalize depth values to 0-255 range

    Returns:
        BGR image (H, W, 3) suitable for display with OpenCV
    """
    if colormap not in COLORMAPS:
        colormap = "inferno"

    if normalize:
        # Normalize to 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 1e-6:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map)

        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    else:
        # Assume already in 0-255 range
        depth_uint8 = depth_map.astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(depth_uint8, COLORMAPS[colormap])

    return colored


def blend_with_image(
    image: np.ndarray,
    depth_colored: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend depth colormap with original image.

    Args:
        image: Original BGR image (H, W, 3)
        depth_colored: Colormapped depth (H, W, 3)
        alpha: Blend factor (0 = image only, 1 = depth only)

    Returns:
        Blended BGR image (H, W, 3)
    """
    alpha = max(0.0, min(1.0, alpha))

    # Resize depth to match image if needed
    if depth_colored.shape[:2] != image.shape[:2]:
        depth_colored = cv2.resize(
            depth_colored,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    blended = cv2.addWeighted(image, 1 - alpha, depth_colored, alpha, 0)
    return blended
