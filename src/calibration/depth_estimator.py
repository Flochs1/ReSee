"""Depth estimation from stereo images."""

import cv2
import numpy as np
from typing import Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """Computes depth maps from rectified stereo image pairs."""

    def __init__(
        self,
        calibration_data: dict,
        baseline_mm: float = 60.0,
        num_disparities: int = 64,
        block_size: int = 9,
        min_depth_m: float = 1.0,
        max_depth_m: float = 5.0
    ):
        """
        Initialize depth estimator.

        Args:
            calibration_data: Dictionary with calibration matrices.
            baseline_mm: Distance between camera centers in mm.
            num_disparities: Number of disparities for stereo matching.
            block_size: Block size for stereo matching (must be odd).
            min_depth_m: Minimum depth for colorization (meters). Anything
                         closer is shown as red.
            max_depth_m: Maximum depth for colorization (meters). Shown as blue.
        """
        self.rectify_map_left = calibration_data['rectify_map_left']
        self.rectify_map_right = calibration_data['rectify_map_right']
        self.Q = calibration_data['Q']

        self.baseline_mm = baseline_mm
        self.baseline_m = baseline_mm / 1000.0
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_depth = min_depth_m
        self.max_depth = max_depth_m

        # Get focal length from Q matrix (in pixels)
        # Q[2,3] = -focal_length, Q[3,2] = 1/baseline
        if self.Q is not None and self.Q[3, 2] != 0:
            self.focal_length = abs(self.Q[2, 3])
        else:
            # Fallback: estimate from camera matrix if available
            if 'camera_matrix_left' in calibration_data:
                self.focal_length = calibration_data['camera_matrix_left'][0, 0]
            else:
                self.focal_length = 1000  # Default fallback

        # Create stereo matcher (SGBM gives better results than BM)
        self.stereo_left = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Create right matcher for WLS filter
        self.stereo_right = cv2.ximgproc.createRightMatcher(self.stereo_left)

        # Create WLS filter for disparity refinement
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_left)
        self.wls_filter.setLambda(8000)  # Smoothing strength
        self.wls_filter.setSigmaColor(1.5)  # Edge sensitivity

        # Pre-compute colormap LUT for RYGB gradient
        self.colormap_lut = self._create_rygb_colormap()

        logger.info(
            f"Depth estimator initialized: baseline={baseline_mm}mm, "
            f"disparities={num_disparities}, block_size={block_size}"
        )

    def _create_rygb_colormap(self) -> np.ndarray:
        """
        Create Red-Yellow-Green-Blue colormap LUT.

        Returns:
            256x1x3 BGR colormap lookup table.
        """
        lut = np.zeros((256, 1, 3), dtype=np.uint8)

        for i in range(256):
            t = i / 255.0  # Normalized position

            # RYGB gradient: Red (near) -> Yellow -> Green -> Blue (far)
            if t < 0.33:
                # Red to Yellow
                ratio = t / 0.33
                r = 255
                g = int(255 * ratio)
                b = 0
            elif t < 0.66:
                # Yellow to Green
                ratio = (t - 0.33) / 0.33
                r = int(255 * (1 - ratio))
                g = 255
                b = 0
            else:
                # Green to Blue
                ratio = (t - 0.66) / 0.34
                r = 0
                g = int(255 * (1 - ratio))
                b = int(255 * ratio)

            # BGR format
            lut[i, 0] = [b, g, r]

        return lut

    def rectify(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rectification to stereo image pair.

        Args:
            left_frame: Left camera frame.
            right_frame: Right camera frame.

        Returns:
            Tuple of (rectified_left, rectified_right).
        """
        left_rect = cv2.remap(
            left_frame,
            self.rectify_map_left[0],
            self.rectify_map_left[1],
            cv2.INTER_LINEAR
        )
        right_rect = cv2.remap(
            right_frame,
            self.rectify_map_right[0],
            self.rectify_map_right[1],
            cv2.INTER_LINEAR
        )
        return left_rect, right_rect

    def compute_disparity(
        self,
        left_rect: np.ndarray,
        right_rect: np.ndarray
    ) -> np.ndarray:
        """
        Compute disparity map from rectified stereo pair with WLS filtering.

        Args:
            left_rect: Rectified left frame.
            right_rect: Rectified right frame.

        Returns:
            Disparity map (float32).
        """
        # Convert to grayscale for stereo matching
        if len(left_rect.shape) == 3:
            gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = left_rect

        if len(right_rect.shape) == 3:
            gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = right_rect

        # Compute left and right disparities for WLS filter
        disp_left = self.stereo_left.compute(gray_left, gray_right)
        disp_right = self.stereo_right.compute(gray_right, gray_left)

        # Apply WLS filter for edge-preserving smoothing
        disparity = self.wls_filter.filter(disp_left, gray_left, None, disp_right)

        # Convert to float (SGBM returns disparity scaled by 16)
        disparity = disparity.astype(np.float32) / 16.0

        # Apply median filter to remove remaining speckle noise
        disparity = cv2.medianBlur(disparity.astype(np.float32), 5)

        return disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map.

        Depth = (focal_length * baseline) / disparity

        Args:
            disparity: Disparity map in pixels.

        Returns:
            Depth map in meters.
        """
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (self.focal_length * self.baseline_m) / disparity
            depth = np.where(disparity > 0, depth, 0)

        return depth

    def colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply RYGB colormap to depth map (linear).

        Red = near (min_depth or closer), Blue = far (max_depth).
        Linear interpolation between min and max depth.

        Args:
            depth: Depth map in meters.

        Returns:
            Colorized depth image (BGR).
        """
        # Linear normalization: anything <= min_depth is 0 (red), max_depth is 1 (blue)
        normalized = np.clip(
            (depth - self.min_depth) / (self.max_depth - self.min_depth),
            0, 1
        )
        normalized = (normalized * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.LUT(
            cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR),
            self.colormap_lut
        )

        # Mask invalid depth (where disparity was <= 0)
        invalid_mask = depth <= 0
        colored[invalid_mask] = [0, 0, 0]  # Black for invalid

        return colored

    def create_legend(
        self,
        height: int,
        width: int = 60
    ) -> np.ndarray:
        """
        Create a vertical color legend showing depth scale.

        Args:
            height: Height of the legend (matches depth map height).
            width: Width of the legend bar.

        Returns:
            Legend image (BGR).
        """
        # Create gradient
        gradient = np.linspace(0, 255, height).astype(np.uint8)
        gradient = gradient.reshape(height, 1)
        gradient = np.tile(gradient, (1, width - 30))  # Leave space for labels

        # Apply colormap
        gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        legend_bar = cv2.LUT(gradient_bgr, self.colormap_lut)

        # Create full legend with labels
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        legend[:, :width - 30] = legend_bar

        # Add depth labels (linear scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        color = (255, 255, 255)
        thickness = 1

        depth_range = self.max_depth - self.min_depth
        labels = [
            (0, f"{self.min_depth:.1f}m"),
            (height // 4, f"{self.min_depth + depth_range * 0.25:.1f}m"),
            (height // 2, f"{self.min_depth + depth_range * 0.5:.1f}m"),
            (3 * height // 4, f"{self.min_depth + depth_range * 0.75:.1f}m"),
            (height - 15, f"{self.max_depth:.1f}m")
        ]

        for y, text in labels:
            cv2.putText(
                legend,
                text,
                (width - 28, y + 10),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        return legend

    def process_frame(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        include_legend: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full depth processing pipeline.

        Args:
            left_frame: Left camera frame.
            right_frame: Right camera frame.
            include_legend: Whether to include depth legend.

        Returns:
            Tuple of (colorized_depth, depth_map_meters).
        """
        # Rectify
        left_rect, right_rect = self.rectify(left_frame, right_frame)

        # Compute disparity
        disparity = self.compute_disparity(left_rect, right_rect)

        # Convert to depth
        depth = self.disparity_to_depth(disparity)

        # Colorize
        colored = self.colorize_depth(depth)

        # Add legend if requested
        if include_legend:
            legend = self.create_legend(colored.shape[0])
            colored = np.hstack((colored, legend))

        return colored, depth
