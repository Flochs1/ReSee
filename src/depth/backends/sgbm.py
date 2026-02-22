"""OpenCV StereoSGBM backend for stereo depth estimation.

Semi-Global Block Matching (SGBM) is a classical stereo matching algorithm.
It works everywhere with no dependencies beyond OpenCV.
"""

import cv2
import numpy as np


class SGBMBackend:
    """OpenCV StereoSGBM stereo matching backend.

    This is the fallback backend that works on all platforms with no
    additional dependencies. It runs on CPU but is reliable.
    """

    def __init__(
        self,
        num_disparities: int = 128,
        block_size: int = 5,
        min_disparity: int = 0,
    ):
        """
        Initialize StereoSGBM backend.

        Args:
            num_disparities: Maximum disparity minus minimum disparity.
                             Must be divisible by 16.
            block_size: Matched block size. Must be odd (3-11 recommended).
            min_disparity: Minimum possible disparity value.
        """
        # Ensure num_disparities is divisible by 16
        num_disparities = max(16, (num_disparities // 16) * 16)

        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(11, block_size))

        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_disparity = min_disparity

        # Compute P1 and P2 penalties (based on OpenCV recommendations)
        # P1: penalty for disparity changes of 1
        # P2: penalty for larger disparity changes
        cn = 3  # number of channels
        self.P1 = 8 * cn * block_size * block_size
        self.P2 = 32 * cn * block_size * block_size

        # Create the stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        # Optional: WLS filter for post-processing (smoother results)
        self.wls_filter = None
        self.right_matcher = None
        try:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self.stereo
            )
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.5)
        except AttributeError:
            # cv2.ximgproc not available (opencv-contrib-python not installed)
            pass

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from stereo pair.

        Args:
            left: Left camera BGR image (H, W, 3)
            right: Right camera BGR image (H, W, 3)

        Returns:
            Disparity map as float32 array (H, W)
        """
        # Convert to grayscale for matching
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity_left = self.stereo.compute(left_gray, right_gray)

        # Apply WLS filter if available
        if self.wls_filter is not None and self.right_matcher is not None:
            disparity_right = self.right_matcher.compute(right_gray, left_gray)
            disparity = self.wls_filter.filter(
                disparity_left, left_gray, None, disparity_right
            )
        else:
            disparity = disparity_left

        # Convert from fixed-point (divide by 16) to float
        disparity = disparity.astype(np.float32) / 16.0

        # Clip negative values (invalid disparities)
        disparity = np.clip(disparity, 0, None)

        return disparity

    def shutdown(self) -> None:
        """Release resources."""
        self.stereo = None
        self.wls_filter = None
        self.right_matcher = None

    @staticmethod
    def get_backend_name() -> str:
        """Return backend identifier."""
        return "sgbm"

    @staticmethod
    def get_backend_info() -> str:
        """Return human-readable backend description."""
        return "OpenCV StereoSGBM (CPU)"
