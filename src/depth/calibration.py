"""Stereo calibration utilities for depth estimation (optional)."""

import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass
class StereoCalibration:
    """Stereo camera calibration data."""

    # Camera matrices
    camera_matrix_left: np.ndarray  # 3x3
    camera_matrix_right: np.ndarray  # 3x3

    # Distortion coefficients
    dist_coeffs_left: np.ndarray  # 1x5 or 1x8
    dist_coeffs_right: np.ndarray  # 1x5 or 1x8

    # Stereo parameters
    rotation_matrix: np.ndarray  # 3x3 rotation between cameras
    translation_vector: np.ndarray  # 3x1 translation between cameras

    # Rectification transforms (computed)
    rect_left: Optional[np.ndarray] = None  # 3x3
    rect_right: Optional[np.ndarray] = None  # 3x3
    proj_left: Optional[np.ndarray] = None  # 3x4
    proj_right: Optional[np.ndarray] = None  # 3x4
    disparity_to_depth: Optional[np.ndarray] = None  # 4x4

    # Undistort/rectify maps (computed for efficiency)
    _map_left_x: Optional[np.ndarray] = None
    _map_left_y: Optional[np.ndarray] = None
    _map_right_x: Optional[np.ndarray] = None
    _map_right_y: Optional[np.ndarray] = None

    def compute_rectification(self, image_size: tuple[int, int]) -> None:
        """
        Compute stereo rectification transforms.

        Args:
            image_size: (width, height) of the images
        """
        (
            self.rect_left,
            self.rect_right,
            self.proj_left,
            self.proj_right,
            self.disparity_to_depth,
            _,
            _,
        ) = cv2.stereoRectify(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            image_size,
            self.rotation_matrix,
            self.translation_vector,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        # Compute undistort/rectify maps for efficiency
        self._map_left_x, self._map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.rect_left,
            self.proj_left,
            image_size,
            cv2.CV_32FC1,
        )

        self._map_right_x, self._map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self.rect_right,
            self.proj_right,
            image_size,
            cv2.CV_32FC1,
        )

    def rectify_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rectify a stereo image pair.

        Args:
            left: Left camera image
            right: Right camera image

        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        if self._map_left_x is None:
            # Compute rectification maps if not done yet
            h, w = left.shape[:2]
            self.compute_rectification((w, h))

        rectified_left = cv2.remap(
            left,
            self._map_left_x,
            self._map_left_y,
            cv2.INTER_LINEAR,
        )

        rectified_right = cv2.remap(
            right,
            self._map_right_x,
            self._map_right_y,
            cv2.INTER_LINEAR,
        )

        return rectified_left, rectified_right

    def save(self, path: Path) -> None:
        """
        Save calibration to JSON file.

        Args:
            path: Path to save calibration data
        """
        data = {
            "camera_matrix_left": self.camera_matrix_left.tolist(),
            "camera_matrix_right": self.camera_matrix_right.tolist(),
            "dist_coeffs_left": self.dist_coeffs_left.tolist(),
            "dist_coeffs_right": self.dist_coeffs_right.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.tolist(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StereoCalibration":
        """
        Load calibration from JSON file.

        Args:
            path: Path to calibration JSON file

        Returns:
            StereoCalibration instance
        """
        with open(path) as f:
            data = json.load(f)

        return cls(
            camera_matrix_left=np.array(data["camera_matrix_left"]),
            camera_matrix_right=np.array(data["camera_matrix_right"]),
            dist_coeffs_left=np.array(data["dist_coeffs_left"]),
            dist_coeffs_right=np.array(data["dist_coeffs_right"]),
            rotation_matrix=np.array(data["rotation_matrix"]),
            translation_vector=np.array(data["translation_vector"]),
        )

    @classmethod
    def create_identity(cls) -> "StereoCalibration":
        """
        Create a default identity calibration (no distortion, no rotation).

        Useful as a starting point or for uncalibrated cameras.

        Returns:
            StereoCalibration with identity/zero parameters
        """
        # Default camera matrix (approximate for typical webcam)
        camera_matrix = np.array(
            [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0],
            ]
        )

        return cls(
            camera_matrix_left=camera_matrix.copy(),
            camera_matrix_right=camera_matrix.copy(),
            dist_coeffs_left=np.zeros(5),
            dist_coeffs_right=np.zeros(5),
            rotation_matrix=np.eye(3),
            translation_vector=np.array([[0.06], [0.0], [0.0]]),  # ~6cm baseline
        )
