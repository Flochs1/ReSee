"""Visual odometry using PnP with stereo depth."""

import cv2
import numpy as np
from typing import Optional, Tuple
from .feature_tracker import FeatureTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VisualOdometry:
    """
    Visual odometry using PnP with stereo depth for scale-aware motion estimation.

    Algorithm:
    1. Extract ORB features from current left frame
    2. Match with previous frame features
    3. Look up 3D positions of previous keypoints using depth map
    4. Solve PnP to get camera motion (rotation + translation)
    5. Accumulate pose over time
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        max_features: int = 500,
        min_features: int = 20,
        ransac_threshold: float = 1.0
    ):
        """
        Initialize visual odometry.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix.
            max_features: Maximum features for ORB extraction.
            min_features: Minimum features required for PnP.
            ransac_threshold: RANSAC reprojection threshold in pixels.
        """
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.ransac_threshold = ransac_threshold
        self.min_features = min_features

        # Feature tracker
        self.feature_tracker = FeatureTracker(
            max_features=max_features,
            min_features=min_features
        )

        # Previous depth map for 3D point lookup
        self.prev_depth_map: Optional[np.ndarray] = None

        # Cumulative rotation and translation (world pose)
        self.R_world = np.eye(3, dtype=np.float64)  # Rotation from world to current camera
        self.t_world = np.zeros((3, 1), dtype=np.float64)  # Position in world frame

        # Frame counter
        self.frame_count = 0

        logger.info(f"Visual odometry initialized (ransac={ransac_threshold})")

    def process_frame(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Process a new frame and estimate camera motion.

        Args:
            frame: Left camera frame (BGR or grayscale).
            depth_map: Depth map in meters (H x W float32).

        Returns:
            Tuple of (R_delta, t_delta, success).
            R_delta: 3x3 rotation matrix (frame-to-frame).
            t_delta: 3x1 translation vector in meters.
            success: True if motion was estimated successfully.
        """
        self.frame_count += 1

        # Extract and match features
        success, prev_kps, curr_kps, matches = self.feature_tracker.process_frame(frame)

        if not success or self.prev_depth_map is None:
            # First frame or not enough matches
            self.prev_depth_map = depth_map.copy()
            return None, None, False

        # Get 3D points from previous frame using depth
        pts_3d, pts_2d = self._get_3d_2d_correspondences(
            prev_kps, curr_kps, self.prev_depth_map
        )

        if len(pts_3d) < self.min_features:
            self.prev_depth_map = depth_map.copy()
            return None, None, False

        # Solve PnP
        R_delta, t_delta, success = self._solve_pnp(pts_3d, pts_2d)

        if success:
            # Update world pose
            self._update_world_pose(R_delta, t_delta)

        # Update previous depth map
        self.prev_depth_map = depth_map.copy()

        return R_delta, t_delta, success

    def _get_3d_2d_correspondences(
        self,
        prev_keypoints: list,
        curr_keypoints: list,
        depth_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D-2D point correspondences for PnP.

        Args:
            prev_keypoints: Keypoints from previous frame.
            curr_keypoints: Matched keypoints in current frame.
            depth_map: Depth map from previous frame.

        Returns:
            Tuple of (points_3d, points_2d) as numpy arrays.
        """
        pts_3d = []
        pts_2d = []

        h, w = depth_map.shape[:2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        for prev_kp, curr_kp in zip(prev_keypoints, curr_keypoints):
            # Get pixel coordinates
            px, py = int(prev_kp.pt[0]), int(prev_kp.pt[1])

            # Bounds check
            if px < 0 or px >= w or py < 0 or py >= h:
                continue

            # Get depth
            z = depth_map[py, px]

            # Skip invalid depths
            if z <= 0.1 or z > 15.0:
                continue

            # Back-project to 3D
            x = (px - cx) * z / fx
            y = (py - cy) * z / fy

            pts_3d.append([x, y, z])
            pts_2d.append(list(curr_kp.pt))

        return np.array(pts_3d, dtype=np.float64), np.array(pts_2d, dtype=np.float64)

    def _solve_pnp(
        self,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Solve PnP to get camera motion.

        Args:
            pts_3d: Nx3 array of 3D points.
            pts_2d: Nx2 array of 2D points.

        Returns:
            Tuple of (R, t, success).
        """
        if len(pts_3d) < 4:
            return None, None, False

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                self.camera_matrix,
                None,  # No distortion (already rectified)
                iterationsCount=100,
                reprojectionError=self.ransac_threshold,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success or inliers is None or len(inliers) < self.min_features:
                return None, None, False

            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)

            return R, tvec, True

        except cv2.error as e:
            logger.debug(f"PnP failed: {e}")
            return None, None, False

    def _update_world_pose(self, R_delta: np.ndarray, t_delta: np.ndarray) -> None:
        """
        Update cumulative world pose from frame-to-frame motion.

        Args:
            R_delta: Frame-to-frame rotation.
            t_delta: Frame-to-frame translation in camera coordinates.
        """
        # Transform translation to world coordinates
        t_world_delta = self.R_world.T @ t_delta

        # Update position (camera moved opposite to observed motion)
        self.t_world = self.t_world - t_world_delta

        # Update rotation
        self.R_world = R_delta @ self.R_world

    def get_position(self) -> Tuple[float, float]:
        """
        Get current camera position in world coordinates.

        Returns:
            (x, y) position in meters.
        """
        return float(self.t_world[0, 0]), float(self.t_world[2, 0])

    def get_heading(self) -> float:
        """
        Get current camera heading (yaw) in radians.

        Returns:
            Heading in radians (0 = initial direction, positive = clockwise).
        """
        # Extract yaw from rotation matrix
        # R_world transforms from world to camera
        # We want camera's heading in world frame
        return np.arctan2(self.R_world[0, 2], self.R_world[2, 2])

    def reset(self) -> None:
        """Reset odometry to origin."""
        self.R_world = np.eye(3, dtype=np.float64)
        self.t_world = np.zeros((3, 1), dtype=np.float64)
        self.prev_depth_map = None
        self.feature_tracker.reset()
        self.frame_count = 0
        logger.info("Visual odometry reset to origin")
