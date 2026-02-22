"""Visual odometry using PnP with stereo depth."""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from .feature_tracker import FeatureTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VisualOdometry:
    """
    Visual odometry using PnP with stereo depth for scale-aware motion estimation.

    Includes panic mode: when tracking fails, keeps comparing against the last
    good frame until a new frame matches well enough to resume tracking.
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

        # ORB detector (we manage features ourselves for panic mode)
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )

        # BFMatcher for ORB (binary descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Last good frame data (used as reference)
        self.last_good_keypoints: Optional[List[cv2.KeyPoint]] = None
        self.last_good_descriptors: Optional[np.ndarray] = None
        self.last_good_depth_map: Optional[np.ndarray] = None

        # Cumulative rotation and translation (world pose)
        self.R_world = np.eye(3, dtype=np.float64)
        self.t_world = np.zeros((3, 1), dtype=np.float64)

        # Panic mode state
        self.in_panic = False
        self.frame_count = 0

        logger.info(f"Visual odometry initialized (ransac={ransac_threshold}, min_features={min_features})")

    def process_frame(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Process a new frame and estimate camera motion.

        When tracking fails, enters panic mode and keeps trying to match
        against the last good frame until tracking recovers.

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

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Extract features from current frame
        curr_keypoints, curr_descriptors = self.orb.detectAndCompute(gray, None)

        if curr_descriptors is None or len(curr_keypoints) < self.min_features:
            # Not enough features in current frame
            if not self.in_panic and self.last_good_descriptors is not None:
                self.in_panic = True
                logger.warning("Tracking LOST - entering panic mode (not enough features)")
            return None, None, False

        # First frame - initialize
        if self.last_good_descriptors is None:
            self.last_good_keypoints = curr_keypoints
            self.last_good_descriptors = curr_descriptors
            self.last_good_depth_map = depth_map.copy()
            return None, None, False

        # Try to match against last good frame
        try:
            matches = self.matcher.match(self.last_good_descriptors, curr_descriptors)
        except cv2.error:
            if not self.in_panic:
                self.in_panic = True
                logger.warning("Tracking LOST - entering panic mode (match failed)")
            return None, None, False

        if len(matches) < self.min_features:
            # Not enough matches - stay in or enter panic mode
            if not self.in_panic:
                self.in_panic = True
                logger.warning(f"Tracking LOST - entering panic mode ({len(matches)} matches < {self.min_features})")
            return None, None, False

        # Sort by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        prev_matched = [self.last_good_keypoints[m.queryIdx] for m in matches]
        curr_matched = [curr_keypoints[m.trainIdx] for m in matches]

        # Get 3D-2D correspondences
        pts_3d, pts_2d = self._get_3d_2d_correspondences(
            prev_matched, curr_matched, self.last_good_depth_map
        )

        if len(pts_3d) < self.min_features:
            if not self.in_panic:
                self.in_panic = True
                logger.warning(f"Tracking LOST - entering panic mode ({len(pts_3d)} 3D points < {self.min_features})")
            return None, None, False

        # Solve PnP
        R_delta, t_delta, success = self._solve_pnp(pts_3d, pts_2d)

        if not success:
            if not self.in_panic:
                self.in_panic = True
                logger.warning("Tracking LOST - entering panic mode (PnP failed)")
            return None, None, False

        # Success! Update world pose
        self._update_world_pose(R_delta, t_delta)

        # Update last good frame to current
        self.last_good_keypoints = curr_keypoints
        self.last_good_descriptors = curr_descriptors
        self.last_good_depth_map = depth_map.copy()

        # Exit panic mode if we were in it
        if self.in_panic:
            self.in_panic = False
            logger.info("Tracking RECOVERED - exiting panic mode")

        return R_delta, t_delta, True

    def _get_3d_2d_correspondences(
        self,
        prev_keypoints: list,
        curr_keypoints: list,
        depth_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D-2D point correspondences for PnP."""
        pts_3d = []
        pts_2d = []

        h, w = depth_map.shape[:2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        for prev_kp, curr_kp in zip(prev_keypoints, curr_keypoints):
            px, py = int(prev_kp.pt[0]), int(prev_kp.pt[1])

            if px < 0 or px >= w or py < 0 or py >= h:
                continue

            z = depth_map[py, px]

            if z <= 0.1 or z > 15.0:
                continue

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
        """Solve PnP to get camera motion."""
        if len(pts_3d) < 4:
            return None, None, False

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                self.camera_matrix,
                None,
                iterationsCount=100,
                reprojectionError=self.ransac_threshold,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success or inliers is None or len(inliers) < self.min_features:
                return None, None, False

            R, _ = cv2.Rodrigues(rvec)
            return R, tvec, True

        except cv2.error as e:
            logger.debug(f"PnP failed: {e}")
            return None, None, False

    def _update_world_pose(self, R_delta: np.ndarray, t_delta: np.ndarray) -> None:
        """Update cumulative world pose from frame-to-frame motion."""
        t_world_delta = self.R_world.T @ t_delta
        self.t_world = self.t_world - t_world_delta
        self.R_world = R_delta @ self.R_world

    def get_position(self) -> Tuple[float, float]:
        """Get current camera position in world coordinates."""
        return float(self.t_world[0, 0]), float(self.t_world[2, 0])

    def get_heading(self) -> float:
        """Get current camera heading (yaw) in radians."""
        return np.arctan2(self.R_world[0, 2], self.R_world[2, 2])

    def is_tracking(self) -> bool:
        """Return True if tracking normally, False if in panic mode."""
        return not self.in_panic

    def reset(self) -> None:
        """Reset odometry to origin."""
        self.R_world = np.eye(3, dtype=np.float64)
        self.t_world = np.zeros((3, 1), dtype=np.float64)
        self.last_good_keypoints = None
        self.last_good_descriptors = None
        self.last_good_depth_map = None
        self.in_panic = False
        self.frame_count = 0
        logger.info("Visual odometry reset to origin")
