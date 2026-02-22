"""ORB feature extraction and matching for visual odometry."""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureTracker:
    """
    ORB-based feature extraction and frame-to-frame matching.

    Uses ORB (Oriented FAST and Rotated BRIEF) for fast feature detection
    and BFMatcher with Hamming distance for matching.
    """

    def __init__(
        self,
        max_features: int = 500,
        min_features: int = 20
    ):
        """
        Initialize feature tracker.

        Args:
            max_features: Maximum number of features to extract.
            min_features: Minimum features required for valid tracking.
        """
        self.max_features = max_features
        self.min_features = min_features

        # Create ORB detector
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

        # Create BFMatcher with Hamming distance (for binary descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Previous frame data
        self.prev_keypoints: Optional[List[cv2.KeyPoint]] = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None

        logger.info(f"Feature tracker initialized (max={max_features}, min={min_features})")

    def extract_features(
        self,
        frame: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Extract ORB features from a frame.

        Args:
            frame: Grayscale or BGR image.

        Returns:
            Tuple of (keypoints, descriptors). Descriptors may be None if no features found.
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(
        self,
        keypoints: List[cv2.KeyPoint],
        descriptors: np.ndarray
    ) -> Tuple[List[cv2.DMatch], List[cv2.KeyPoint], List[cv2.KeyPoint]]:
        """
        Match current features with previous frame features.

        Args:
            keypoints: Current frame keypoints.
            descriptors: Current frame descriptors.

        Returns:
            Tuple of (matches, prev_matched_keypoints, curr_matched_keypoints).
        """
        if self.prev_descriptors is None or descriptors is None:
            return [], [], []

        if len(descriptors) < self.min_features or len(self.prev_descriptors) < self.min_features:
            return [], [], []

        # Match descriptors
        try:
            matches = self.matcher.match(self.prev_descriptors, descriptors)
        except cv2.error:
            return [], [], []

        if len(matches) < self.min_features:
            return [], [], []

        # Sort by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        prev_matched = [self.prev_keypoints[m.queryIdx] for m in matches]
        curr_matched = [keypoints[m.trainIdx] for m in matches]

        return matches, prev_matched, curr_matched

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[bool, List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
        """
        Process a new frame: extract features and match with previous frame.

        Args:
            frame: Input frame (grayscale or BGR).

        Returns:
            Tuple of (success, prev_keypoints, curr_keypoints, matches).
            success is True if enough matches were found.
        """
        # Extract features from current frame
        keypoints, descriptors = self.extract_features(frame)

        if descriptors is None or len(keypoints) < self.min_features:
            # Not enough features, but still update for next frame
            self.prev_keypoints = keypoints if keypoints else []
            self.prev_descriptors = descriptors
            self.prev_frame = frame.copy()
            return False, [], [], []

        # Match with previous frame
        matches, prev_matched, curr_matched = self.match_features(keypoints, descriptors)

        success = len(matches) >= self.min_features

        # Update previous frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame = frame.copy()

        return success, prev_matched, curr_matched, matches

    def reset(self) -> None:
        """Reset tracker state (clears previous frame data)."""
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        logger.debug("Feature tracker reset")
