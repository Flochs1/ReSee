"""Stereo camera calibration module."""

import cv2
import numpy as np
import signal
import time
from pathlib import Path
from typing import Optional, Tuple, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalibrationError(Exception):
    """Exception raised for calibration errors."""
    pass


class StereoCalibrator:
    """Handles stereo camera calibration using checkerboard pattern."""

    def __init__(
        self,
        checkerboard_rows: int = 6,
        checkerboard_cols: int = 9,
        square_size_mm: float = 25.0
    ):
        """
        Initialize stereo calibrator.

        Args:
            checkerboard_rows: Number of inner corners vertically.
            checkerboard_cols: Number of inner corners horizontally.
            square_size_mm: Size of each checkerboard square in mm.
        """
        self.rows = checkerboard_rows
        self.cols = checkerboard_cols
        self.square_size = square_size_mm

        # Calibration data
        self.camera_matrix_left: Optional[np.ndarray] = None
        self.camera_matrix_right: Optional[np.ndarray] = None
        self.dist_coeffs_left: Optional[np.ndarray] = None
        self.dist_coeffs_right: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None  # Rotation between cameras
        self.T: Optional[np.ndarray] = None  # Translation between cameras
        self.rectify_map_left: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.rectify_map_right: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.Q: Optional[np.ndarray] = None  # Disparity-to-depth mapping matrix

        # Image size for which calibration was computed
        self.image_size: Optional[Tuple[int, int]] = None

    def generate_checkerboard(self, width: int = 800, height: int = 600) -> np.ndarray:
        """
        Generate a checkerboard pattern image for display.

        Args:
            width: Output image width.
            height: Output image height.

        Returns:
            Checkerboard image as numpy array.
        """
        # Calculate square size to fit the pattern in the image
        squares_h = self.cols + 1
        squares_v = self.rows + 1

        # Add margin
        margin = 50
        available_w = width - 2 * margin
        available_h = height - 2 * margin

        square_size = min(available_w // squares_h, available_h // squares_v)

        pattern_w = square_size * squares_h
        pattern_h = square_size * squares_v

        # Create white image
        img = np.ones((height, width), dtype=np.uint8) * 255

        # Calculate offset to center the pattern
        offset_x = (width - pattern_w) // 2
        offset_y = (height - pattern_h) // 2

        # Draw checkerboard
        for row in range(squares_v):
            for col in range(squares_h):
                if (row + col) % 2 == 0:
                    x1 = offset_x + col * square_size
                    y1 = offset_y + row * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    img[y1:y2, x1:x2] = 0

        # Convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Add text instructions
        cv2.putText(
            img_bgr,
            f"Checkerboard: {self.cols}x{self.rows} inner corners",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        cv2.putText(
            img_bgr,
            "Position this pattern in front of the camera",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        return img_bgr

    def find_corners(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find checkerboard corners in both frames.

        Args:
            left_frame: Left camera frame.
            right_frame: Right camera frame.

        Returns:
            (success, left_corners, right_corners) tuple.
        """
        # Convert to grayscale
        if len(left_frame.shape) == 3:
            gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = left_frame

        if len(right_frame.shape) == 3:
            gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = right_frame

        # Find corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, (self.cols, self.rows), flags
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, (self.cols, self.rows), flags
        )

        if ret_left and ret_right:
            # Refine corners for better accuracy
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
            corners_left = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), criteria
            )
            corners_right = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), criteria
            )
            return True, corners_left, corners_right

        return False, None, None

    def capture_calibration_frames(
        self,
        camera,
        num_frames: int = 15,
        window_name: str = "Calibration"
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Interactive capture of calibration frames.

        Args:
            camera: StereoCamera instance.
            num_frames: Number of calibration frames to capture.
            window_name: Name of the display window.

        Returns:
            List of (left_frame, right_frame, left_corners, right_corners) tuples.
        """
        # Generate and display checkerboard pattern
        pattern = self.generate_checkerboard(800, 600)
        pattern_window = "Calibration Pattern"
        cv2.namedWindow(pattern_window, cv2.WINDOW_NORMAL)
        cv2.imshow(pattern_window, pattern)

        # Create preview window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        captured_frames = []
        last_capture_time = 0
        min_capture_interval = 0.5  # Minimum seconds between captures
        cancelled = False

        def handle_sigint(signum, frame):
            nonlocal cancelled
            cancelled = True

        # Set up Ctrl+C handler
        original_handler = signal.signal(signal.SIGINT, handle_sigint)

        logger.info(f"Starting calibration capture. Need {num_frames} frames.")
        logger.info("Hold checkerboard in view - auto-captures every 0.5s")
        logger.info("Move pattern to different angles. Press ESC to cancel.")

        while len(captured_frames) < num_frames and not cancelled:
            # Get frames
            frame_pair = camera.get_frame(timeout=0.5)
            if frame_pair is None:
                continue

            left_frame, right_frame = frame_pair

            # Try to find corners
            success, corners_left, corners_right = self.find_corners(
                left_frame, right_frame
            )

            # Create preview
            preview_left = left_frame.copy()
            preview_right = right_frame.copy()

            if success:
                # Draw detected corners
                cv2.drawChessboardCorners(
                    preview_left, (self.cols, self.rows), corners_left, success
                )
                cv2.drawChessboardCorners(
                    preview_right, (self.cols, self.rows), corners_right, success
                )
                status_color = (0, 255, 0)  # Green
                status_text = "Pattern detected - auto-capturing..."
            else:
                status_color = (0, 0, 255)  # Red
                status_text = "No pattern detected"

            # Combine previews
            combined = np.hstack((preview_left, preview_right))

            # Scale for display
            scale = 0.5
            h, w = combined.shape[:2]
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

            # Add status text
            cv2.putText(
                combined,
                f"Captured: {len(captured_frames)}/{num_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            cv2.putText(
                combined,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )

            cv2.imshow(window_name, combined)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                cancelled = True
                break

            current_time = time.time()

            # Auto-capture when pattern detected (no key press needed)
            if success:
                if current_time - last_capture_time >= min_capture_interval:
                    captured_frames.append((
                        left_frame.copy(),
                        right_frame.copy(),
                        corners_left,
                        corners_right
                    ))
                    last_capture_time = current_time
                    logger.info(f"Captured frame {len(captured_frames)}/{num_frames}")

                    # Flash effect (also check for ESC during flash)
                    flash = np.ones_like(combined) * 255
                    cv2.imshow(window_name, flash)
                    flash_key = cv2.waitKey(100) & 0xFF
                    if flash_key == 27:
                        cancelled = True
                        break

        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)

        cv2.destroyWindow(pattern_window)
        cv2.destroyWindow(window_name)

        if cancelled:
            logger.info("Calibration cancelled by user")
            return []

        logger.info(f"Captured {len(captured_frames)} calibration frames")
        return captured_frames

    def calibrate(
        self,
        frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> bool:
        """
        Perform stereo calibration using captured frames.

        Args:
            frames: List of (left_frame, right_frame, left_corners, right_corners).

        Returns:
            True if calibration successful, False otherwise.
        """
        if len(frames) < 5:
            logger.error("Need at least 5 calibration frames")
            return False

        logger.info(f"Starting stereo calibration with {len(frames)} frames...")

        # Prepare object points (3D points in real world space)
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        objp *= self.square_size

        obj_points = []  # 3D points
        img_points_left = []  # 2D points in left image
        img_points_right = []  # 2D points in right image

        for left_frame, right_frame, corners_left, corners_right in frames:
            obj_points.append(objp)
            img_points_left.append(corners_left)
            img_points_right.append(corners_right)

        # Get image size from first frame
        self.image_size = (frames[0][0].shape[1], frames[0][0].shape[0])

        logger.info("Calibrating individual cameras...")

        # Calibrate left camera
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, _, _ = \
            cv2.calibrateCamera(
                obj_points, img_points_left, self.image_size, None, None
            )

        # Calibrate right camera
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, _, _ = \
            cv2.calibrateCamera(
                obj_points, img_points_right, self.image_size, None, None
            )

        logger.info(f"Left camera RMS error: {ret_left:.4f}")
        logger.info(f"Right camera RMS error: {ret_right:.4f}")

        logger.info("Performing stereo calibration...")

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-5
        )

        ret_stereo, _, _, _, _, self.R, self.T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_left,
            img_points_right,
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self.image_size,
            criteria=criteria,
            flags=flags
        )

        logger.info(f"Stereo calibration RMS error: {ret_stereo:.4f}")

        # Compute rectification transforms
        logger.info("Computing rectification maps...")

        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self.image_size,
            self.R,
            self.T,
            alpha=0
        )

        # Compute rectification maps
        self.rectify_map_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            R1,
            P1,
            self.image_size,
            cv2.CV_32FC1
        )

        self.rectify_map_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right,
            self.dist_coeffs_right,
            R2,
            P2,
            self.image_size,
            cv2.CV_32FC1
        )

        logger.info("Stereo calibration complete!")
        return True

    def save(self, path: str) -> bool:
        """
        Save calibration data to file.

        Args:
            path: Path to save calibration file (.npz).

        Returns:
            True if saved successfully, False otherwise.
        """
        if self.camera_matrix_left is None:
            logger.error("No calibration data to save")
            return False

        try:
            # Create directory if needed
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            np.savez(
                path,
                camera_matrix_left=self.camera_matrix_left,
                camera_matrix_right=self.camera_matrix_right,
                dist_coeffs_left=self.dist_coeffs_left,
                dist_coeffs_right=self.dist_coeffs_right,
                R=self.R,
                T=self.T,
                Q=self.Q,
                rectify_map_left_x=self.rectify_map_left[0],
                rectify_map_left_y=self.rectify_map_left[1],
                rectify_map_right_x=self.rectify_map_right[0],
                rectify_map_right_y=self.rectify_map_right[1],
                image_size=np.array(self.image_size)
            )

            logger.info(f"Calibration saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load calibration data from file.

        Args:
            path: Path to calibration file (.npz).

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            data = np.load(path)

            self.camera_matrix_left = data['camera_matrix_left']
            self.camera_matrix_right = data['camera_matrix_right']
            self.dist_coeffs_left = data['dist_coeffs_left']
            self.dist_coeffs_right = data['dist_coeffs_right']
            self.R = data['R']
            self.T = data['T']
            self.Q = data['Q']
            self.rectify_map_left = (
                data['rectify_map_left_x'],
                data['rectify_map_left_y']
            )
            self.rectify_map_right = (
                data['rectify_map_right_x'],
                data['rectify_map_right_y']
            )
            self.image_size = tuple(data['image_size'])

            logger.info(f"Calibration loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def is_valid(self, path: str, current_resolution: Tuple[int, int]) -> bool:
        """
        Check if calibration file exists and matches current resolution.

        Args:
            path: Path to calibration file.
            current_resolution: Current camera resolution (width, height).

        Returns:
            True if calibration is valid for current resolution.
        """
        if not Path(path).exists():
            return False

        try:
            data = np.load(path)
            stored_size = tuple(data['image_size'])
            return stored_size == current_resolution
        except Exception:
            return False

    def get_calibration_data(self) -> dict:
        """
        Get calibration data as dictionary.

        Returns:
            Dictionary containing calibration matrices.
        """
        return {
            'camera_matrix_left': self.camera_matrix_left,
            'camera_matrix_right': self.camera_matrix_right,
            'dist_coeffs_left': self.dist_coeffs_left,
            'dist_coeffs_right': self.dist_coeffs_right,
            'R': self.R,
            'T': self.T,
            'Q': self.Q,
            'rectify_map_left': self.rectify_map_left,
            'rectify_map_right': self.rectify_map_right,
            'image_size': self.image_size
        }
