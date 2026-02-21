"""
ReSee - Stereo Camera Viewer with Depth Estimation

Stereo camera viewer with optional depth mapping.
Compatible with macOS, Linux, and Raspberry Pi.
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from dotenv import load_dotenv

from src.config.settings import get_settings
from src.utils.logger import setup_logger, get_logger
from src.utils.timing import FPSController, FrameTimer
from src.camera.stereo_capture import StereoCamera, StereoCameraError
from src.camera.display import VideoDisplay
from src.calibration import StereoCalibrator, DepthEstimator, CalibrationError
from src.gemini import GeminiClient, GeminiClientError


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ReSee - Stereo Camera Viewer with Depth Estimation"
    )
    parser.add_argument(
        '--recalibrate',
        action='store_true',
        help='Force recalibration even if calibration file exists'
    )
    parser.add_argument(
        '--no-depth',
        action='store_true',
        help='Disable depth estimation (stereo view only)'
    )
    return parser.parse_args()


class ReSeeApp:
    """Stereo camera viewer application with depth estimation."""

    def __init__(self, recalibrate: bool = False, no_depth: bool = False):
        """
        Initialize ReSee camera viewer.

        Args:
            recalibrate: Force recalibration even if data exists.
            no_depth: Disable depth estimation.
        """
        # CLI options
        self.recalibrate = recalibrate
        self.no_depth = no_depth

        # Load configuration
        self.settings = get_settings()

        # Setup logging
        self.logger = setup_logger(
            __name__,
            level=self.settings.logging.level,
            use_colors=self.settings.logging.console_colors
        )

        self.logger.info("=" * 60)
        self.logger.info("ReSee - Stereo Camera Viewer")
        self.logger.info("=" * 60)

        # Components
        self.camera: Optional[StereoCamera] = None
        self.display: Optional[VideoDisplay] = None
        self.depth_estimator: Optional[DepthEstimator] = None
        self.calibrator: Optional[StereoCalibrator] = None
        self.gemini_client: Optional[GeminiClient] = None

        # Timing
        self.fps_controller: Optional[FPSController] = None
        self.frame_timer: Optional[FrameTimer] = None

        # Control flags
        self.running = False
        self.depth_enabled = False
        self.gemini_available = False

        # Frame storage for Gemini analysis
        self.last_camera_frame: Optional[np.ndarray] = None
        self.last_depth_frame: Optional[np.ndarray] = None

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"\nReceived signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def initialize(self) -> bool:
        """
        Initialize camera, display, and depth estimation.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.logger.info("Initializing components...")

            # Initialize camera
            self.camera = StereoCamera(
                width=self.settings.camera.resolution.width,
                height=self.settings.camera.resolution.height,
                fps=self.settings.camera.fps,
                device_mode=self.settings.camera.device_mode,
                device_indices=tuple(self.settings.camera.device_indices),
                buffer_size=self.settings.camera.buffer_size
            )
            self.camera.open()
            self.logger.info(f"Camera initialized (mode: {self.camera.detected_mode})")

            # Initialize display
            if self.settings.display.preview_enabled:
                self.display = VideoDisplay(
                    window_name=self.settings.display.window_name,
                    scale=self.settings.display.scale,
                    show_fps=self.settings.display.show_fps,
                    show_status=self.settings.display.show_status
                )

                if self.display.is_available():
                    self.logger.info("Video display initialized")
                else:
                    self.logger.info("Display not available, running headless")
                    self.display = None
            else:
                self.logger.info("Video display disabled (headless mode)")
                self.display = None

            # Initialize timing controllers
            self.fps_controller = FPSController(self.settings.camera.fps)
            self.frame_timer = FrameTimer(window_size=30)

            # Initialize depth estimation if enabled
            if self.settings.depth.enabled and not self.no_depth:
                if not self._initialize_depth():
                    self.logger.warning("Depth estimation disabled due to initialization failure")
            elif self.no_depth:
                self.logger.info("Depth estimation disabled by --no-depth flag")
            else:
                self.logger.info("Depth estimation disabled in config")

            # Initialize Gemini client
            self._initialize_gemini()

            self.logger.info("All components initialized successfully")
            return True

        except StereoCameraError as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def _initialize_depth(self) -> bool:
        """
        Initialize depth estimation with calibration.

        Returns:
            True if depth estimation is ready, False otherwise.
        """
        depth_cfg = self.settings.depth
        calib_path = depth_cfg.calibration_file
        current_res = (
            self.settings.camera.resolution.width,
            self.settings.camera.resolution.height
        )

        self.calibrator = StereoCalibrator()

        # Check if we need to calibrate
        need_calibration = self.recalibrate or not self.calibrator.is_valid(
            calib_path, current_res
        )

        if need_calibration:
            self.logger.info("Starting stereo camera calibration...")

            if not self.display or not self.display.is_available():
                self.logger.error("Display required for calibration")
                return False

            # Capture calibration frames
            frames = self.calibrator.capture_calibration_frames(
                self.camera,
                num_frames=15
            )

            if not frames:
                self.logger.error("Calibration cancelled or failed")
                return False

            # Perform calibration
            if not self.calibrator.calibrate(frames):
                self.logger.error("Stereo calibration failed")
                return False

            # Save calibration
            if not self.calibrator.save(calib_path):
                self.logger.error("Failed to save calibration")
                return False

            self.logger.info("Calibration complete and saved")
        else:
            # Load existing calibration
            if not self.calibrator.load(calib_path):
                self.logger.error("Failed to load calibration")
                return False
            self.logger.info("Loaded existing calibration data")

        # Create depth estimator
        try:
            self.depth_estimator = DepthEstimator(
                calibration_data=self.calibrator.get_calibration_data(),
                baseline_mm=depth_cfg.baseline_mm,
                num_disparities=depth_cfg.num_disparities,
                block_size=depth_cfg.block_size,
                min_depth_m=depth_cfg.min_depth_m,
                max_depth_m=depth_cfg.max_depth_m
            )
            self.depth_enabled = True
            self.logger.info("Depth estimation enabled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create depth estimator: {e}")
            return False

    def _initialize_gemini(self) -> None:
        """Initialize Gemini client for obstacle analysis."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            self.logger.warning("GEMINI_API_KEY not found in .env - Gemini analysis disabled")
            return

        try:
            self.gemini_client = GeminiClient(api_key=api_key)
            self.gemini_available = True
            self.logger.info("Gemini client ready (press SPACE to analyze obstacles)")
        except GeminiClientError as e:
            self.logger.warning(f"Gemini initialization failed: {e}")

    def _analyze_with_gemini(self) -> None:
        """Send current frames to Gemini for obstacle analysis."""
        if not self.gemini_available or not self.gemini_client:
            self.logger.warning("Gemini not available")
            return

        if self.last_camera_frame is None or self.last_depth_frame is None:
            self.logger.warning("No frames available for analysis")
            return

        self.logger.info("Analyzing obstacles with Gemini...")
        print("\n" + "=" * 60)
        print("GEMINI OBSTACLE ANALYSIS")
        print("=" * 60)

        result = self.gemini_client.analyze_obstacles(
            self.last_camera_frame,
            self.last_depth_frame
        )

        if result:
            print(result)
        else:
            print("Failed to get analysis from Gemini")

        print("=" * 60 + "\n")

    def run_viewer(self) -> None:
        """Main camera viewing loop."""
        self.logger.info("Starting stereo camera viewer...")
        self.logger.info(f"Target FPS: {self.settings.camera.fps}")
        self.logger.info(f"Resolution: {self.settings.camera.resolution.width}x{self.settings.camera.resolution.height} per camera")
        if self.depth_enabled:
            self.logger.info("Depth estimation: ENABLED")
        else:
            self.logger.info("Depth estimation: DISABLED")
        if self.gemini_available:
            self.logger.info("Gemini analysis: ENABLED (press SPACE to analyze)")
        self.logger.info("Press Ctrl+C or ESC to stop")
        self.logger.info("-" * 60)

        self.running = True
        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                # Control frame rate
                self.fps_controller.wait()

                # Get frames from camera
                frame_pair = self.camera.get_frame(timeout=0.5)

                if frame_pair is None:
                    self.logger.warning("No frames received from camera")
                    continue

                left_frame, right_frame = frame_pair

                # Update frame timing
                self.frame_timer.tick()
                current_fps = self.frame_timer.get_fps()
                frame_count += 1

                # Display video if available
                if self.display and self.display.is_available():
                    target_w = self.settings.camera.resolution.width
                    target_h = self.settings.camera.resolution.height

                    # Resize frames if needed
                    if left_frame.shape[:2] != (target_h, target_w):
                        left_resized = cv2.resize(left_frame, (target_w, target_h))
                    else:
                        left_resized = left_frame

                    if right_frame.shape[:2] != (target_h, target_w):
                        right_resized = cv2.resize(right_frame, (target_w, target_h))
                    else:
                        right_resized = right_frame

                    # Combine side-by-side for stereo view
                    stereo_combined = np.hstack((left_resized, right_resized))

                    # Process depth if enabled
                    if self.depth_enabled and self.depth_estimator:
                        depth_colored, _ = self.depth_estimator.process_frame(
                            left_resized, right_resized, include_legend=True
                        )

                        # Store frames for Gemini analysis
                        self.last_camera_frame = left_resized.copy()
                        self.last_depth_frame = depth_colored.copy()

                        # Resize depth to match stereo width
                        stereo_width = stereo_combined.shape[1]
                        depth_height = depth_colored.shape[0]
                        depth_width = depth_colored.shape[1]

                        # Scale depth to match stereo width while preserving aspect ratio
                        scale_factor = stereo_width / depth_width
                        new_depth_height = int(depth_height * scale_factor)
                        depth_resized = cv2.resize(
                            depth_colored,
                            (stereo_width, new_depth_height)
                        )

                        # Stack stereo on top, depth below
                        combined_frame = np.vstack((stereo_combined, depth_resized))
                    else:
                        combined_frame = stereo_combined

                    # Create status
                    elapsed = time.time() - start_time
                    depth_status = " | Depth: ON" if self.depth_enabled else ""
                    status = f"Frames: {frame_count} | Elapsed: {elapsed:.1f}s{depth_status}"

                    self.display.show_frame(
                        combined_frame,
                        fps=current_fps,
                        status=status
                    )

                    # Check for key presses
                    key = self.display.check_key_press(1)
                    if key == 27:  # ESC
                        self.logger.info("ESC pressed, stopping...")
                        break
                    elif key == 32:  # SPACE
                        self._analyze_with_gemini()

                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"Frames: {frame_count}, Current FPS: {current_fps:.1f}, "
                        f"Average FPS: {avg_fps:.1f}"
                    )

        except Exception as e:
            self.logger.error(f"Error in viewer loop: {e}")

        finally:
            self.running = False

            # Final statistics
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0

            self.logger.info("-" * 60)
            self.logger.info("Session Statistics:")
            self.logger.info(f"  Total frames: {frame_count}")
            self.logger.info(f"  Total time: {elapsed:.1f}s")
            self.logger.info(f"  Average FPS: {avg_fps:.1f}")
            self.logger.info("Viewer stopped")

    def cleanup(self) -> None:
        """Cleanup all resources."""
        self.logger.info("Cleaning up resources...")

        # Close display
        if self.display:
            self.display.close()

        # Close camera
        if self.camera:
            self.camera.close()

        self.logger.info("Cleanup complete")

    def run(self) -> int:
        """
        Run the application.

        Returns:
            Exit code (0 = success, 1 = error).
        """
        # Setup signal handlers
        self.setup_signal_handlers()

        # Initialize components
        if not self.initialize():
            self.logger.error("Initialization failed")
            return 1

        try:
            # Run viewer
            self.run_viewer()

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return 1

        finally:
            # Cleanup
            self.cleanup()

        self.logger.info("ReSee terminated")
        return 0


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code.
    """
    args = parse_args()
    app = ReSeeApp(
        recalibrate=args.recalibrate,
        no_depth=args.no_depth
    )
    return app.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
