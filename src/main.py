"""
ReSee - Stereo Camera Viewer with Depth Estimation

Stereo camera viewer with optional depth mapping.
Compatible with macOS, Linux, and Raspberry Pi.
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from src.config.settings import get_settings
from src.utils.logger import setup_logger, get_logger
from src.utils.timing import FPSController, FrameTimer
from src.camera.stereo_capture import StereoCamera, StereoCameraError
from src.camera.display import VideoDisplay
from src.calibration import StereoCalibrator, DepthEstimator, CalibrationError


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
    parser.add_argument(
        '--no-detection',
        action='store_true',
        help='Disable object detection'
    )
    return parser.parse_args()


class ReSeeApp:
    """Stereo camera viewer application with depth estimation."""

    def __init__(self, recalibrate: bool = False, no_depth: bool = False, no_detection: bool = False):
        """
        Initialize ReSee camera viewer.

        Args:
            recalibrate: Force recalibration even if data exists.
            no_depth: Disable depth estimation.
            no_detection: Disable object detection.
        """
        # CLI options
        self.recalibrate = recalibrate
        self.no_depth = no_depth
        self.no_detection = no_detection

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
        self.detection_pipeline = None  # Initialized in initialize()
        self.birdseye_view = None  # Initialized in initialize()
        self.world_map = None  # Initialized in initialize()

        # Timing
        self.fps_controller: Optional[FPSController] = None
        self.frame_timer: Optional[FrameTimer] = None

        # Control flags
        self.running = False
        self.depth_enabled = False
        self.detection_enabled = False

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

            # Initialize detection pipeline if enabled
            if not self.no_detection:
                self._initialize_detection()
                # Initialize bird's eye view and world map if detection is enabled
                if self.detection_enabled:
                    from src.detection import BirdsEyeView, WorldMap
                    self.birdseye_view = BirdsEyeView(
                        max_depth_m=15.0  # Always show 15m range
                    )
                    self.world_map = WorldMap(
                        fov_degrees=75.0,
                        anchor_persistence_seconds=5.0,
                        heading_smoothing=0.1
                    )
                    self.logger.info("Bird's eye view with world tracking enabled")
            else:
                self.logger.info("Object detection disabled by --no-detection flag")

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

    def _initialize_detection(self) -> bool:
        """
        Initialize object detection pipeline.

        Returns:
            True if detection is ready, False otherwise.
        """
        try:
            from src.detection import DetectionPipeline
            self.detection_pipeline = DetectionPipeline(enabled=True)
            self.detection_enabled = self.detection_pipeline.is_enabled()
            if self.detection_enabled:
                self.logger.info("Object detection enabled")
            return self.detection_enabled
        except ImportError as e:
            self.logger.warning(f"Detection not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize detection: {e}")
            return False

    def run_viewer(self) -> None:
        """Main camera viewing loop."""
        self.logger.info("Starting stereo camera viewer...")
        self.logger.info(f"Target FPS: {self.settings.camera.fps}")
        self.logger.info(f"Resolution: {self.settings.camera.resolution.width}x{self.settings.camera.resolution.height} per camera")
        if self.depth_enabled:
            self.logger.info("Depth estimation: ENABLED")
        else:
            self.logger.info("Depth estimation: DISABLED")
        if self.detection_enabled:
            self.logger.info("Object detection: ENABLED")
        else:
            self.logger.info("Object detection: DISABLED")
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

                    # Process depth if enabled
                    depth_map = None
                    if self.depth_enabled and self.depth_estimator:
                        depth_colored, depth_map = self.depth_estimator.process_frame(
                            left_resized, right_resized, include_legend=True
                        )

                    # Process detection if enabled
                    tracks = []
                    if self.detection_enabled and self.detection_pipeline:
                        left_resized, tracks = self.detection_pipeline.process(
                            left_resized, depth_map
                        )

                    # Combine side-by-side for stereo view
                    stereo_combined = np.hstack((left_resized, right_resized))

                    # Stack depth visualization if available
                    if self.depth_enabled and self.depth_estimator:
                        stereo_width = stereo_combined.shape[1]
                        depth_height = depth_colored.shape[0]
                        depth_width = depth_colored.shape[1]

                        scale_factor = stereo_width / depth_width
                        new_depth_height = int(depth_height * scale_factor)
                        depth_resized = cv2.resize(
                            depth_colored,
                            (stereo_width, new_depth_height)
                        )

                        combined_frame = np.vstack((stereo_combined, depth_resized))
                    else:
                        combined_frame = stereo_combined

                    # Add bird's eye view if detection is enabled (always show, even with no objects)
                    if self.birdseye_view and self.world_map:
                        # Update world map with tracked objects
                        current_time = time.monotonic()
                        world_objects = self.world_map.update(
                            tracks,
                            frame_width=target_w,
                            timestamp=current_time
                        )
                        camera_state = self.world_map.get_camera_state()

                        # Render world-fixed bird's eye view
                        birdseye_frame = self.birdseye_view.render_world(
                            world_objects,
                            camera_state,
                            current_time=current_time
                        )
                        # Scale bird's eye view to match combined frame width
                        combined_width = combined_frame.shape[1]
                        bev_scale = combined_width / birdseye_frame.shape[1]
                        bev_height = int(birdseye_frame.shape[0] * bev_scale)
                        birdseye_resized = cv2.resize(
                            birdseye_frame,
                            (combined_width, bev_height)
                        )
                        combined_frame = np.vstack((combined_frame, birdseye_resized))

                    # Create status
                    elapsed = time.time() - start_time
                    depth_status = " | Depth: ON" if self.depth_enabled else ""
                    detection_status = " | Detection: ON" if self.detection_enabled else ""
                    status = f"Frames: {frame_count} | Elapsed: {elapsed:.1f}s{depth_status}{detection_status}"

                    self.display.show_frame(
                        combined_frame,
                        fps=current_fps,
                        status=status
                    )

                    # Check for ESC key to quit
                    key = self.display.check_key_press(1)
                    if key == 27:  # ESC
                        self.logger.info("ESC pressed, stopping...")
                        break

                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"Frames: {frame_count}, FPS: {current_fps:.1f}, Avg: {avg_fps:.1f}"
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
        no_depth=args.no_depth,
        no_detection=args.no_detection
    )
    return app.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
