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

from src.config.settings import get_settings
from src.utils.logger import setup_logger, get_logger
from src.utils.timing import FPSController, FrameTimer
from src.camera.stereo_capture import StereoCamera, StereoCameraError
from src.camera.display import VideoDisplay
from src.calibration import StereoCalibrator, DepthEstimator, CalibrationError
from src.odometry import VisualOdometry, WorldState
from src.gemini.navigator import GeminiNavigator
from src.voice import VoiceInterface, GeminiVoiceInterface, GEMINI_VOICE_AVAILABLE


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
    parser.add_argument(
        '--no-navigation',
        action='store_true',
        help='Disable Gemini navigation assistance'
    )
    return parser.parse_args()


class ReSeeApp:
    """Stereo camera viewer application with depth estimation."""

    def __init__(
        self,
        recalibrate: bool = False,
        no_depth: bool = False,
        no_detection: bool = False,
        no_navigation: bool = False
    ):
        """
        Initialize ReSee camera viewer.

        Args:
            recalibrate: Force recalibration even if data exists.
            no_depth: Disable depth estimation.
            no_detection: Disable object detection.
            no_navigation: Disable Gemini navigation assistance.
        """
        # CLI options
        self.recalibrate = recalibrate
        self.no_depth = no_depth
        self.no_detection = no_detection
        self.no_navigation = no_navigation

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
        self.visual_odometry: Optional[VisualOdometry] = None
        self.world_state: Optional[WorldState] = None
        self.navigator: Optional[GeminiNavigator] = None
        self.voice_interface: Optional[VoiceInterface] = None

        # Timing
        self.fps_controller: Optional[FPSController] = None
        self.frame_timer: Optional[FrameTimer] = None

        # Control flags
        self.running = False
        self.depth_enabled = False
        self.detection_enabled = False
        self.odometry_enabled = False
        self.navigation_enabled = False
        self.voice_enabled = False

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
                # Initialize bird's eye view if detection is enabled
                if self.detection_enabled:
                    from src.detection import BirdsEyeView
                    self.birdseye_view = BirdsEyeView(
                        max_depth_m=15.0  # Always show 15m range
                    )
                    self.logger.info("Bird's eye view enabled")
            else:
                self.logger.info("Object detection disabled by --no-detection flag")

            # Initialize navigation assistant if available
            self._initialize_navigation()

            # Initialize voice interface if navigation is enabled
            if self.navigation_enabled:
                self._initialize_voice()

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

            # Initialize visual odometry if enabled
            if self.settings.odometry.enabled:
                self._initialize_odometry()

            return True

        except Exception as e:
            self.logger.error(f"Failed to create depth estimator: {e}")
            return False

    def _initialize_odometry(self) -> bool:
        """
        Initialize visual odometry.

        Returns:
            True if odometry is ready, False otherwise.
        """
        try:
            odom_cfg = self.settings.odometry
            camera_matrix = self.calibrator.camera_matrix_left

            if camera_matrix is None:
                self.logger.warning("No camera matrix available for odometry")
                return False

            self.visual_odometry = VisualOdometry(
                camera_matrix=camera_matrix,
                max_features=odom_cfg.max_features,
                min_features=odom_cfg.min_features,
                ransac_threshold=odom_cfg.ransac_threshold
            )

            self.world_state = WorldState(
                max_history=odom_cfg.max_trajectory_history
            )

            self.odometry_enabled = True
            self.logger.info("Visual odometry enabled")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize odometry: {e}")
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

    def _initialize_navigation(self) -> bool:
        """
        Initialize Gemini navigation assistant.

        Returns:
            True if navigation is ready, False otherwise.
        """
        if self.no_navigation:
            self.logger.info("Navigation disabled by --no-navigation flag")
            return False

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self.logger.info("Navigation disabled (GEMINI_API_KEY not set)")
            return False

        try:
            self.navigator = GeminiNavigator(
                api_key=api_key,
                routine_interval=1.0,
                danger_zone_m=5.0,
                closing_speed_threshold=0.5,
                enable_routine=False  # Disable heartbeat, only trigger on danger or voice
            )
            self.navigation_enabled = True
            self.logger.info("Gemini navigation enabled")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize navigation: {e}")
            return False

    def _initialize_voice(self) -> bool:
        """
        Initialize voice interface for TTS and speech recognition.

        Returns:
            True if voice is ready, False otherwise.
        """
        # Use local Whisper-based voice interface (Gemini voice disabled)
        try:
            self.voice_interface = VoiceInterface(use_whisper=True)
            self.voice_interface.start()

            if self.navigator:
                self.navigator.set_voice_interface(self.voice_interface)

            self.voice_enabled = True
            self.logger.info("Local Whisper voice interface enabled (say 'Resee' to speak)")
            return True
        except Exception as e:
            self.logger.warning(f"Voice interface not available: {e}")
            return False

    def _create_voice_bottom_panel(self, width: int) -> np.ndarray:
        """
        Create voice monitor panel for bottom of screen.

        Args:
            width: Width of the panel (full screen width).

        Returns:
            Voice panel as numpy array.
        """
        # Fixed height - voice panel at bottom
        panel_h = 400
        panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        if not self.voice_interface:
            cv2.putText(panel, "Voice interface not available", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            return panel

        margin = 30
        bar_height = 60

        # Title
        cv2.putText(panel, "VOICE INTERFACE", (margin, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        # Status indicator (large circle)
        indicator_x = width - margin - 30
        indicator_color = (0, 255, 0) if self.voice_interface.is_listening else (80, 80, 80)
        cv2.circle(panel, (indicator_x, 35), 20, indicator_color, -1)
        status_text = "LISTENING" if self.voice_interface.is_listening else "IDLE"
        cv2.putText(panel, status_text, (indicator_x - 80, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, indicator_color, 1)

        # Row 1: Audio level bar (large)
        y1 = 70
        label_w = 180
        cv2.putText(panel, "AUDIO LEVEL", (margin, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        bar_x = margin + label_w
        bar_w = width - bar_x - margin
        cv2.rectangle(panel, (bar_x, y1), (bar_x + bar_w, y1 + bar_height), (50, 50, 50), -1)

        # Audio energy is now 0-1 normalized from the monitor thread
        energy = min(1.0, self.voice_interface.audio_energy)
        fill_w = int(energy * bar_w)
        if fill_w > 0:
            # Color gradient: green -> yellow -> red based on level
            if energy < 0.3:
                color = (0, 180, 0)
            elif energy < 0.6:
                color = (0, 255, 100)
            elif energy < 0.8:
                color = (0, 255, 255)
            else:
                color = (0, 100, 255)
            cv2.rectangle(panel, (bar_x, y1), (bar_x + fill_w, y1 + bar_height), color, -1)

        cv2.rectangle(panel, (bar_x, y1), (bar_x + bar_w, y1 + bar_height), (80, 80, 80), 2)

        # Energy value text
        energy_pct = int(energy * 100)
        cv2.putText(panel, f"{energy_pct}%", (bar_x + bar_w + 10, y1 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        # Row 2: Transcription area (larger)
        y2 = 160
        cv2.putText(panel, "TRANSCRIPT", (margin, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        transcript_h = 80
        cv2.rectangle(panel, (bar_x, y2), (bar_x + bar_w, y2 + transcript_h), (40, 40, 40), -1)

        # Show transcription text (larger)
        text = self.voice_interface.last_heard if self.voice_interface.last_heard else "Say 'Resee' to ask a question..."
        text_color = (100, 255, 100) if self.voice_interface.last_heard else (100, 100, 100)
        cv2.putText(panel, text[:60], (bar_x + 15, y2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        if len(text) > 60:
            cv2.putText(panel, text[60:120], (bar_x + 15, y2 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

        cv2.rectangle(panel, (bar_x, y2), (bar_x + bar_w, y2 + transcript_h), (80, 80, 80), 2)

        # Row 3: Status message
        y3 = 270
        cv2.putText(panel, "STATUS", (margin, y3 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        status_h = 50
        cv2.rectangle(panel, (bar_x, y3), (bar_x + bar_w, y3 + status_h), (35, 35, 35), -1)

        status = self.voice_interface.status
        cv2.putText(panel, status, (bar_x + 15, y3 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.rectangle(panel, (bar_x, y3), (bar_x + bar_w, y3 + status_h), (80, 80, 80), 2)

        # Row 4: Instructions
        y4 = 350
        cv2.putText(panel, "Say 'Resee' followed by your question, or press 'V' for push-to-talk",
                    (margin, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        cv2.putText(panel, f"Threshold: {self.voice_interface.energy_threshold:.0f}",
                    (width - 200, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return panel

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
        if self.odometry_enabled:
            self.logger.info("Visual odometry: ENABLED")
        else:
            self.logger.info("Visual odometry: DISABLED")
        if self.navigation_enabled:
            self.logger.info("Navigation assistant: ENABLED")
        else:
            self.logger.info("Navigation assistant: DISABLED")
        if self.voice_enabled:
            self.logger.info("Voice interface: ENABLED (say 'Resee' to ask)")
        else:
            self.logger.info("Voice interface: DISABLED")
        self.logger.info("Press Ctrl+C or ESC to stop")
        if self.odometry_enabled:
            self.logger.info("Press 'R' to reset odometry")
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
                    depth_colored = None
                    if self.depth_enabled and self.depth_estimator:
                        depth_colored, depth_map = self.depth_estimator.process_frame(
                            left_resized, right_resized, include_legend=True
                        )

                    # Process visual odometry if enabled
                    if self.odometry_enabled and self.visual_odometry and depth_map is not None:
                        R, t, vo_success = self.visual_odometry.process_frame(
                            left_resized, depth_map
                        )
                        if vo_success:
                            vo_pos = self.visual_odometry.get_position()
                            vo_heading = self.visual_odometry.get_heading()
                            self.world_state.update_pose(R, t, vo_pos, vo_heading)

                    # Process detection if enabled
                    tracks = []
                    if self.detection_enabled and self.detection_pipeline:
                        left_resized, tracks = self.detection_pipeline.process(
                            left_resized, depth_map
                        )

                    # Process navigation assistance if enabled
                    if self.navigation_enabled and self.navigator:
                        self.navigator.process_frame(
                            left_frame=left_resized,
                            depth_colored=depth_colored if self.depth_enabled else None,
                            tracks=tracks,
                            timestamp=time.monotonic()
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

                        left_panel = np.vstack((stereo_combined, depth_resized))
                    else:
                        left_panel = stereo_combined

                    # Add bird's eye view on the right
                    if self.birdseye_view:
                        # Get camera pose and trajectory if odometry enabled
                        camera_pose = None
                        trajectory = None
                        if self.odometry_enabled and self.world_state:
                            camera_pose = self.world_state.camera_pose
                            trajectory = self.world_state.pose_history

                        # Render bird's eye view
                        birdseye_frame = self.birdseye_view.render(
                            tracks,
                            frame_width=target_w,
                            camera_pose=camera_pose,
                            trajectory=trajectory
                        )

                        # Scale bird's eye view to match left panel HEIGHT
                        left_height = left_panel.shape[0]
                        bev_scale = left_height / birdseye_frame.shape[0]
                        bev_width = int(birdseye_frame.shape[1] * bev_scale)
                        birdseye_resized = cv2.resize(birdseye_frame, (bev_width, left_height))

                        # Combine horizontally
                        combined_frame = np.hstack((left_panel, birdseye_resized))
                    else:
                        combined_frame = left_panel

                    # Add voice panel at bottom (full width)
                    if self.voice_enabled and self.voice_interface:
                        voice_panel = self._create_voice_bottom_panel(combined_frame.shape[1])
                        combined_frame = np.vstack((combined_frame, voice_panel))

                    # Create status
                    elapsed = time.time() - start_time
                    depth_status = " | Depth: ON" if self.depth_enabled else ""
                    detection_status = " | Detection: ON" if self.detection_enabled else ""
                    odometry_status = " | VO: ON" if self.odometry_enabled else ""
                    status = f"Frames: {frame_count} | Elapsed: {elapsed:.1f}s{depth_status}{detection_status}{odometry_status}"

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
                    elif key == ord('r') or key == ord('R'):
                        # Reset odometry
                        if self.odometry_enabled and self.visual_odometry and self.world_state:
                            self.visual_odometry.reset()
                            self.world_state.reset()
                            self.logger.info("Visual odometry reset to origin")
                    elif key == ord('v') or key == ord('V'):
                        # Push-to-talk: listen for voice query
                        if self.voice_enabled and self.voice_interface:
                            self.voice_interface.listen_async()
                            self.logger.info("Listening for voice query...")

                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    self.logger.info(
                        f"Frames: {frame_count}, FPS: {current_fps:.1f}, Avg: {avg_fps:.1f}"
                    )

        except Exception as e:
            import traceback
            self.logger.error(f"Error in viewer loop: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")

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

        # Shutdown voice interface
        if self.voice_interface:
            self.voice_interface.shutdown()

        # Shutdown navigator
        if self.navigator:
            self.navigator.shutdown()

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
        no_detection=args.no_detection,
        no_navigation=args.no_navigation
    )
    return app.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
