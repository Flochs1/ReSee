"""
ReSee - Stereo Camera Viewer

Simple stereo camera viewer at 10fps.
Compatible with macOS, Linux, and Raspberry Pi.
"""

import signal
import sys
import time
from typing import Optional

from src.config.settings import get_settings
from src.utils.logger import setup_logger, get_logger
from src.utils.timing import FPSController, FrameTimer
from src.camera.stereo_capture import StereoCamera, StereoCameraError
from src.camera.display import VideoDisplay


class ReSeeApp:
    """Simple stereo camera viewer application."""

    def __init__(self):
        """Initialize ReSee camera viewer."""
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

        # Timing
        self.fps_controller: Optional[FPSController] = None
        self.frame_timer: Optional[FrameTimer] = None

        # Control flags
        self.running = False

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"\nReceived signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def initialize(self) -> bool:
        """
        Initialize camera and display.

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

            self.logger.info("All components initialized successfully")
            return True

        except StereoCameraError as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def run_viewer(self) -> None:
        """Main camera viewing loop."""
        self.logger.info("Starting stereo camera viewer...")
        self.logger.info(f"Target FPS: {self.settings.camera.fps}")
        self.logger.info(f"Resolution: {self.settings.camera.resolution.width}x{self.settings.camera.resolution.height} per camera")
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
                    # Resize and combine frames for display
                    import numpy as np
                    import cv2

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

                    # Combine side-by-side
                    combined_frame = np.hstack((left_resized, right_resized))

                    # Create status
                    elapsed = time.time() - start_time
                    status = f"Frames: {frame_count} | Elapsed: {elapsed:.1f}s"

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
    app = ReSeeApp()
    return app.run()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
