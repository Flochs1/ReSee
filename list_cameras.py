#!/usr/bin/env python3
"""
Camera Detection Utility

Lists all available camera devices and their properties.
Use this to find the correct device indices for your ELP stereo camera.
"""

import cv2
import sys


def list_cameras(max_test=10):
    """
    List all available camera devices.

    Args:
        max_test: Maximum number of device indices to test
    """
    print("=" * 60)
    print("Camera Detection Utility")
    print("=" * 60)
    print()

    available_cameras = []

    for index in range(max_test):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            backend = cap.getBackendName()

            # Try to read a frame
            ret, frame = cap.read()

            if ret:
                actual_height, actual_width = frame.shape[:2]

                print(f"Camera {index}:")
                print(f"  Backend: {backend}")
                print(f"  Resolution: {width}x{height} (reported)")
                print(f"  Actual frame: {actual_width}x{actual_height}")
                print(f"  FPS: {fps}")

                # Check if this might be a stereo camera
                if actual_width > 3000:
                    print(f"  *** LIKELY STEREO CAMERA (side-by-side) ***")
                elif actual_width > 2500:
                    print(f"  *** POSSIBLE STEREO CAMERA ***")

                print()

                available_cameras.append({
                    'index': index,
                    'width': actual_width,
                    'height': actual_height,
                    'fps': fps,
                    'backend': backend
                })

            cap.release()

    if not available_cameras:
        print("No cameras detected!")
        print()
        print("Troubleshooting:")
        print("1. Make sure your ELP camera is connected via USB")
        print("2. Check if the camera is recognized by your system:")
        print("   macOS: system_profiler SPUSBDataType | grep -i camera")
        print("   Linux: lsusb | grep -i camera")
        print("3. Try restarting the camera or computer")
        return

    print("=" * 60)
    print(f"Found {len(available_cameras)} camera(s)")
    print("=" * 60)
    print()

    # Recommend configuration
    stereo_candidates = [
        cam for cam in available_cameras
        if cam['width'] >= 3000  # Side-by-side stereo is typically 3840 or similar
    ]

    if stereo_candidates:
        cam = stereo_candidates[0]
        print("RECOMMENDED CONFIGURATION for config.yaml:")
        print()
        print("camera:")
        print(f"  device_mode: single  # Side-by-side stereo")
        print(f"  device_indices: [{cam['index']}, {cam['index']}]")
        print()
    else:
        # Look for two cameras with similar resolution
        if len(available_cameras) >= 2:
            cam1 = available_cameras[0]
            cam2 = available_cameras[1]

            print("RECOMMENDED CONFIGURATION for config.yaml:")
            print()
            print("camera:")
            print(f"  device_mode: dual  # Two separate cameras")
            print(f"  device_indices: [{cam1['index']}, {cam2['index']}]")
            print()
        else:
            print("Could not identify stereo camera configuration.")
            print("You may need to manually configure device_indices in config.yaml")
            print()


if __name__ == "__main__":
    try:
        list_cameras()
    except KeyboardInterrupt:
        print("\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
