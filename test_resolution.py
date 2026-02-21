#!/usr/bin/env python3
"""
Test different resolutions for ELP stereo camera.
"""

import cv2
import sys


def test_resolutions(device_index=0):
    """Test various resolutions on the camera."""

    # Common resolutions to test (width x height)
    # For stereo side-by-side, width should be 2x the per-camera width
    test_resolutions = [
        (2560, 720),   # Current: 1280x720 per camera
        (3200, 1200),  # Target: 1600x1200 per camera
        (3840, 1080),  # 1920x1080 per camera
        (2560, 960),   # 1280x960 per camera
        (3200, 900),   # 1600x900 per camera
        (2560, 1440),  # 1280x1440 per camera
    ]

    print("=" * 60)
    print(f"Testing resolutions for Camera {device_index}")
    print("=" * 60)
    print()

    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {device_index}")
        return

    print("Testing various resolutions...")
    print()

    successful = []

    for width, height in test_resolutions:
        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try to read a frame
        ret, frame = cap.read()

        if ret:
            frame_height, frame_width = frame.shape[:2]

            success = (frame_width == width and frame_height == height)

            status = "✓ SUCCESS" if success else "✗ FAILED"

            print(f"Requested: {width}x{height}")
            print(f"  Reported: {actual_width}x{actual_height}")
            print(f"  Actual:   {frame_width}x{frame_height}")
            print(f"  Status:   {status}")

            if success:
                per_camera_width = frame_width // 2
                per_camera_height = frame_height
                print(f"  Per camera: {per_camera_width}x{per_camera_height}")
                successful.append((width, height, per_camera_width, per_camera_height))

            print()

    cap.release()

    if successful:
        print("=" * 60)
        print("SUPPORTED RESOLUTIONS:")
        print("=" * 60)
        for combined_w, combined_h, per_w, per_h in successful:
            print(f"  Combined: {combined_w}x{combined_h} = {per_w}x{per_h} per camera")
        print()

        # Recommend highest resolution
        best = successful[-1]
        print("RECOMMENDED for config.yaml:")
        print()
        print("camera:")
        print("  resolution:")
        print(f"    width: {best[2]}")
        print(f"    height: {best[3]}")
        print(f"  fps: 10")
        print(f"  device_mode: single")
        print(f"  device_indices: [{device_index}, {device_index}]")
        print()
    else:
        print("No supported resolutions found!")


if __name__ == "__main__":
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_resolutions(device)
