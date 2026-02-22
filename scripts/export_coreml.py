#!/usr/bin/env python3
"""
Export YOLOv8n to CoreML format.

Run once:
    python scripts/export_coreml.py

Creates models/yolov8n.mlpackage for PyTorch-free inference.
"""

from pathlib import Path


def main():
    print("=" * 50)
    print("Exporting YOLOv8n to CoreML")
    print("=" * 50)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("\nLoading ultralytics (requires PyTorch)...")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    print("Exporting to CoreML...")
    model.export(format="coreml", imgsz=1280)

    # Move to models directory
    import shutil
    src = Path("yolov8n.mlpackage")
    dst = models_dir / "yolov8n.mlpackage"

    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"\n✓ Exported to: {dst}")
        print("\nNow you can run detection without PyTorch!")
    else:
        print("\n✗ Export failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
