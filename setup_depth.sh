#!/bin/bash
# setup_depth.sh - Setup GPU-accelerated stereo depth estimation
#
# This script:
# 1. Installs ONNX Runtime with CoreML support (Apple Silicon)
# 2. Downloads HITNet ONNX stereo matching model
# 3. Verifies GPU detection and runs a test
#
# Backends:
#   - Primary: HITNet (ONNX Runtime + CoreML GPU)
#   - Fallback: OpenCV StereoSGBM (CPU, always available)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/hitnet"
MODEL_FILE="${MODEL_DIR}/hitnet.onnx"

echo "=========================================="
echo "ReSee Stereo Depth Estimation Setup"
echo "=========================================="
echo ""

# Check platform
if [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM="macOS"
    # Check for Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        ARCH="Apple Silicon (arm64)"
        HAS_COREML=true
    else
        ARCH="Intel (x86_64)"
        HAS_COREML=true  # CoreML works on Intel Macs too
    fi
else
    PLATFORM="$(uname)"
    ARCH="$(uname -m)"
    HAS_COREML=false
fi

echo "Platform: ${PLATFORM}"
echo "Architecture: ${ARCH}"
echo ""

# Step 1: Install ONNX Runtime
echo "Step 1: Installing ONNX Runtime..."
pip install onnxruntime

# Step 2: Download HITNet ONNX model
echo ""
echo "Step 2: Downloading HITNet ONNX model..."

mkdir -p "${MODEL_DIR}"

if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "  Downloading HITNet ONNX model from PINTO model zoo..."

    # PINTO's model zoo resources archive
    ARCHIVE_URL="https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/142_HITNET/resources.tar.gz"
    ARCHIVE_FILE="${MODEL_DIR}/resources.tar.gz"

    if command -v curl &> /dev/null; then
        echo "  Downloading archive (this may take a moment)..."
        curl -L -o "${ARCHIVE_FILE}" "${ARCHIVE_URL}" 2>&1

        if [[ -f "${ARCHIVE_FILE}" ]]; then
            echo "  Extracting archive..."
            cd "${MODEL_DIR}"
            tar -xzf resources.tar.gz 2>&1

            # Find and select the best ONNX model (480x640 resolution, optimized)
            echo "  Looking for ONNX models..."
            SELECTED_MODEL=""

            # Prefer flyingthings_finalpass_xl at 480x640 (best quality)
            for model in flyingthings_finalpass_xl/saved_model_480x640/model_float32_opt.onnx \
                         flyingthings_finalpass_xl/saved_model_480x640/model_float32.onnx \
                         eth3d/saved_model_480x640/model_float32_opt.onnx \
                         middlebury_d400/saved_model_480x640/model_float32_opt.onnx \
                         */saved_model_480x640/model_float32_opt.onnx \
                         */saved_model_*/model_float32_opt.onnx \
                         */saved_model_*/model_float32.onnx; do
                if [[ -f "$model" ]]; then
                    SELECTED_MODEL="$model"
                    echo "  Found: ${model}"
                    break
                fi
            done

            if [[ -n "${SELECTED_MODEL}" && -f "${SELECTED_MODEL}" ]]; then
                cp "${SELECTED_MODEL}" hitnet.onnx
                echo "  Selected model: ${SELECTED_MODEL}"
            else
                echo "  No ONNX model found in archive"
            fi

            # Clean up archive
            rm -f resources.tar.gz
            cd "${SCRIPT_DIR}"
        fi
    elif command -v wget &> /dev/null; then
        wget -O "${ARCHIVE_FILE}" "${ARCHIVE_URL}" 2>&1 && {
            cd "${MODEL_DIR}"
            tar -xzf resources.tar.gz 2>&1
            rm -f resources.tar.gz
            cd "${SCRIPT_DIR}"
        }
    else
        echo "  Neither curl nor wget available. Please download manually."
    fi

    if [[ -f "${MODEL_FILE}" ]]; then
        FILE_SIZE=$(ls -lh "${MODEL_FILE}" | awk '{print $5}')
        echo "  Model ready: ${MODEL_FILE} (${FILE_SIZE})"
    else
        echo ""
        echo "  WARNING: Model download or extraction failed."
        echo "  Please download manually from:"
        echo "    https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET"
        echo ""
        echo "  Save an ONNX model to: ${MODEL_FILE}"
    fi
else
    FILE_SIZE=$(ls -lh "${MODEL_FILE}" | awk '{print $5}')
    echo "  Model already exists: ${MODEL_FILE} (${FILE_SIZE})"
fi

# Step 3: Verify installation
echo ""
echo "Step 3: Verifying installation..."

python3 << 'EOF'
import sys

print("Checking ONNX Runtime installation...")

# Check ONNX Runtime
try:
    import onnxruntime as ort
    print(f"  ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"  Available providers: {providers}")
except ImportError as e:
    print(f"  ERROR: ONNX Runtime not installed: {e}")
    sys.exit(1)

# Check for CoreML (Apple Silicon GPU)
has_coreml = "CoreMLExecutionProvider" in providers
if has_coreml:
    print("  CoreML (Apple GPU): Available")
else:
    print("  CoreML: Not available (will use CPU)")

# Check HITNet model
print("")
print("Checking HITNet model...")
from pathlib import Path

model_path = Path("models/hitnet/hitnet.onnx")

if model_path.exists():
    file_size = model_path.stat().st_size / (1024 * 1024)
    print(f"  Model found: {model_path} ({file_size:.1f} MB)")

    # Try to load it
    try:
        sess_options = ort.SessionOptions()
        if has_coreml:
            providers_list = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers_list = ["CPUExecutionProvider"]

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers_list,
        )
        print("  Model loaded successfully!")

        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        print(f"  Input: {input_info.name} {input_info.shape}")
        print(f"  Output: {output_info.name}")

        # Quick inference test
        import numpy as np
        print("")
        print("Running inference test...")

        # Create dummy input based on model shape
        shape = input_info.shape
        if shape[1] == 6:  # (B, 6, H, W) - concatenated stereo
            h, w = shape[2], shape[3]
            dummy_input = np.random.rand(1, 6, h, w).astype(np.float32)
        else:
            dummy_input = np.random.rand(*shape).astype(np.float32)

        import time
        # Warm-up
        _ = session.run(None, {input_info.name: dummy_input})

        # Timed run
        start = time.time()
        output = session.run(None, {input_info.name: dummy_input})
        elapsed = (time.time() - start) * 1000

        print(f"  Output shape: {output[0].shape}")
        print(f"  Inference time: {elapsed:.1f}ms")

    except Exception as e:
        print(f"  WARNING: Could not load model: {e}")
else:
    print(f"  Model NOT found at: {model_path}")
    print("  Please download manually from PINTO's model zoo.")

# Check OpenCV (fallback)
print("")
print("Checking OpenCV (SGBM fallback)...")
try:
    import cv2
    print(f"  OpenCV version: {cv2.__version__}")

    # Check for ximgproc (optional, for WLS filter)
    try:
        _ = cv2.ximgproc.createDisparityWLSFilter
        print("  opencv-contrib (WLS filter): Available")
    except AttributeError:
        print("  opencv-contrib (WLS filter): Not available (optional)")
except ImportError as e:
    print(f"  ERROR: OpenCV not installed: {e}")
    sys.exit(1)

print("")
print("==========================================")
if model_path.exists() and has_coreml:
    print("Setup complete! HITNet ready with CoreML GPU.")
    print("Backend: hitnet (ONNX + CoreML)")
elif model_path.exists():
    print("Setup complete! HITNet ready (CPU mode).")
    print("Backend: hitnet (ONNX CPU)")
else:
    print("Setup complete! Using SGBM fallback.")
    print("Backend: sgbm (OpenCV)")
    print("")
    print("To enable HITNet, download the model:")
    print("  https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET")
print("==========================================")
print("")
print("To enable depth in ReSee, set in config/config.yaml:")
print("  depth:")
print("    enabled: true")
print("    backend: auto")
EOF

echo ""
echo "Setup finished!"
