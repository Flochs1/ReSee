# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application (with depth estimation)
python -m src.main

# Run without depth estimation
python -m src.main --no-depth

# Force recalibration
python -m src.main --recalibrate

# Run headless (no display)
OPENCV_HEADLESS=1 python -m src.main --no-depth

# List available cameras and get recommended config
python list_cameras.py

# Test camera resolutions
python test_resolution.py

# Install dependencies
pip install -r requirements.txt

# System dependencies (for audio capture)
# macOS:
brew install portaudio
# Ubuntu/Debian:
sudo apt-get install python3-pyaudio portaudio19-dev

# Alternative: use the run script (manages venv automatically)
./run.sh
```

There is no test suite currently — `tests/` is empty.

## Controls

- **ESC**: Stop the application (when display is active)
- **Ctrl+C**: Graceful shutdown
- **SPACE**: Capture calibration frame (during calibration mode)

## Architecture

ReSee is a stereo camera viewer that captures from an ELP dual-lens USB camera, displays a side-by-side preview, and optionally computes a depth map. The codebase also contains scaffolding for a Gemini Live API integration (in `src/gemini/`) that is not yet wired into the main application loop.

**Entry point:** `src/main.py:main()` parses CLI args, creates `ReSeeApp`, and calls `run()`.

**CLI arguments:**
- `--recalibrate`: Force stereo calibration even if calibration data exists
- `--no-depth`: Disable depth estimation (stereo view only)

**Data flow:**
1. `StereoCamera` (`src/camera/stereo_capture.py`) opens the camera and runs a background capture thread that continuously reads frames into a bounded `Queue` (size 3, drops oldest on overflow).
2. `ReSeeApp.run_viewer()` (`src/main.py`) polls the queue at the configured FPS, combines left+right frames side-by-side using `np.hstack`, and passes them to `VideoDisplay`.
3. `VideoDisplay` (`src/camera/display.py`) renders the combined frame via OpenCV's `imshow`.

**Camera mode detection** (`StereoCamera.detect_camera_mode`): the ELP camera can appear in three ways — as a single device emitting a 2×-wide side-by-side frame (`single`), as two separate device indices (`dual`), or as one device where left/right are retrieved with different flags (`retrieve`). Auto-detection order: dual → single → retrieve.

**Timing utilities** (`src/utils/timing.py`): `FPSController` uses busy-wait with sleep compensation for frame rate limiting. `FrameTimer` calculates rolling FPS over a 30-frame window.

**Configuration** (`src/config/settings.py`): Pydantic models load from `config/config.yaml` (YAML) and `.env` (API keys via `python-dotenv`). `get_settings()` is a module-level singleton. The `GeminiConfig` and `AudioConfig` sections are optional and currently `None` in the active config.

**Stereo calibration** (`src/calibration/stereo_calibrator.py`): `StereoCalibrator` handles interactive checkerboard-based stereo calibration. On first run (or with `--recalibrate`), displays a 9x6 checkerboard pattern and prompts user to capture 15 frames from different angles. Computes camera matrices, distortion coefficients, rotation/translation between cameras, and rectification maps. Saves to `config/calibration/stereo_calib.npz`.

**Depth estimation** (`src/calibration/depth_estimator.py`): `DepthEstimator` applies rectification maps, computes disparity using StereoSGBM, converts to depth using `depth = (focal * baseline) / disparity`, and applies RYGB colormap (Red=near, Blue=far). Includes legend showing depth scale.

**Gemini integration** (`src/gemini/`): `GeminiLiveClient` manages a WebSocket connection to the Gemini BidiGenerateContent endpoint. `GeminiMessage` constructs the protocol messages (setup, realtime video/audio input, text). This subsystem is fully implemented but not called from `src/main.py`.

**Logging** (`src/utils/logger.py`): Logger setup with optional colorama coloring.

## Configuration

Key settings in `config/config.yaml`:

- `camera.device_mode`: `auto` | `single` | `dual` | `retrieve` — set manually if auto-detection fails
- `camera.device_indices`: device indices (e.g. `[0, 0]` for single mode, `[0, 1]` for dual)
- `camera.resolution.width/height`: per-camera resolution (combined frame is 2× the width)
- `display.preview_enabled`: set `false` for headless operation
- `display.scale`: scales the preview window (default `0.5`)
- `depth.enabled`: enable/disable depth estimation (default `true`)
- `depth.baseline_mm`: stereo camera baseline in millimeters (default `60.0`)
- `depth.calibration_file`: path to calibration data (default `config/calibration/stereo_calib.npz`)
- `depth.num_disparities`: disparity search range, must be divisible by 16 (default `64`)
- `depth.block_size`: block matching size, must be odd (default `9`)
- `depth.min_depth_m` / `depth.max_depth_m`: depth range for colorization (default `0.3` - `5.0` meters)

`.env` file (copy from `.env.example`) is required by `run.sh` but only needed at runtime if using the Gemini integration (`GEMINI_API_KEY`).
