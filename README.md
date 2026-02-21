# ReSee - Stereo Camera + Gemini Live API Integration

Real-time multimodal AI system that streams stereo video and audio from an ELP camera to Google's Gemini Live API for intelligent scene analysis.

## Features

- **Stereo Camera Support**: Auto-detection for ELP stereo camera configurations
- **Real-time Streaming**: 10fps video at 1080p + 16kHz audio to Gemini Live API
- **Side-by-side Video**: Combines left/right camera feeds into 3840x1080 frames
- **Live AI Analysis**: Receive real-time insights from Gemini about what it sees and hears
- **Headless Support**: Runs with or without display (automatic detection)
- **Robust Error Handling**: Automatic reconnection and graceful degradation

## Requirements

### Hardware
- **Camera**: ELP Stereo USB camera (2x synchronized global shutter cameras)
- **Microphone**: System default microphone or USB microphone
- **Computer**: macOS, Linux, or Windows with USB ports
- **Internet**: Stable connection for Gemini Live API

### Software
- **Python**: 3.9 or higher
- **Operating System**: macOS 12+, Ubuntu 22.04+, or Windows 10+

## Installation

### 1. Clone the Repository

```bash
cd /Users/nidsc/Desktop/ReSee
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

**macOS**:
```bash
brew install portaudio
```

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev
```

**Windows**:
- PyAudio wheel will be installed automatically via pip

### 5. Configure API Key

1. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Edit `.env` and add your API key:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

## Configuration

Edit `config/config.yaml` to customize settings:

```yaml
camera:
  resolution:
    width: 1920      # Per camera (combined will be 3840x1080)
    height: 1080
  fps: 10            # Target frames per second
  device_mode: auto  # auto, single, dual, retrieve
  jpeg_quality: 85   # JPEG compression quality (1-100)

audio:
  sample_rate: 16000 # Gemini requires 16kHz
  channels: 1        # 1=mono, 2=stereo

gemini:
  model: gemini-2.5-flash-native-audio-preview-12-2025
  max_retries: 5

display:
  preview_enabled: true  # Show video window
  scale: 0.5             # Display scale (0.5 = half size)
  show_fps: true         # Show FPS counter
```

## Usage

### Basic Usage

```bash
python -m src.main
```

### Headless Mode

If running on a server without display:
```bash
export OPENCV_HEADLESS=1
python -m src.main
```

### Controls

- **ESC**: Stop the application (when display is active)
- **Ctrl+C**: Graceful shutdown

## Camera Configuration

The application auto-detects your ELP camera configuration. It supports three modes:

### Mode 1: Single Device (Side-by-Side)
Camera appears as one device outputting 3840x1080 side-by-side frames.

### Mode 2: Dual Devices
Camera appears as two separate devices (indices 0 and 1).

### Mode 3: Retrieve Mode
Single device using OpenCV's `retrieve()` with different flags.

### Manual Configuration

If auto-detection fails, set `device_mode` in `config/config.yaml`:

```yaml
camera:
  device_mode: dual  # or single, or retrieve
  device_indices: [0, 1]  # For dual mode
```

## Troubleshooting

### Camera Not Detected

**Check camera connection**:
```bash
# macOS
system_profiler SPUSBDataType

# Linux
lsusb
v4l2-ctl --list-devices

# Windows
Device Manager > Imaging Devices
```

**List available cameras**:
```python
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

### Audio Issues

**List audio devices**:
```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"{i}: {info['name']}")
```

**Specify device in config**:
```yaml
audio:
  device_index: 2  # Use specific device index
```

### Gemini API Errors

**Invalid API Key**:
- Verify your API key in `.env`
- Check you're using the correct key from [Google AI Studio](https://aistudio.google.com/app/apikey)

**Rate Limiting**:
- Reduce FPS in config: `camera.fps: 5`
- The application will automatically retry with backoff

**Connection Timeout**:
- Check internet connection
- Increase timeout: `gemini.timeout: 60`

### Display Issues

**No display available (headless)**:
- The application automatically falls back to headless mode
- Set `display.preview_enabled: false` in config to suppress warnings

**Window doesn't update**:
- Ensure you have a display server (X11, Wayland)
- Try reducing `display.scale` if performance is low

## Project Structure

```
ReSee/
├── src/
│   ├── main.py                    # Application entry point
│   ├── camera/
│   │   ├── stereo_capture.py      # ELP camera handling
│   │   ├── frame_processor.py     # Frame combining & encoding
│   │   └── display.py             # Video preview window
│   ├── audio/
│   │   └── microphone.py          # Audio capture
│   ├── gemini/
│   │   ├── live_client.py         # WebSocket client
│   │   └── message_handler.py     # Protocol messages
│   ├── config/
│   │   └── settings.py            # Configuration management
│   └── utils/
│       ├── logger.py              # Logging setup
│       └── timing.py              # FPS control
├── config/
│   └── config.yaml                # User configuration
├── requirements.txt               # Python dependencies
└── .env                           # API keys (create from .env.example)
```

## API Reference

### Gemini Live API

This application uses the [Gemini Live API](https://ai.google.dev/gemini-api/docs/live) for real-time multimodal streaming.

**Supported Models**:
- `gemini-2.5-flash-native-audio-preview-12-2025` (default)

**Message Format**:
- **Video**: JPEG frames encoded as base64
- **Audio**: 16-bit PCM at 16kHz, mono, base64-encoded
- **Protocol**: WebSocket bidirectional streaming

## Performance Tips

### Optimize Frame Rate
- Lower FPS reduces bandwidth: `camera.fps: 5`
- Adjust JPEG quality: `camera.jpeg_quality: 75`

### Reduce Display Overhead
- Disable preview: `display.preview_enabled: false`
- Reduce scale: `display.scale: 0.3`

### Audio Optimization
- Use smaller chunks: `audio.chunk_size: 512`

## Known Limitations

1. **Depth Mapping**: Not yet implemented (deferred for future release)
2. **PyAudio Installation**: May require manual portaudio installation on some systems
3. **Gemini Quota**: Subject to Google's API rate limits
4. **USB Bandwidth**: Multiple high-resolution cameras may saturate USB 2.0

## Future Enhancements

- [ ] Stereo depth mapping using OpenCV algorithms
- [ ] Session recording to disk
- [ ] GUI application (PyQt/Tkinter)
- [ ] Multiple camera pairs support
- [ ] GPU acceleration for encoding
- [ ] WebRTC streaming option

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is for educational and research purposes. Please comply with:
- [Google's Gemini API Terms of Service](https://ai.google.dev/gemini-api/terms)
- Applicable camera and audio recording laws in your jurisdiction

## Support

For issues and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [Gemini Live API Documentation](https://ai.google.dev/gemini-api/docs/live)
3. Open an issue on the project repository

## Acknowledgments

- **Google Gemini**: For the powerful Live API
- **ELP**: For quality stereo camera hardware
- **OpenCV**: For video processing capabilities
- **PyAudio**: For audio capture support

---

**Built with Python, OpenCV, and Gemini Live API**
