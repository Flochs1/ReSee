#!/bin/bash

# ReSee Startup Script
# Sets up virtual environment and runs the application
# Compatible with macOS and Linux (Ubuntu/Debian/Fedora)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}ReSee - Stereo Camera + Gemini Live API${NC}"
echo "========================================"
echo ""

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case "$ID" in
                ubuntu|debian)
                    echo "debian"
                    ;;
                fedora|rhel|centos)
                    echo "fedora"
                    ;;
                *)
                    echo "linux"
                    ;;
            esac
        else
            echo "linux"
        fi
    else
        echo "unknown"
    fi
}

# Check if PortAudio is installed
check_portaudio() {
    local os_type=$1

    case "$os_type" in
        macos)
            brew list portaudio &>/dev/null
            return $?
            ;;
        debian)
            dpkg -l | grep -q portaudio19-dev
            return $?
            ;;
        fedora)
            rpm -qa | grep -q portaudio-devel
            return $?
            ;;
        *)
            # Generic check - look for portaudio in pkg-config
            pkg-config --exists portaudio-2.0 &>/dev/null
            return $?
            ;;
    esac
}

# Install PortAudio based on OS
install_portaudio() {
    local os_type=$1

    echo -e "${YELLOW}PortAudio not found. Installing...${NC}"

    case "$os_type" in
        macos)
            if ! command -v brew &> /dev/null; then
                echo -e "${RED}ERROR: Homebrew is not installed.${NC}"
                echo "Please install Homebrew from: https://brew.sh"
                echo ""
                echo "Run this command:"
                echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                echo ""
                exit 1
            fi

            echo "Installing PortAudio via Homebrew..."
            brew install portaudio
            ;;

        debian)
            echo "Installing PortAudio via apt..."
            echo -e "${YELLOW}This may require sudo password.${NC}"

            sudo apt-get update -qq
            sudo apt-get install -y portaudio19-dev
            ;;

        fedora)
            echo "Installing PortAudio via dnf..."
            echo -e "${YELLOW}This may require sudo password.${NC}"

            sudo dnf install -y portaudio-devel
            ;;

        *)
            echo -e "${RED}ERROR: Unknown Linux distribution.${NC}"
            echo "Please install PortAudio manually:"
            echo "  Ubuntu/Debian: sudo apt-get install portaudio19-dev"
            echo "  Fedora/RHEL:   sudo dnf install portaudio-devel"
            echo ""
            exit 1
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}PortAudio installed successfully!${NC}"
    else
        echo -e "${RED}ERROR: Failed to install PortAudio.${NC}"
        exit 1
    fi
}

# Main setup process
echo -e "${BLUE}Step 1: Checking system dependencies...${NC}"

OS_TYPE=$(detect_os)
echo "Detected OS: $OS_TYPE"

# Check and install PortAudio if needed
if check_portaudio "$OS_TYPE"; then
    echo -e "${GREEN}PortAudio is already installed.${NC}"
else
    install_portaudio "$OS_TYPE"
fi

echo ""
echo -e "${BLUE}Step 2: Setting up Python environment...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo -e "${BLUE}Step 3: Installing Python dependencies...${NC}"

# Install/update dependencies
pip install -r requirements.txt --upgrade

# Clean up unused packages
echo "Removing stale packages..."
pip autoremove -y 2>/dev/null || true

echo -e "${GREEN}Dependencies installed successfully!${NC}"

# Check if ReID model needs to be exported
echo ""
echo -e "${BLUE}Step 4: Checking ML models...${NC}"

REID_MODEL_DINO="models/reid_dinov2.mlpackage"
REID_MODEL_MOBILE="models/reid_mobilenet.mlpackage"

if [ ! -d "$REID_MODEL_DINO" ] && [ ! -d "$REID_MODEL_MOBILE" ]; then
    echo -e "${YELLOW}ReID model not found. Exporting DINOv2-Small...${NC}"
    echo "This is a one-time setup that requires PyTorch (~2-3 minutes)."
    echo ""

    # Install PyTorch temporarily for export
    echo "Installing PyTorch (temporary, for export only)..."
    pip install torch torchvision --quiet

    # Run the export script
    echo "Exporting DINOv2 ReID model to CoreML..."
    python scripts/export_reid_dinov2.py

    if [ -d "$REID_MODEL_DINO" ]; then
        echo -e "${GREEN}DINOv2 ReID model exported successfully!${NC}"

        # Uninstall PyTorch to keep runtime lean
        echo "Removing PyTorch (no longer needed at runtime)..."
        pip uninstall torch torchvision -y --quiet 2>/dev/null || true
    else
        echo -e "${YELLOW}WARNING: ReID model export failed. Will use handcrafted features.${NC}"
    fi
else
    if [ -d "$REID_MODEL_DINO" ]; then
        echo -e "${GREEN}DINOv2 ReID model already exists.${NC}"
    else
        echo -e "${GREEN}MobileNet ReID model already exists.${NC}"
    fi
fi

# Check if .env exists
echo ""
echo -e "${BLUE}Step 5: Checking configuration...${NC}"

if [ ! -f ".env" ]; then
    echo ""
    echo -e "${YELLOW}WARNING: .env file not found!${NC}"
    echo "Please create it: cp .env.example .env"
    echo "Then add your Gemini API key from: https://aistudio.google.com/app/apikey"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to exit..."
else
    echo -e "${GREEN}Configuration file (.env) found.${NC}"
fi

# Run the application
echo ""
echo -e "${GREEN}Step 6: Starting ReSee...${NC}"
echo "========================================"
echo ""
echo "CLI options: --recalibrate (force calibration), --no-depth (disable depth)"
echo ""
python -m src.main "$@"

# Deactivate virtual environment on exit
deactivate
