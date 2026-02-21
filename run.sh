#!/bin/bash

# ReSee - Run script
# This script activates the virtual environment and runs the application

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}ReSee - Stereo Camera + Gemini Live API${NC}"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import cv2" 2>/dev/null; then
    echo -e "${YELLOW}Dependencies not installed. Installing...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed${NC}"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}ERROR: .env file not found${NC}"
    echo "Please create .env file from .env.example and add your Gemini API key"
    echo "  cp .env.example .env"
    echo "  # Then edit .env and add your API key"
    exit 1
fi

# Run the application
echo -e "${GREEN}Starting ReSee...${NC}"
echo ""
python -m src.main

# Deactivate virtual environment on exit
deactivate
