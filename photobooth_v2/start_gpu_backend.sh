#!/usr/bin/env bash
# Start the FastAPI GPU Photobooth Backend
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting AI Zombie Photobooth GPU Backend on port 8000..."
if [ -f "E:/miniconda3/envs/comfyui/python.exe" ]; then
    PYTHON_BIN="E:/miniconda3/envs/comfyui/python.exe"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

cd "$SCRIPT_DIR/backend"
"$PYTHON_BIN" server.py
