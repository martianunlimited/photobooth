#!/usr/bin/env bash
set -e
echo "=========================================================="
echo " Starting AI Zombie Mirror Frontend Kiosk (Port 3000)     "
echo "=========================================================="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/frontend"
echo "Kiosk running at: http://127.0.0.1:3000"
python serve.py 3000
