#!/usr/bin/env bash
set -e
echo "=========================================================="
echo " Starting AI Zombie Mirror GPU Inference Server (FastAPI) "
echo "=========================================================="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/backend"
python realtime_server.py
