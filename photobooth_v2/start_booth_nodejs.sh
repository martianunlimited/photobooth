#!/usr/bin/env bash
# Start the Node.js Express Kiosk Proxy and Frontend Server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend_node"

echo "Starting Photobooth Node.js Kiosk Server on http://127.0.0.1:3000..."
export PORT=3000
export REMOTE_GPU_URL="http://127.0.0.1:8000"
node server.js
