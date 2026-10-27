@echo off
setlocal
cd /d "%~dp0frontend_node"
echo Starting Photobooth Node.js Kiosk Server on http://127.0.0.1:3000...
set PORT=3000
set REMOTE_GPU_URL=http://127.0.0.1:8000
node server.js
pause
