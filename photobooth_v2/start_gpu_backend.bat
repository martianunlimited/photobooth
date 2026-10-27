@echo off
setlocal
cd /d "%~dp0"
echo Starting AI Zombie Photobooth GPU Backend on port 8000...
if exist "E:\miniconda3\envs\comfyui\python.exe" (
    set "PYTHON_BIN=E:\miniconda3\envs\comfyui\python.exe"
) else (
    set "PYTHON_BIN=python"
)
cd backend
"%PYTHON_BIN%" server.py
pause
