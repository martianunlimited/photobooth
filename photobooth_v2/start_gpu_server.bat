@echo off
echo ==========================================================
echo  Starting AI Zombie Mirror GPU Inference Server (FastAPI) 
echo ==========================================================
cd /d "%~dp0backend"
if exist "E:\miniconda3\envs\comfyui\python.exe" (
    "E:\miniconda3\envs\comfyui\python.exe" realtime_server.py
) else (
    python realtime_server.py
)
pause
