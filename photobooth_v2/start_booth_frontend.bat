@echo off
echo ==========================================================
echo  Starting AI Zombie Mirror Frontend Kiosk (Port 3000)     
echo ==========================================================
cd /d "%~dp0frontend"
echo Kiosk running at http://127.0.0.1:3000
python serve.py 3000
pause
