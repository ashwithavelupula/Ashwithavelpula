@echo off
setlocal
cd /d "%~dp0"
title Nexus Offline Assistant

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup not done yet. Running one-click setup...
  call setup_and_run.bat
  exit /b 0
)

echo Starting Nexus... keep this window OPEN.
echo Open your browser at:  http://127.0.0.1:5000
echo Press CTRL+C here to stop.
net start NexusDB >nul 2>&1
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:5000
.venv\Scripts\python app.py
pause >nul
