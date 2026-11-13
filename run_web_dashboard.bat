@echo off
cd /d "%~dp0"
echo Starting Gyanovate VisionAI Command Center...
python --version >nul 2>&1
if not errorlevel 1 (
    python app.py
) else (
    py app.py
)
pause
