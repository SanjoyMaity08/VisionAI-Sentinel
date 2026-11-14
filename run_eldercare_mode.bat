@echo off
cd /d "%~dp0"
echo Starting Gyanovate VisionAI - Domestic & Eldercare Mode...
python --version >nul 2>&1
if not errorlevel 1 (
    python visionai_camera_demo.py --camera 0 --mode fall
) else (
    py visionai_camera_demo.py --camera 0 --mode fall
)
pause

