@echo off
cd /d "%~dp0"
echo Starting Gyanovate VisionAI - Security & Logistics Mode...
python --version >nul 2>&1
if not errorlevel 1 (
    python visionai_camera_demo.py --camera 0 --mode loitering
) else (
    py visionai_camera_demo.py --camera 0 --mode loitering
)
pause

