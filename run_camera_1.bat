@echo off
cd /d "%~dp0"
echo ========================================================
echo   GYANOVATE VISIONAI - 3-MINUTE PERIODIC THREAT DISPATCH
echo ========================================================
echo Starting Camera 1 (External USB Camera)...
echo  * Zero continuous Telegram spam
echo  * Dispatches strictly on a 3-minute cycle
echo  * Sends 1 validated 5-second video clip per 3 minutes
echo  * Only monitors core safety violations (no false events)
echo  * Press 'D' in camera window to send clip immediately
echo ========================================================
python --version >nul 2>&1
if not errorlevel 1 (
    python visionai_camera_demo.py --camera 1
) else (
    py visionai_camera_demo.py --camera 1
)
pause
