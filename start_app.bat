@echo off
cd /d "%~dp0"
title VisionAI Sentinel - Command Center
color 0B

rem If the dashboard is already running, just open it instead of starting a second server.
netstat -ano | findstr ":8080" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo VisionAI is already running. Opening the dashboard...
    start "" "http://127.0.0.1:8080"
    exit /b 0
)

echo.
echo ================================================================
echo   VISIONAI SENTINEL - CCMS COMMAND CENTER
echo ================================================================
echo.
echo   Starting up... The dashboard opens in a few seconds and the
echo   AI model finishes warming up in the background.
echo.
echo   Dashboard opens automatically in your default browser.
echo   If it does not, open the URL printed below.
echo.
echo   DO NOT CLOSE THIS WINDOW while VisionAI is running.
echo   Press Ctrl+C to stop the application.
echo.
echo ================================================================
echo.

python --version >nul 2>&1
if not errorlevel 1 (
    python -u "%~dp0app.py"
) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
        py -u "%~dp0app.py"
    ) else (
        echo ERROR: Python is not installed or not in PATH.
        echo Please install Python 3.10+ from https://python.org
        echo.
    )
)

echo.
echo ================================================================
echo   VisionAI has stopped. Press any key to exit.
echo ================================================================
pause >nul
