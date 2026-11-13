@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo GYANOVATE VISIONAI - DEPENDENCY INSTALLER
echo ================================================
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 goto python_error
    set "PYTHON_CMD=py"
) else (
    set "PYTHON_CMD=python"
)

echo Using Python runtime:
%PYTHON_CMD% --version

echo.
echo [1/2] Checking pip...
%PYTHON_CMD% -m pip install --upgrade pip --quiet

echo.
echo [2/2] Installing VisionAI requirements...
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 goto install_error

echo.
echo ================================================
echo Installation complete! All dependencies ready.
echo Next: double-click start_app.bat to launch CCMS.
echo ================================================
pause
exit /b 0

:python_error
echo Python was not found. Install a supported 64-bit Python version and add it to PATH.
pause
exit /b 1

:install_error
echo Package installation failed. Read the error above and share a screenshot.
pause
exit /b 1
