@echo off
setlocal
cd /d "%~dp0"

echo =======================================
echo   AI Virtual Mouse - One Click Start
echo =======================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher (py) not found.
    echo Install Python 3.10 from https://www.python.org/downloads/
    pause
    exit /b 1
)

py -3.10 --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 is not installed.
    echo Install Python 3.10, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...
    py -3.10 -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [SETUP] Installing/updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo [RUN] Starting Virtual Mouse...
echo Press Q in camera window to close.
echo.
".venv\Scripts\python.exe" src\virtual_mouse.py

endlocal
