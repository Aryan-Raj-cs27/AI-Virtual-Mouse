@echo off
cd /d "%~dp0"

echo Starting AI Virtual Mouse...
echo.

set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONDONTWRITEBYTECODE=1

python -m pip install -r requirements.txt
if errorlevel 1 goto error

python src\virtual_mouse.py
if errorlevel 1 goto error

echo.
echo Virtual Mouse closed successfully.
pause
exit /b 0

:error
echo.
echo ERROR: Something went wrong!
pause
exit /b 1
