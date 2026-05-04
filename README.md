# AI Virtual Mouse

Control your mouse cursor using hand gestures through a webcam.

This project uses:
- Python
- OpenCV
- MediaPipe
- PyAutoGUI

## Features
- Real-time hand tracking
- Cursor movement with index finger
- Left click gesture
- Right click gesture
- Scroll mode
- Drag mode

## Requirements
- Windows 10/11
- Webcam
- Python 3.10 (required for the pinned MediaPipe version)

## Quick Start (ZIP Download Users)
If you downloaded this project as a ZIP from GitHub:

1. Extract the ZIP.
2. Open the extracted folder.
3. Double-click `run_virtual_mouse.bat`.
4. Wait for first-time setup to complete.
5. Use gestures in front of your webcam.
6. Press `Q` in the camera window to exit.

That is all. The BAT file automatically creates `.venv`, installs dependencies, and starts the app.

## Manual Setup (PowerShell)
From the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/virtual_mouse.py
```

## One-Click Run Options
- File Explorer: double-click `run_virtual_mouse.bat`
- VS Code: Run Task -> `Run Virtual Mouse (One Click)`

## Basic Controls
- Move cursor: raise index finger and move hand
- Left click: pinch thumb + index finger
- Right click: pinch thumb + middle finger
- Scroll: index + middle finger up, move vertically
- Drag: hold fist briefly to begin drag, release to drop

## Troubleshooting
- `Python 3.10 is not installed`
	- Install Python 3.10, then rerun `run_virtual_mouse.bat`.
- Webcam window does not appear
	- Check camera permission for Python in Windows privacy settings.
- Cursor feels jittery
	- Improve lighting and keep your hand within camera frame.
- MediaPipe or dependency errors
	- Delete `.venv`, then run `run_virtual_mouse.bat` again.

## Project Structure
```text
VirtualMouseAI/
	docs/
	src/
		virtual_mouse.py
	requirements.txt
	run_virtual_mouse.bat
	README.md
```

## Notes for Contributors
- Do not commit `.venv`.
- Keep dependencies in `requirements.txt`.
- Test using:

```powershell
python src/virtual_mouse.py
```
