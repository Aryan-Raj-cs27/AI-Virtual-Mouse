# AI Virtual Mouse

AI Virtual Mouse turns webcam hand tracking into direct cursor control with gesture-based clicks, scrolling, and drag interactions.

## Project Status

- Status: Complete
- Readiness: Production-ready desktop demo
- Deployment: Local webcam application; no hosted deployment applies

## Architecture

The app captures camera frames, extracts hand landmarks with MediaPipe, translates landmark geometry into cursor and gesture events, and dispatches those actions to the operating system through PyAutoGUI.

## Tech Stack

- Python 3.10+
- OpenCV
- MediaPipe
- PyAutoGUI
- Windows desktop runtime

## Features

- Real-time hand tracking
- Cursor movement with the index finger
- Left-click, right-click, scroll, and drag gestures
- One-click batch runner for ZIP installs

## Requirements

- Windows 10/11
- Webcam
- Python 3.10 for the pinned dependency set

## Local Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/virtual_mouse.py
```

## One-Click Run

- File Explorer: double-click `run_virtual_mouse.bat`
- VS Code: run the `Run Virtual Mouse (One Click)` task

## Basic Controls

- Move cursor: raise the index finger and move the hand
- Left click: pinch thumb and index finger
- Right click: pinch thumb and middle finger
- Scroll: raise index and middle fingers, then move vertically
- Drag: hold a fist briefly to begin drag, then release to drop

## Troubleshooting

- Python 3.10 missing: install Python 3.10, then rerun the BAT file
- Webcam window missing: check camera permissions for Python in Windows privacy settings
- Cursor jitter: improve lighting and keep the hand centered in frame
- Dependency errors: delete `.venv`, then rerun the setup flow

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

## License

MIT License - see `LICENSE`.
