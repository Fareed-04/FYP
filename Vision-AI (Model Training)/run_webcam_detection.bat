@echo off
REM Run YOLO detection with webcam
echo Running YOLO detection with webcam...
echo.
echo Controls:
echo   - Press 'q' to quit
echo   - Press 's' to pause
echo   - Press 'p' to save a screenshot
echo.

cd /d "%~dp0"
cd ..
call .venv\Scripts\activate.bat
python "Vision-AI (Model Training)\yolo_detect.py" --model "Vision-AI (Model Training)\my_model.pt" --source usb0

pause

