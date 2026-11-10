@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo MediFlow Python environment is missing. See README.md for setup.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B -m mediflow_datasets.capture_pilot
if errorlevel 1 pause
