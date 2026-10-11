@echo off
rem Deca-Auto launcher for Windows
rem Starts deca_auto from the virtual environment created by install.bat.
rem Any arguments are passed through, e.g. run.bat --no-gui --config config.toml
setlocal

if not exist "%~dp0.venv\Scripts\deca-auto.exe" (
    echo [ERROR] Deca-Auto is not installed. Run install.bat first.
    pause
    exit /b 1
)

"%~dp0.venv\Scripts\deca-auto.exe" %*
if errorlevel 1 pause
