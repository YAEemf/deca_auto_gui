@echo off
rem Deca-Auto installer for Windows
rem Creates a virtual environment in .venv and installs deca_auto into it.
setlocal
cd /d "%~dp0"

set "PY="
python -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>&1 && set "PY=python"
if not defined PY (
    py -3 -c "import sys; sys.exit(sys.version_info < (3, 12))" >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    echo [ERROR] Python 3.12 or later was not found.
    echo         Install it from https://www.python.org/downloads/ and run this script again.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment in .venv ...
    %PY% -m venv .venv || goto :fail
) else (
    echo [1/3] Using existing virtual environment in .venv
)

echo [2/3] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail

echo [3/3] Installing deca_auto and its dependencies. This may take several minutes ...
".venv\Scripts\python.exe" -m pip install -e . || goto :fail

echo.
echo Installation completed.
echo To start Deca-Auto, run run.bat
echo.
pause
exit /b 0

:fail
echo.
echo [ERROR] Installation failed. See the messages above.
pause
exit /b 1
