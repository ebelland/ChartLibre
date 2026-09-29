@echo off
rem Double-click to install ChartLibre's libraries into .venv inside this folder
rem (install.py does the work).
setlocal
cd /d "%~dp0"

set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo ChartLibre needs Python 3.11 or newer.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
  echo then open this file again.
  start "" "https://www.python.org/downloads/"
  pause
  exit /b 1
)

%PY% install.py %*
set "STATUS=%ERRORLEVEL%"
echo.
pause
exit /b %STATUS%
