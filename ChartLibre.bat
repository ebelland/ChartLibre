@echo off
rem ChartLibre launcher (dev/tools/make_launcher.py). It works from wherever
rem the ChartLibre folder is: the first launch installs the libraries into
rem .venv inside that folder, every later one just starts the application.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" goto run

set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo ChartLibre needs Python 3.11 or newer.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
  echo then open ChartLibre again.
  start "" "https://www.python.org/downloads/"
  pause
  exit /b 1
)

echo First launch: installing the libraries ChartLibre needs into this folder.
echo This needs an internet connection and takes a few minutes.
echo.
%PY% -m venv .venv || goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto failed

:run
start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py" %*
exit /b 0

:failed
echo.
echo The installation did not complete; the messages above say why.
if exist .venv rmdir /s /q .venv
pause
exit /b 1
