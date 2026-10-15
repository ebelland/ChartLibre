@echo off
rem Windows: upgrade ChartLibre's libraries (double-click, or run in a terminal).
rem Any options go to update_dependencies.py: --dry-run, --write-pins, --test, --python PATH.
cd /d "%~dp0\..\.."
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
if exist .python\python.exe set PY=.python\python.exe
"%PY%" dev\tools\update_dependencies.py %*
pause
