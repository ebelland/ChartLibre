@echo off
rem ChartLibre launcher (dev/tools/make_launcher.py). It works from wherever
rem the ChartLibre folder is. The first launch downloads Python and the
rem libraries into .python inside that folder (a ready-made package already
rem has them); every later launch just starts the application.
setlocal
cd /d "%~dp0"
if exist ".python\CHARTLIBRE_READY" goto runtime
if exist ".venv\Scripts\pythonw.exe" goto venv

echo First launch: ChartLibre downloads what it needs into this folder -
echo Python and its scientific libraries: about 400 MB to download, 1.7 GB
echo on disk. This needs an internet connection and takes a few minutes.
echo.
if not exist ".python\python.exe" call :getpython || goto failed
".python\python.exe" install.py --runtime || goto failed

:runtime
start "" ".python\pythonw.exe" "%~dp0main.py" %*
exit /b 0

:venv
rem An install made before .python.
start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py" %*
exit /b 0

:getpython
set "URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.15.0+20261013-x86_64-pc-windows-msvc-install_only.tar.gz"
set "SUM=0d9950ba8c6399203e60ed6a2452fc4e8def35c83fb6da00ec815784b7887483"
set "ARCHIVE=.python-download.tar.gz"
echo Downloading Python 3.15.0 ...
curl.exe -fL --retry 3 -o "%ARCHIVE%" "%URL%" || exit /b 1
set "GOT="
for /f "skip=1 delims=" %%h in ('certutil -hashfile "%ARCHIVE%" SHA256') do if not defined GOT set "GOT=%%h"
set "GOT=%GOT: =%"
if /i not "%GOT%"=="%SUM%" (
  echo The download is damaged.
  del "%ARCHIVE%"
  exit /b 1
)
if exist .python rmdir /s /q .python
mkdir .python
tar -xzf "%ARCHIVE%" -C .python --strip-components 1 || exit /b 1
del "%ARCHIVE%"
exit /b 0

:failed
echo.
echo The installation did not complete; the messages above say why.
echo Check the internet connection and open ChartLibre again.
pause
exit /b 1
