@echo off
rem ChartLibre launcher (dev/tools/make_launcher.py). It works from wherever
rem the ChartLibre folder is. The first launch downloads Python into .python
rem and the libraries into .venv, both inside that folder; every later launch
rem just starts the application. Nothing needs installing first.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" goto run

echo First launch: ChartLibre downloads what it needs into this folder -
echo Python and its scientific libraries: about 400 MB to download, 1.7 GB
echo on disk. This needs an internet connection and takes a few minutes.
echo.
if not exist ".python\python.exe" call :getpython || goto failed
".python\python.exe" install.py || goto failed

:run
start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py" %*
exit /b 0

:getpython
set "URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.13.16+20261003-x86_64-pc-windows-msvc-install_only.tar.gz"
set "SUM=5e100ee3d592ff500f4408a624f054d202e32d9dba8a12b2226bef81083fd778"
set "ARCHIVE=.python-download.tar.gz"
echo Downloading Python 3.13.16 ...
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
