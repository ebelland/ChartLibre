"""Make the double-clickable launchers that ship in the project folder.

    python3 dev/tools/make_launcher.py        # into the project folder
    python3 dev/tools/make_launcher.py DIR    # somewhere else

``ChartLibre.app`` (macOS) and ``ChartLibre.bat`` (Windows) find the folder
they sit in, so the whole folder can be copied anywhere and started from
there. The first time, each one looks for Python 3.11 or newer, asks, and
installs requirements.txt into ``.venv`` inside the folder; after that it
just starts the application. Nothing is written outside the folder.

These are launchers, not a standalone build: a self-contained executable
would need PyInstaller and would bundle PySide6, SciPy and the rest - well
over a gigabyte per platform.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from app import APP_ICON, APP_NAME, APP_VERSION  # noqa: E402


def _render_icon_pngs(sizes: list[int]) -> dict[int, bytes]:
    """The application icon as PNG bytes at each size, drawn by Qt.

    APP_ICON is bare stroke path data, so it is wrapped in the standard SVG
    document first - handed over on its own it is not an SVG at all, and
    the renderer draws nothing. The mark goes in white on a blue rounded
    square laid out on Apple's app-icon grid (an 824 body on a 1024 canvas),
    so it reads in the Dock and the Finder the way other Mac apps do.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter
    from PySide6.QtSvg import QSvgRenderer

    from app.styles.style import svg_icon_document

    app = QGuiApplication.instance() or QGuiApplication([])
    mark = svg_icon_document(APP_ICON, "#ffffff").replace('stroke-width="1.8"', 'stroke-width="2.2"')
    renderer = QSvgRenderer(QByteArray(mark.encode("utf-8")))
    if not renderer.isValid():
        raise SystemExit("The application icon could not be read as SVG.")
    images: dict[int, bytes] = {}
    for size in sizes:
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        unit = size / 1024
        body = QRectF(100 * unit, 100 * unit, 824 * unit, 824 * unit)
        gradient = QLinearGradient(body.topLeft(), body.bottomLeft())
        gradient.setColorAt(0.0, QColor("#4C8DF6"))
        gradient.setColorAt(1.0, QColor("#1F4FD1"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(body, 185 * unit, 185 * unit)
        inset = 150 * unit
        renderer.render(painter, body.adjusted(inset, inset, -inset, -inset))
        painter.end()
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        # A str, whatever the stubs say: PySide6 refuses b"PNG" at run time.
        image.save(buffer, "PNG")  # pyright: ignore[reportArgumentType, reportCallIssue]
        images[size] = buffer.data().data()
    del app
    return images


#: The macOS launcher. $HERE is the folder the .app sits in.
MACOS_SCRIPT = r"""#!/bin/bash
# ChartLibre launcher (dev/tools/make_launcher.py). It works from wherever
# the ChartLibre folder is: the first launch installs the libraries into
# .venv inside that folder, every later one just starts the application.
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$HERE" || exit 1
VENV="$HERE/.venv"
LOG="$HERE/.venv-setup.log"

ask() {  # ask "message" "button" -> exit status 0 when the button was pressed
  local answer
  answer=$(osascript -e "button returned of (display dialog \"$1\" buttons {\"Cancel\", \"$2\"} default button 2 with title \"ChartLibre\")" 2>/dev/null)
  [ "$answer" = "$2" ]
}

if [ ! -x "$VENV/bin/python3" ]; then
  PY=""
  CANDIDATES=$(ls -d /Library/Frameworks/Python.framework/Versions/3.*/bin/python3 2>/dev/null | sort -t. -k2,2nr)
  for c in $CANDIDATES /opt/homebrew/bin/python3 /usr/local/bin/python3 $(command -v python3); do
    if [ -x "$c" ] && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      PY="$c"
      break
    fi
  done
  if [ -z "$PY" ]; then
    if ask "ChartLibre needs Python 3.11 or newer.\n\nInstall it from python.org, then open ChartLibre again." "Open python.org"; then
      open "https://www.python.org/downloads/"
    fi
    exit 1
  fi
  ask "First launch: ChartLibre will install the libraries it needs into its own folder.\n\nThis needs an internet connection and takes a few minutes. ChartLibre opens by itself when it is ready." "Install" || exit 0
  osascript -e 'display notification "Installing... ChartLibre opens when it is ready." with title "ChartLibre"'
  if ! { "$PY" -m venv "$VENV" \
         && "$VENV/bin/python3" -m pip install --upgrade pip \
         && "$VENV/bin/python3" -m pip install -r "$HERE/requirements.txt"; } >"$LOG" 2>&1; then
    rm -rf "$VENV"
    osascript -e "display dialog \"The installation did not complete. The details are in:\n$LOG\" buttons {\"OK\"} with title \"ChartLibre\" with icon stop" >/dev/null 2>&1
    exit 1
  fi
fi
exec "$VENV/bin/python3" "$HERE/main.py" "$@"
"""

#: The Windows launcher. %~dp0 is the folder the .bat sits in.
WINDOWS_SCRIPT = r"""@echo off
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
"""


def make_macos_app(target_dir: Path) -> Path:
    app_dir = target_dir / f"{APP_NAME}.app"
    if app_dir.exists():
        shutil.rmtree(app_dir)
    macos = app_dir / "Contents" / "MacOS"
    resources = app_dir / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    executable = macos / APP_NAME
    executable.write_text(MACOS_SCRIPT, encoding="utf-8")
    executable.chmod(0o755)

    if shutil.which("iconutil"):
        iconset = target_dir / f"{APP_NAME}.iconset"
        iconset.mkdir(exist_ok=True)
        pngs = _render_icon_pngs([16, 32, 64, 128, 256, 512, 1024])
        for size in (16, 32, 128, 256, 512):
            (iconset / f"icon_{size}x{size}.png").write_bytes(pngs[size])
            (iconset / f"icon_{size}x{size}@2x.png").write_bytes(pngs[size * 2])
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(resources / f"{APP_NAME}.icns")],
            check=True,
        )
        shutil.rmtree(iconset)

    (app_dir / "Contents" / "Info.plist").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>{APP_NAME}</string>
  <key>CFBundleDisplayName</key><string>{APP_NAME}</string>
  <key>CFBundleIdentifier</key><string>org.chartlibre.launcher</string>
  <key>CFBundleVersion</key><string>{APP_VERSION}</string>
  <key>CFBundleShortVersionString</key><string>{APP_VERSION}</string>
  <key>CFBundleExecutable</key><string>{APP_NAME}</string>
  <key>CFBundleIconFile</key><string>{APP_NAME}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    return app_dir


def make_windows_launcher(target_dir: Path) -> Path:
    launcher = target_dir / f"{APP_NAME}.bat"
    # CRLF: cmd.exe reads a .bat line by line and a bare LF confuses its
    # labels and goto.
    launcher.write_bytes(WINDOWS_SCRIPT.replace("\n", "\r\n").encode("utf-8"))
    return launcher


LAUNCHER_SOURCE = Path(__file__).resolve().parent / "launcher"


def make_windows_exe(target_dir: Path) -> Path | None:
    """ChartLibre.exe: the .bat's job without a console window after setup.

    Built with Zig, which cross-compiles for Windows from any system:
    ``ZIG=/path/to/zig`` or a ``zig`` on the PATH. Without one the .exe is
    left as it is - the .bat does the same job.
    """
    zig = os.environ.get("ZIG") or shutil.which("zig")
    if not zig:
        print("zig not found: ChartLibre.exe not rebuilt (set ZIG=/path/to/zig).")
        return None
    import tempfile
    from io import BytesIO

    from PIL import Image

    with tempfile.TemporaryDirectory() as work:
        build = Path(work)
        pngs = _render_icon_pngs([256])
        Image.open(BytesIO(pngs[256])).save(
            build / f"{APP_NAME}.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        )
        for name in ("chartlibre_launcher.c", "chartlibre_launcher.rc"):
            shutil.copy(LAUNCHER_SOURCE / name, build / name)
        exe = target_dir / f"{APP_NAME}.exe"
        subprocess.run(
            [
                zig, "cc", "-target", "x86_64-windows-gnu", "-Os", "-s", "-municode", "-Wl,--subsystem,windows",
                "chartlibre_launcher.c", "chartlibre_launcher.rc", "-o", str(exe),
            ],
            cwd=build,
            check=True,
        )
    for leftover in target_dir.glob(f"{APP_NAME}.pdb"):
        leftover.unlink()
    return exe


def main() -> None:
    target_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else PROJECT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    # Both, whatever this machine is: the folder ships to either platform.
    print(make_macos_app(target_dir))
    print(make_windows_launcher(target_dir))
    exe = make_windows_exe(target_dir)
    if exe is not None:
        print(exe)


if __name__ == "__main__":
    main()
