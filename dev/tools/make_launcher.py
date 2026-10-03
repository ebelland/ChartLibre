"""Make the double-clickable launchers that ship in the project folder.

    python3 dev/tools/make_launcher.py        # into the project folder
    python3 dev/tools/make_launcher.py DIR    # somewhere else

``ChartLibre.app`` (macOS), ``ChartLibre.bat``/``ChartLibre.exe`` (Windows)
and ``ChartLibre.sh`` (Linux) find the folder they sit in, so the whole
folder can be copied anywhere and started from there. The first time, each
one downloads a standalone Python into ``.python`` (see PYTHON_ARCHIVES),
checks its SHA-256, and runs install.py with it, which installs
requirements.txt into ``.venv``; after that it just starts the application.
Nobody has to install Python first, and nothing is written outside the
folder.

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
        images[size] = bytes(buffer.data().data())
    del app
    return images


#: The Python every launcher downloads on the first launch, into ``.python``
#: in the ChartLibre folder: a standalone CPython build from
#: python-build-standalone (the builds uv installs), so nobody has to install
#: Python first. Pinned to one release, and each archive is checked against
#: its SHA-256 before anything in it runs. To move to a newer one, take the
#: names and digests from the release page and run this file again.
PYTHON_RELEASE: str = "20261003"
PYTHON_VERSION: str = "3.13.16"
PYTHON_BASE_URL: str = (
    f"https://github.com/astral-sh/python-build-standalone/releases/download/{PYTHON_RELEASE}"
)
#: Platform key -> (archive name, sha256).
PYTHON_ARCHIVES: dict[str, tuple[str, str]] = {
    "macos-arm64": (
        f"cpython-{PYTHON_VERSION}+{PYTHON_RELEASE}-aarch64-apple-darwin-install_only.tar.gz",
        "d8975d7df4f08f7b1c7aafcdfacbddcec3d366415f2c1a72b2466b6850815933",
    ),
    "macos-x86_64": (
        f"cpython-{PYTHON_VERSION}+{PYTHON_RELEASE}-x86_64-apple-darwin-install_only.tar.gz",
        "8e9cb087305bfb8969f68a905f79f41469d4aa5220c1aa71ada7fc9953bdba0f",
    ),
    # Windows on ARM runs this one too, emulated: Qt's own arm64 wheels are
    # not what PySide6 publishes for every library ChartLibre needs.
    "windows-x86_64": (
        f"cpython-{PYTHON_VERSION}+{PYTHON_RELEASE}-x86_64-pc-windows-msvc-install_only.tar.gz",
        "5e100ee3d592ff500f4408a624f054d202e32d9dba8a12b2226bef81083fd778",
    ),
    "linux-x86_64": (
        f"cpython-{PYTHON_VERSION}+{PYTHON_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz",
        "0a0272910b10417c659a9312fb3f2d7a6d774da7bd510999be7a3ba83273dc1f",
    ),
    "linux-aarch64": (
        f"cpython-{PYTHON_VERSION}+{PYTHON_RELEASE}-aarch64-unknown-linux-gnu-install_only.tar.gz",
        "6477121f22904963a066ac8ff89983baae781510815590e894daa3c11ce86652",
    ),
}


def _archive(key: str) -> tuple[str, str]:
    """(download URL, sha256) of the Python archive for *key*."""
    name, digest = PYTHON_ARCHIVES[key]
    return f"{PYTHON_BASE_URL}/{name}", digest


def _fill(template: str) -> str:
    """Put the pinned archives' URLs and digests into a launcher template."""
    for key in PYTHON_ARCHIVES:
        url, digest = _archive(key)
        marker = key.upper().replace("-", "_")
        template = template.replace(f"@{marker}_URL@", url).replace(f"@{marker}_SHA@", digest)
    return template.replace("@PYTHON_VERSION@", PYTHON_VERSION)


#: The macOS launcher. $HERE is the folder the .app sits in.
MACOS_SCRIPT = _fill(r"""#!/bin/bash
# ChartLibre launcher (dev/tools/make_launcher.py). It works from wherever
# the ChartLibre folder is. The first launch downloads Python and the
# libraries into .python inside that folder (a ready-made package already
# has them); every later launch just starts the application.
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$HERE" || exit 1
VENV="$HERE/.venv"
RUNTIME="$HERE/.python"
LOG="$HERE/.venv-setup.log"

ask() {  # ask "message" "button" -> exit status 0 when the button was pressed
  local answer
  answer=$(osascript -e "button returned of (display dialog \"$1\" buttons {\"Cancel\", \"$2\"} default button 2 with title \"ChartLibre\")" 2>/dev/null)
  [ "$answer" = "$2" ]
}

say() {  # a notification, so the first launch is not silent; never fails
  osascript -e "display notification \"$1\" with title \"ChartLibre\"" >/dev/null 2>&1
  return 0
}

get_python() {  # download Python @PYTHON_VERSION@ into .python, once
  [ -x "$RUNTIME/bin/python3" ] && return 0
  case "$(uname -m)" in
    arm64) url="@MACOS_ARM64_URL@"; sum="@MACOS_ARM64_SHA@" ;;
    *)     url="@MACOS_X86_64_URL@"; sum="@MACOS_X86_64_SHA@" ;;
  esac
  archive="$HERE/.python-download.tar.gz"
  echo "Downloading $url"
  curl -fL --retry 3 -o "$archive" "$url" || return 1
  echo "$sum  $archive" | shasum -a 256 -c - || { rm -f "$archive"; echo "The download is damaged."; return 1; }
  rm -rf "$RUNTIME" && mkdir -p "$RUNTIME" || return 1
  tar -xzf "$archive" -C "$RUNTIME" --strip-components 1 || { rm -rf "$RUNTIME"; return 1; }
  rm -f "$archive"
}

if [ -f "$RUNTIME/CHARTLIBRE_READY" ]; then
  # A ready-made package arrives quarantined, every file of it; once the
  # app itself has been allowed to open, the rest of its folder is too.
  if xattr -p com.apple.quarantine "$RUNTIME/bin/python3" >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$HERE" 2>/dev/null
  fi
  exec "$RUNTIME/bin/python3" "$HERE/main.py" "$@"
fi
if [ -x "$VENV/bin/python3" ]; then  # an install made before .python
  exec "$VENV/bin/python3" "$HERE/main.py" "$@"
fi

ask "First launch: ChartLibre will download what it needs into its own folder - Python and its scientific libraries: about 400 MB to download, 1.7 GB on disk.\n\nThis needs an internet connection and takes a few minutes. ChartLibre opens by itself when it is ready." "Install" || exit 0
say "Step 1 of 2: downloading Python..."
if ! { get_python \
       && say "Step 2 of 2: installing the scientific libraries. ChartLibre opens when it is ready." \
       && "$RUNTIME/bin/python3" "$HERE/install.py" --runtime; } >"$LOG" 2>&1; then
  osascript -e "display dialog \"The installation did not complete. The details are in:\n$LOG\n\nCheck the internet connection and open ChartLibre again.\" buttons {\"OK\"} with title \"ChartLibre\" with icon stop" >/dev/null 2>&1
  exit 1
fi
exec "$RUNTIME/bin/python3" "$HERE/main.py" "$@"
""")

#: The Windows launcher. %~dp0 is the folder the .bat sits in. curl.exe,
#: tar.exe and certutil come with Windows 10 (1803 and later) and 11.
WINDOWS_SCRIPT = _fill(r"""@echo off
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
set "URL=@WINDOWS_X86_64_URL@"
set "SUM=@WINDOWS_X86_64_SHA@"
set "ARCHIVE=.python-download.tar.gz"
echo Downloading Python @PYTHON_VERSION@ ...
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
""")


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

#: The Linux launcher. Like the .bat: from its own folder, download Python and
#: install on the first run, then start. install.py puts it in the
#: applications menu.
LINUX_SCRIPT = _fill(r"""#!/bin/sh
# ChartLibre launcher for Linux (dev/tools/make_launcher.py). It works from
# wherever the ChartLibre folder is. The first launch downloads Python and
# the libraries into .python inside that folder (a ready-made package
# already has them); every later launch just starts the application.
# install.py adds it to the applications menu.
cd "$(dirname "$0")" || exit 1

get_python() {  # download Python @PYTHON_VERSION@ into .python, once
  [ -x .python/bin/python3 ] && return 0
  case "$(uname -m)" in
    x86_64|amd64)  url="@LINUX_X86_64_URL@"; sum="@LINUX_X86_64_SHA@" ;;
    aarch64|arm64) url="@LINUX_AARCH64_URL@"; sum="@LINUX_AARCH64_SHA@" ;;
    *) echo "ChartLibre has no Python for this processor ($(uname -m))."; return 1 ;;
  esac
  archive=.python-download.tar.gz
  echo "Downloading Python @PYTHON_VERSION@ ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$archive" "$url" || return 1
  else
    wget -O "$archive" "$url" || return 1
  fi
  echo "$sum  $archive" | sha256sum -c - || { rm -f "$archive"; echo "The download is damaged."; return 1; }
  rm -rf .python && mkdir -p .python || return 1
  tar -xzf "$archive" -C .python --strip-components 1 || { rm -rf .python; return 1; }
  rm -f "$archive"
}

if [ -f .python/CHARTLIBRE_READY ]; then
  # A ready-made package: add the menu entry the first time it runs here.
  [ -f "$HOME/.local/share/applications/chartlibre.desktop" ] || .python/bin/python3 install.py --menu-only
  exec .python/bin/python3 main.py "$@"
fi
if [ -x .venv/bin/python3 ]; then  # an install made before .python
  exec .venv/bin/python3 main.py "$@"
fi

echo "First launch: ChartLibre downloads what it needs into this folder -"
echo "Python and its scientific libraries: about 400 MB to download, 1.7 GB"
echo "on disk. This needs an internet connection and takes a few minutes."
get_python || exit 1
.python/bin/python3 install.py --runtime || exit 1
exec .python/bin/python3 main.py "$@"
""")

#: The icon the Linux menu entry shows, tracked so install.py needs no Qt.
LINUX_ICON = LAUNCHER_SOURCE / "chartlibre.png"


#: The icon of the Windows installer (dev/tools/installer/chartlibre.iss).
WINDOWS_ICON = LAUNCHER_SOURCE / "chartlibre.ico"


def make_windows_icon() -> Path:
    """chartlibre.ico, every size Windows asks for, for the installer."""
    from io import BytesIO

    from PIL import Image

    Image.open(BytesIO(_render_icon_pngs([256])[256])).save(
        WINDOWS_ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )
    return WINDOWS_ICON


def make_linux_launcher(target_dir: Path) -> Path:
    """ChartLibre.sh (executable) and the icon its menu entry uses."""
    launcher = target_dir / f"{APP_NAME}.sh"
    launcher.write_text(LINUX_SCRIPT, encoding="utf-8")
    launcher.chmod(0o755)
    LINUX_ICON.write_bytes(_render_icon_pngs([256])[256])
    return launcher


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
    print(make_linux_launcher(target_dir))
    print(make_windows_icon())
    exe = make_windows_exe(target_dir)
    if exe is not None:
        print(exe)


if __name__ == "__main__":
    main()
