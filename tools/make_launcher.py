"""Make a double-clickable launcher for ChartLibre, for the machine it runs on.

    python3 tools/make_launcher.py            # into dist/
    python3 tools/make_launcher.py ~/Applications

macOS: ``ChartLibre.app`` - a small bundle whose executable starts main.py
with the Python that ran this script, and whose icon is the application's
own. Drag it to Applications or the Dock.

Windows: ``ChartLibre.bat`` - starts main.py with pythonw, so no console
window stays open; pin a shortcut to it on the Start menu or the taskbar.

This is a launcher, not a standalone build: it uses the Python and the
packages already installed (see requirements.txt). A self-contained
executable would need PyInstaller and bundle PySide6, SciPy and the rest -
well over a gigabyte.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
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
        image.save(buffer, "PNG")
        images[size] = bytes(buffer.data())
    del app
    return images


def make_macos_app(target_dir: Path) -> Path:
    app_dir = target_dir / f"{APP_NAME}.app"
    if app_dir.exists():
        shutil.rmtree(app_dir)
    macos = app_dir / "Contents" / "MacOS"
    resources = app_dir / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    executable = macos / APP_NAME
    executable.write_text(
        "#!/bin/bash\n"
        "# Made by tools/make_launcher.py: starts ChartLibre from its project folder.\n"
        f'cd "{PROJECT_DIR}" || exit 1\n'
        f'exec "{sys.executable}" "{PROJECT_DIR / "main.py"}" "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)

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
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    interpreter = pythonw if pythonw.exists() else Path(sys.executable)
    launcher = target_dir / f"{APP_NAME}.bat"
    launcher.write_text(
        "@echo off\r\n"
        "rem Made by tools/make_launcher.py: starts ChartLibre without a console window.\r\n"
        f'cd /d "{PROJECT_DIR}"\r\n'
        f'start "" "{interpreter}" "{PROJECT_DIR / "main.py"}" %*\r\n',
        encoding="utf-8",
    )
    (target_dir / f"{APP_NAME}.png").write_bytes(_render_icon_pngs([256])[256])
    return launcher


def main() -> None:
    target_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else PROJECT_DIR / "dist"
    target_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    if system == "Darwin":
        made = make_macos_app(target_dir)
    elif system == "Windows":
        made = make_windows_launcher(target_dir)
    else:
        raise SystemExit("Linux: run `python3 main.py`, or add a .desktop entry pointing at it.")
    print(made)


if __name__ == "__main__":
    main()
