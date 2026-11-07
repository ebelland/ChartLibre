"""Qt's plugins made visible to Qt on macOS: imported, and run, before any Qt module but QtCore's package.

Shared by ChartLibre.py and the tests (dev/tests/conftest.py): both start Qt from the same environment.
"""
from __future__ import annotations

import os
import sys

from PySide6 import __file__ as PYSIDE6_FILE


def _qt_plugins_outside_dot_folder(plugins_dir: str) -> None:
    """Point Qt at links to its plugins kept outside a hidden folder.

    iCloud's Desktop & Documents sync marks everything inside a folder whose
    name starts with a dot as hidden - a .venv on the Desktop included - and,
    as it turned out, inside a .nosync folder too (python.nosync). Qt
    skips hidden files when it lists a plugin folder, finds no "cocoa"
    platform plugin, and aborts before a window exists ("Could not find the
    Qt platform plugin"); the style and SVG icon plugins vanish the same way.

    Clearing the flag at start-up was not enough: iCloud sets it again within
    seconds, so whether the application started depended on who won the
    race. Instead a folder of symbolic links to every plugin is kept in
    ~/Library/Caches, which iCloud does not touch, and handed to Qt through
    QT_PLUGIN_PATH; Qt lists the links, which are not hidden, and loads what
    they point to. A few hundred links, rebuilt in milliseconds when needed.
    """
    if sys.platform != "darwin":
        return
    plugins = os.path.realpath(plugins_dir)
    if not _plugins_hidden(plugins):
        return  # Qt sees its plugins: nothing to do

    import hashlib

    from PySide6 import __version__ as pyside_version

    key = hashlib.sha1(f"{plugins}|{pyside_version}".encode()).hexdigest()[:12]
    mirror = os.path.join(os.path.expanduser("~/Library/Caches/ChartLibre/qt-plugins"), key)
    try:
        for folder, _dirs, files in os.walk(plugins):
            target_folder = os.path.join(mirror, os.path.relpath(folder, plugins))
            os.makedirs(target_folder, exist_ok=True)
            for name in files:
                source = os.path.join(folder, name)
                link = os.path.join(target_folder, name)
                if os.path.islink(link) and os.readlink(link) == source:
                    continue
                if os.path.lexists(link):
                    os.remove(link)
                os.symlink(source, link)
    except OSError:
        return  # leave Qt to its own plugin folder, and to say why if it fails
    existing = os.environ.get("QT_PLUGIN_PATH")
    os.environ["QT_PLUGIN_PATH"] = mirror if not existing else os.pathsep.join([mirror, existing])


def _plugins_hidden(plugins: str) -> bool:
    """Whether Qt would skip its plugins: inside a dot-folder, or the macOS platform plugin flagged hidden."""
    if any(part.startswith(".") for part in plugins.split(os.sep) if part):
        return True
    import stat

    try:
        return bool(os.stat(os.path.join(plugins, "platforms", "libqcocoa.dylib")).st_flags & stat.UF_HIDDEN)
    except (OSError, AttributeError):
        return False


def ensure_visible_plugins() -> None:
    """Run the check on this PySide6's plugin folder."""
    _qt_plugins_outside_dot_folder(os.path.join(os.path.dirname(PYSIDE6_FILE), "Qt", "plugins"))
