from __future__ import annotations

import os
import sys

from PySide6 import __file__ as PYSIDE6_FILE


def _qt_plugins_outside_dot_folder(plugins_dir: str) -> None:
    """Point Qt at links to its plugins kept outside a hidden folder.

    iCloud's Desktop & Documents sync marks everything inside a folder whose
    name starts with a dot as hidden - a .venv on the Desktop included. Qt
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
    if not any(part.startswith(".") for part in plugins.split(os.sep) if part):
        return  # not inside a dot-folder: nothing hides it

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


_qt_plugins_outside_dot_folder(os.path.join(os.path.dirname(PYSIDE6_FILE), "Qt", "plugins"))
# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
from app.logs.logger import AppLogger, applogger
AppLogger.configure()
applogger.debug("Application starting...")
from PySide6.QtWidgets import QApplication, QMessageBox

from app import APP_NAME, APP_VERSION
from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow

from app.styles.style import apply_platform_style, ensure_icon_theme
from app.utils.config import get_language, set_last_database
from app.utils.i18n import install_qt_translations, set_language
from app.utils.startup import select_database

# ----------------------------------------------------------------------
# CLI / bootstrap
# ----------------------------------------------------------------------

def run_app() -> int:
    # Before the QApplication is built: on macOS this is what Cocoa's native
    # Application menu reads for the two entries it labels itself rather than
    # from any QAction's own text - "About {name}" for the one carrying
    # AboutRole, "Quit {name}" for the one every Qt app gets whether it adds
    # one or not. Unset, both read the running executable's name instead -
    # "About Python", "Quit Python" - because this is not a signed .app
    # bundle with its own Info.plist to name it. Nothing about the two
    # entries themselves can be renamed independently of the app's own name;
    # that pairing is Cocoa's, not this application's.
    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(APP_VERSION)
    QApplication.setOrganizationName(APP_NAME)

    app = QApplication(sys.argv)

    # sys._jit is new in Python 3.14: an older Python has no JIT to report.
    jit_module = getattr(sys, "_jit", None)
    if jit_module is None:
        applogger.info("JIT: not available in Python %s", sys.version.split()[0])
    elif not jit_module.is_available():
        applogger.info("JIT is not included in the build")
    else:
        applogger.info("JIT is included in the build %s",
                       "and enabled" if jit_module.is_enabled() else "but not enabled")

    # Language before any widget is built: menus read their labels through the
    # translator when they are constructed.
    applogger.info("Interface language: %s", set_language(get_language()))

    # Qt's own strings, which ours cannot reach: QMessageBox builds its Yes
    # and No from Qt's catalogue, so every confirmation in the app asked in
    # the user's language and answered in English until this was installed.
    install_qt_translations(app)

    apply_platform_style(app)

    # Icons come from the desktop's own theme wherever there is one. GNOME and
    # KDE name it themselves; a bare window manager does not, and without this
    # that whole class of desktop silently falls back to the shipped SVGs.
    applogger.info("Icon theme: %s", ensure_icon_theme() or "none installed")

    try:
        # Which database this run is about, and what to do when there is
        # none, lives in app/utils/startup.py - it shows boxes, and those
        # come from the catalogue every other box in the app comes from.
        db_path = select_database(sys.argv[1] if len(sys.argv) > 1 else None)
        if db_path is None:
            applogger.fatal("No database selected. Application exiting.")
            return 0

        repo = SqliteRepo(db_path=db_path)
        if repo is None:
            applogger.fatal("Database not valid. Application exiting.")
            return 0
        applogger.info("Database selected: %s", db_path)
        set_last_database(db_path)

        window = MainWindow(repo=repo, db_path=db_path)
        window.show()

        return app.exec()

    except Exception as exc:  # noqa: BLE001
        applogger.exception("Application startup failed: %s", exc)
        QMessageBox.critical(None, APP_NAME, f"Application startup failed:\n{exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(run_app())
