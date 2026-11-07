from __future__ import annotations

import os
import sys

from app.utils.qt_plugins import ensure_visible_plugins

# Before any Qt widget module: macOS may have flagged the plugins hidden (app/utils/qt_plugins.py).
ensure_visible_plugins()

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
from app.logs.logger import AppLogger, applogger, clear_log_file
from app.utils.config import clear_log_at_start

AppLogger.configure()
if clear_log_at_start():
    # Settings > Advanced: the log holds this session only.
    clear_log_file()
applogger.debug("Application starting...")
from PySide6.QtWidgets import QApplication, QMessageBox

from app import APP_NAME, APP_VERSION
from app.data.sqlite_repo import SqliteRepo
from app.main_window.main_window import MainWindow
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

    except Exception as exc:
        applogger.exception("Application startup failed: %s", exc)
        QMessageBox.critical(None, APP_NAME, f"Application startup failed:\n{exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(run_app())
