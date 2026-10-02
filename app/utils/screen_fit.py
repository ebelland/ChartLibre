"""Keep every window on the screen it opens on.

A dialog's size comes from three places - the shared sizes in style.py, the
geometry user.json remembers, a layout's own minimum - and none of them knows
the screen. 1020 x 700 plus a title bar is taller than a 1366 x 768 laptop
leaves above the Windows taskbar; a geometry saved on a 27" monitor is larger
still when the laptop is opened alone; and a window whose bottom is off the
screen hides its OK and Cancel. So when any window is shown, this shrinks it
to the screen's available area - never below its own minimum size - and
moves it fully onto that area.

Installed once on the application (main.py); nothing else needs to call it.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QRect, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget

from app.logs.logger import applogger


def fit_to_screen(window: QWidget) -> bool:
    """Shrink and move *window* into its screen's available area; True if it changed.

    Not a full-screen or maximised window: the system sized it to the screen.
    """
    if window.isFullScreen() or window.isMaximized() or window.isMinimized():
        return False
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return False
    available: QRect = screen.availableGeometry()
    frame = window.frameGeometry()
    # The title bar and borders: the part of the frame outside the client.
    extra_width = max(0, frame.width() - window.width())
    extra_height = max(0, frame.height() - window.height())
    width = min(window.width(), max(window.minimumWidth(), available.width() - extra_width))
    height = min(window.height(), max(window.minimumHeight(), available.height() - extra_height))
    changed = False
    if (width, height) != (window.width(), window.height()):
        window.resize(width, height)
        changed = True
        if width + extra_width > available.width() or height + extra_height > available.height():
            applogger.warning(
                "%s needs %d x %d at least, more than this screen's %d x %d.",
                type(window).__name__, window.minimumWidth(), window.minimumHeight(),
                available.width(), available.height(), show_dialog=False, raise_error=False,
            )
    frame = window.frameGeometry()
    x = min(max(frame.x(), available.left()), max(available.left(), available.right() + 1 - frame.width()))
    y = min(max(frame.y(), available.top()), max(available.top(), available.bottom() + 1 - frame.height()))
    if (x, y) != (frame.x(), frame.y()):
        window.move(x, y)
        changed = True
    return changed


class ScreenFitFilter(QObject):
    """Fits every top-level window to its screen as it is shown."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            event.type() == QEvent.Type.Show
            and isinstance(watched, QWidget)
            and watched.isWindow()
            and watched.windowType() not in _NOT_WINDOWS
        ):
            # After the show completes: a saved geometry is often restored
            # by the window's own showEvent, which runs after this filter.
            QTimer.singleShot(0, lambda window=watched: _fit_if_alive(window))
        return False


#: Top-level widgets that are not windows to fit: menus, combo lists, tips.
_NOT_WINDOWS = frozenset({Qt.WindowType.Popup, Qt.WindowType.ToolTip, Qt.WindowType.SplashScreen})


def _fit_if_alive(window: QWidget) -> None:
    try:
        if window.isVisible():
            fit_to_screen(window)
    except RuntimeError:  # deleted before the timer fired
        return


def install(app: QApplication) -> ScreenFitFilter:
    """Fit every window this application shows. Keep the returned filter alive."""
    fitter = ScreenFitFilter(app)
    app.installEventFilter(fitter)
    return fitter
