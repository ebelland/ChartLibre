"""Keep every window on the screen it opens on.

A dialog's size comes from three places - the shared sizes in style.py, the
geometry user.json remembers, a layout's own minimum - and none of them knows
the screen. 1020 x 700 plus a title bar is taller than a 1366 x 768 laptop
leaves above the Windows taskbar; a geometry saved on a 27" monitor is larger
still when the laptop is opened alone; and a window whose bottom is off the
screen hides its OK and Cancel. So when any window is shown, this shrinks it
to the screen's available area - never below its own minimum size - and
moves it fully onto that area.

Installed per window - every dialog through style.apply_dialog_shell, and the
main window - by :func:`fit_on_show`.
"""
from __future__ import annotations

import shiboken6
from PySide6.QtCore import QEvent, QObject, QRect, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

from app.logs.logger import applogger


def fit_to_screen(window: QWidget, *, assumed_frame: tuple[int, int] = (0, 0)) -> bool:
    """Shrink and move *window* into its screen's available area; True if it changed.

    Not a full-screen or maximised window: the system sized it to the screen.
    *assumed_frame* (width, height) stands for the title bar and borders of a
    window not on screen yet, whose frame the system has not measured.
    """
    if window.isFullScreen() or window.isMaximized() or window.isMinimized():
        return False
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return False
    available: QRect = screen.availableGeometry()
    frame = window.frameGeometry()
    # The title bar and borders: the part of the frame outside the client.
    extra_width = max(0, frame.width() - window.width()) or assumed_frame[0]
    extra_height = max(0, frame.height() - window.height()) or assumed_frame[1]
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
                available.width(), available.height(),
            )
    frame = window.frameGeometry()
    x = min(max(frame.x(), available.left()), max(available.left(), available.right() + 1 - frame.width()))
    y = min(max(frame.y(), available.top()), max(available.top(), available.bottom() + 1 - frame.height()))
    if (x, y) != (frame.x(), frame.y()):
        window.move(x, y)
        changed = True
    return changed


class _FitOnShow(QObject):
    """Watches one window - and only it - and fits it each time it is shown."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            # Now, before the window is on screen: fitted afterwards, it
            # appeared at one size and place and jumped to another - the
            # flicker of Fit Curve on a laptop (bugs2.pptx). The title bar is
            # not measured yet, so a usual one is assumed.
            if not watched.property(_FITTED_ONCE):
                watched.setProperty(_FITTED_ONCE, True)
                fit_to_screen(watched, assumed_frame=_USUAL_FRAME)
            # And after the show completes: a saved geometry is sometimes
            # restored by the window's own showEvent, which runs after this
            # filter. Usually there is nothing left to change.
            QTimer.singleShot(0, lambda window=watched: _fit_if_alive(window))
        return False


def _fit_if_alive(window: QWidget) -> None:
    if shiboken6.isValid(window) and window.isVisible():
        fit_to_screen(window)


def fit_on_show(window: QWidget) -> None:
    """Fit *window* to its screen every time it is shown.

    Installed on the window itself. An application-wide filter did the same
    for every window at once and crashed the application: PySide wraps
    every object an application filter sees, Qt's internal ones included,
    and wrapping one that was being destroyed is a segmentation fault - as
    well as a Python call for every event of every object.
    """
    if window.property(_INSTALLED):
        return
    window.setProperty(_INSTALLED, True)
    window.installEventFilter(_FitOnShow(window))


_INSTALLED = "_chartlibre_fit_on_show"
_FITTED_ONCE = "_chartlibre_fitted_before_show"
#: A title bar and borders, before the system has measured the window's own.
_USUAL_FRAME = (16, 40)
