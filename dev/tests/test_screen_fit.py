"""Every window fits the screen it opens on (laptops included)."""
from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog, QMenu

from app.utils import screen_fit


def _available() -> QRect:
    screen = QGuiApplication.primaryScreen()
    assert screen is not None
    return screen.availableGeometry()


def test_a_window_larger_than_the_screen_is_shrunk_and_moved_onto_it(qapp) -> None:
    fitter = screen_fit.install(qapp)
    try:
        dialog = QDialog()
        available = _available()
        dialog.resize(available.width() + 500, available.height() + 400)
        dialog.move(available.right() - 50, available.bottom() - 50)
        dialog.show()
        QApplication.processEvents()
        QApplication.processEvents()
        frame = dialog.frameGeometry()
        assert dialog.width() <= available.width() and dialog.height() <= available.height()
        assert available.contains(frame.topLeft()) and frame.bottom() <= available.bottom() + 1
        dialog.close()
    finally:
        qapp.removeEventFilter(fitter)


def test_a_window_is_never_shrunk_below_its_minimum(qapp) -> None:
    dialog = QDialog()
    available = _available()
    dialog.setMinimumSize(available.width() + 10, 100)
    dialog.resize(available.width() + 200, 200)
    screen_fit.fit_to_screen(dialog)
    assert dialog.width() == available.width() + 10


def test_a_window_that_fits_is_left_alone_and_menus_are_not_windows(qapp) -> None:
    dialog = QDialog()
    dialog.resize(300, 200)
    dialog.move(_available().topLeft())
    dialog.show()
    QApplication.processEvents()
    assert not screen_fit.fit_to_screen(dialog)
    dialog.close()
    assert QMenu().windowType() in screen_fit._NOT_WINDOWS
