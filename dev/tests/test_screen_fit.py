"""Every window fits the screen it opens on (laptops included)."""
from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout

from app.styles.style import apply_dialog_shell
from app.utils import screen_fit


def _available() -> QRect:
    screen = QGuiApplication.primaryScreen()
    assert screen is not None
    return screen.availableGeometry()


def _settle() -> None:
    for _ in range(3):
        QApplication.processEvents()


def test_a_dialog_larger_than_the_screen_is_shrunk_and_moved_onto_it(qapp) -> None:
    dialog = QDialog()
    apply_dialog_shell(dialog, QVBoxLayout(dialog), size=None)
    available = _available()
    dialog.resize(available.width() + 500, available.height() + 400)
    dialog.move(available.right() - 50, available.bottom() - 50)
    dialog.show()
    _settle()
    frame = dialog.frameGeometry()
    assert dialog.width() <= available.width() and dialog.height() <= available.height()
    assert available.contains(frame.topLeft()) and frame.bottom() <= available.bottom() + 1
    dialog.close()


def test_a_window_is_never_shrunk_below_its_minimum(qapp) -> None:
    dialog = QDialog()
    available = _available()
    dialog.setMinimumSize(available.width() + 10, 100)
    dialog.resize(available.width() + 200, 200)
    screen_fit.fit_to_screen(dialog)
    assert dialog.width() == available.width() + 10


def test_a_window_that_fits_is_left_alone_and_installing_twice_is_once(qapp) -> None:
    dialog = QDialog()
    screen_fit.fit_on_show(dialog)
    screen_fit.fit_on_show(dialog)
    assert len(dialog.findChildren(screen_fit._FitOnShow)) == 1
    dialog.resize(300, 200)
    dialog.move(_available().topLeft())
    dialog.show()
    _settle()
    assert not screen_fit.fit_to_screen(dialog)
    dialog.close()


def test_windows_destroyed_right_after_showing_are_harmless(qapp) -> None:
    """The crash of 2026-10-03: windows and their children created, shown and
    destroyed while a fit was pending."""
    for _ in range(50):
        dialog = QDialog()
        apply_dialog_shell(dialog, QVBoxLayout(dialog), size="small")
        QLabel("x", dialog)
        dialog.show()
        dialog.deleteLater()
    _settle()
