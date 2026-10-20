"""A window larger than its screen is fitted as it is shown, not just after: it does not jump (bugs2.pptx)."""
from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog

from app.utils.screen_fit import fit_on_show


def test_fitted_before_it_appears(qapp) -> None:
    available = QGuiApplication.primaryScreen().availableGeometry()
    dialog = QDialog()
    dialog.resize(available.width() + 300, available.height() + 300)
    fit_on_show(dialog)
    dialog.show()  # no event loop turn yet: only the Show event's own fit has run
    try:
        assert dialog.width() <= available.width() and dialog.height() <= available.height()
    finally:
        dialog.close()
