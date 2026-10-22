# -*- coding: utf-8 -*-
"""Visual Matplotlib hatch selector and reusable preview icons."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import QWidget

from app.styles.style import create_hidpi_pixmap
from app.widgets.drop_down_controls.icon_combo import ComboEntry, IconComboBox

_ICON = QSize(40, 16)
_ENTRIES: tuple[ComboEntry, ...] = (
    ("None", ""),
    ("Diagonal /", "/"),
    ("Back diagonal \\", "\\"),
    ("Vertical |", "|"),
    ("Horizontal -", "-"),
    ("Cross +", "+"),
    ("Diagonal cross x", "x"),
    ("Small circles o", "o"),
    ("Large circles O", "O"),
    ("Dots .", "."),
    ("Stars *", "*"),
)


@lru_cache(maxsize=128)
def hatch_preview_icon(hatch: str) -> QIcon:
    """Return a compact tile rendered with Qt's closest hatch pattern."""
    pixmap = create_hidpi_pixmap(_ICON.width(), _ICON.height())
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRect(1, 1, _ICON.width() - 2, _ICON.height() - 2)
        painter.setPen(QPen(QColor("#666666")))
        painter.setBrush(QBrush(QColor("#fafafa")))
        painter.drawRect(rect)
        if not hatch:
            return QIcon(pixmap)
        pen = QPen(QColor("#3a3a3a"))
        pen.setWidth(1)
        painter.setPen(pen)
        step = 6
        if "/" in hatch or "x" in hatch:
            for x in range(-rect.height(), rect.width(), step):
                painter.drawLine(rect.left() + x, rect.bottom(), rect.left() + x + rect.height(), rect.top())
        if "\\" in hatch or "x" in hatch:
            for x in range(0, rect.width() + rect.height(), step):
                painter.drawLine(rect.left() + x, rect.top(), rect.left() + x - rect.height(), rect.bottom())
        if "|" in hatch or "+" in hatch:
            for x in range(rect.left() + 3, rect.right(), step):
                painter.drawLine(x, rect.top(), x, rect.bottom())
        if "-" in hatch or "+" in hatch:
            for y in range(rect.top() + 3, rect.bottom(), step):
                painter.drawLine(rect.left(), y, rect.right(), y)
        if any(symbol in hatch for symbol in ("o", "O", ".", "*")):
            radius = 1 if "." in hatch else (3 if "O" in hatch else 2)
            for y in range(rect.top() + 4, rect.bottom(), step):
                for x in range(rect.left() + 4, rect.right(), step):
                    if "*" in hatch:
                        painter.drawLine(x - 2, y, x + 2, y)
                        painter.drawLine(x, y - 2, x, y + 2)
                    else:
                        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
    finally:
        painter.end()
    return QIcon(pixmap)


class HatchStyleCombo(IconComboBox):
    """Combo box for Matplotlib hatch tokens with pattern previews."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, entries=_ENTRIES, icon_for=hatch_preview_icon, icon_size=_ICON)
        self.setEditable(True)

    def current_hatch(self) -> str:
        return self.currentText().strip()

    def set_current_hatch(self, hatch: str | None) -> None:
        text = str(hatch or "")
        if not self.set_current_value(text):
            self.setEditText(text)
