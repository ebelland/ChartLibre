# -*- coding: utf-8 -*-
"""Compact colour editor backed by Qt's standard QColorDialog.

The value is a Matplotlib colour as the user wrote it: a name ("red", "tab:blue"),
a cycle reference ("C0"), a grey level ("0.5") or hex, "#rrggbbaa" with an alpha.
Qt knows only some of those and reads an alpha as "#aarrggbb", so colours go to and
from Qt as numbers, never as the text, and a value Qt cannot read is kept as it is.
"""
from __future__ import annotations

from matplotlib.colors import to_hex, to_rgba
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QPushButton, QWidget

from app.widgets.drop_down_controls.color_combo import color_preview_icon


def qcolor_from_matplotlib(value: str) -> QColor | None:
    """The Qt colour of a Matplotlib colour *value*; None when Matplotlib does not know it."""
    try:
        red, green, blue, alpha = to_rgba(value)
    except ValueError:
        return None
    return QColor.fromRgbF(red, green, blue, alpha)


def matplotlib_hex(color: QColor) -> str:
    """*color* as Matplotlib writes hex: "#rrggbb", or "#rrggbbaa" when it is not opaque."""
    return to_hex((color.redF(), color.greenF(), color.blueF(), color.alphaF()), keep_alpha=color.alpha() < 255)


class ColorPickerButton(QPushButton):
    colorChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = ""
        self.clicked.connect(self._choose_color)
        self._refresh()

    def color(self) -> str:
        return self._color

    def set_color(self, value: object) -> None:
        """Show *value*; what was written stays as written, even a name Qt cannot read."""
        self._color = str(value or "").strip()
        self._refresh()

    def _refresh(self) -> None:
        self.setIcon(color_preview_icon(self._color))
        self.setText(self._color or "Default")
        self.setToolTip("Choose colour" if self._color else "Choose colour (currently default)")

    def _choose_color(self) -> None:
        initial = (qcolor_from_matplotlib(self._color) if self._color else None) or QColor("#1f77b4")
        dialog = QColorDialog(initial, self)
        dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        if dialog.exec() == QColorDialog.DialogCode.Accepted:
            self.set_color(matplotlib_hex(dialog.selectedColor()))
            self.colorChanged.emit(self._color)
