"""One option box for a marker or a line style: Standard (the list), Custom (typed), Symbol (a ready-made one).

It wraps the combo that the series panel already reads and writes: whatever is chosen here ends up selected in
the combo (a custom value gets an entry of its own there), so loading, saving and Undo stay the combo's.
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.utils.i18n import _

STANDARD, CUSTOM, SYMBOL = "standard", "custom", "symbol"

#: Symbols kept in memory: characters of DejaVu Sans, the font inside matplotlib, so they are drawn the same on
#: macOS, Windows and Linux; the ones the font lacks are left out at run time.
SYMBOLS = ("→", "←", "↑", "↓", "↗", "↘", "➜", "★", "☆", "♥", "♦", "♣", "♠", "●", "○", "■", "□", "▲", "▼", "◆",
           "◇", "✚", "✖", "✔", "✘", "☀", "☁", "☂", "✈", "⚑", "♀", "♂", "☺", "⌂", "⚡", "♪", "✿", "❄", "⚓", "☢")


class StyleChooser(QWidget):
    """Standard / Custom / Symbol around *combo*.

    *is_custom* says whether a value is one of its own; *select* puts a value in the combo (adding its entry);
    *hint* is the Custom box's help; *symbols* None leaves the Symbol choice out (line styles).
    """

    def __init__(self, combo: QComboBox, *, is_custom: Callable[[str], bool], select: Callable[[str], bool],
                 hint: str, symbols: tuple[str, ...] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._combo, self._is_custom, self._select = combo, is_custom, select
        index = combo.findData("__custom__")
        if index >= 0:
            combo.removeItem(index)  # the option box takes its place

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(10)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, QRadioButton] = {}
        choices = [(STANDARD, _("Standard")), (CUSTOM, _("Custom"))] + ([(SYMBOL, _("Symbol"))] if symbols else [])
        for key, label in choices:
            button = QRadioButton(label, self)
            self._group.addButton(button)
            self._buttons[key] = button
            button.toggled.connect(lambda on, k=key: on and self._show(k))
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)

        column.addWidget(combo)
        self._text = QLineEdit(self)
        self._text.setPlaceholderText(hint)
        self._text.setToolTip(hint)
        self._text.editingFinished.connect(self._typed)
        column.addWidget(self._text)
        self._symbols = QComboBox(self)
        if symbols:
            from app.plugin_api.charts.markers import _dejavu_has

            font = self._symbols.font()
            font.setFamily("DejaVu Sans")
            font.setPointSizeF(font.pointSizeF() * 1.3)
            self._symbols.setFont(font)
            for symbol in symbols:
                if _dejavu_has(symbol):
                    self._symbols.addItem(symbol, symbol)
            self._symbols.setIconSize(QSize(16, 16))
            self._symbols.activated.connect(lambda _i: self._select_value(self._symbols.currentData()))
        column.addWidget(self._symbols)
        self._buttons[STANDARD].setChecked(True)
        self._show(STANDARD)

    def _show(self, key: str) -> None:
        self._combo.setVisible(key == STANDARD)
        self._text.setVisible(key == CUSTOM)
        self._symbols.setVisible(key == SYMBOL)
        if key == SYMBOL and self._symbols.count() and self._symbols.findData(self._value()) < 0:
            self._select_value(self._symbols.currentData())

    def _value(self) -> str:
        return str(self._combo.currentData() or "")

    def _select_value(self, value: str) -> None:
        if value and value != self._value():
            self._select(value)

    def _typed(self) -> None:
        text = self._text.text().strip() if self._buttons[CUSTOM].isChecked() else ""
        if not text:
            return
        if self._is_custom(text) or self._combo.findData(text) >= 0:
            self._text.setStyleSheet("")
            self._select_value(text)
        else:
            self._text.setStyleSheet("color:#dc2626;")  # not drawable: kept typed, not applied

    def sync(self) -> None:
        """Show the choice that holds what the combo has now (after a series is loaded)."""
        value = self._value()
        key = STANDARD
        if self._is_custom(value):
            key = SYMBOL if SYMBOL in self._buttons and self._symbols.findData(value) >= 0 else CUSTOM
        if key == CUSTOM:
            self._text.setText(value)
            self._text.setStyleSheet("")
        if key == SYMBOL:
            self._symbols.setCurrentIndex(self._symbols.findData(value))
        self._group.blockSignals(True)
        self._buttons[key].setChecked(True)
        self._group.blockSignals(False)
        self._show(key)
