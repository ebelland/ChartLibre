"""One option box for a marker or a line style: Standard (the list), Custom (typed), Symbol (a ready-made one).

It wraps the combo that the series panel already reads and writes: whatever is chosen here ends up selected in
the combo (a custom value gets an entry of its own there), so loading, saving and Undo stay the combo's.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.utils.i18n import _

STANDARD, CUSTOM, SYMBOL, PICTURE = "standard", "custom", "symbol", "picture"
PREFIX = "image:"

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
                 hint: str, symbols: tuple[str, ...] | None = None, pictures: Any = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._combo, self._is_custom, self._select = combo, is_custom, select
        self._pictures = pictures  # () -> the project's repository (or None): its pictures (markers only)
        index = combo.findData("__custom__")
        if index >= 0:
            combo.removeItem(index)  # the option box takes its place

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        row = QGridLayout()  # one choice a line, what it needs beside it
        row.setColumnStretch(1, 1)
        row.setHorizontalSpacing(10)
        row.setVerticalSpacing(8)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, QRadioButton] = {}
        choices = [(STANDARD, _("Standard")), (CUSTOM, _("Custom"))] + ([(SYMBOL, _("Symbol"))] if symbols else []) \
            + ([(PICTURE, _("Picture"))] if pictures is not None else [])
        for key, label in choices:
            button = QRadioButton(label, self)
            self._group.addButton(button)
            self._buttons[key] = button
            button.toggled.connect(lambda on, k=key: on and self._show(k))
            row.addWidget(button, len(self._buttons) - 1, 0)
        column.addLayout(row)

        self._text = QLineEdit(self)
        self._text.setPlaceholderText(hint)
        self._text.setToolTip(hint)
        self._text.editingFinished.connect(self._typed)
        # Applied while typing too, a moment after the last key: waiting for Return or a click elsewhere
        # looked as if the pattern did nothing.
        from PySide6.QtCore import QTimer

        self._typing = QTimer(self)
        self._typing.setSingleShot(True)
        self._typing.setInterval(500)
        self._typing.timeout.connect(self._typed)
        self._text.textEdited.connect(lambda _t: self._typing.start())
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
        self._picture_row = QWidget(self)
        picture_row = QHBoxLayout(self._picture_row)
        picture_row.setContentsMargins(0, 0, 0, 0)
        self._picture_list = QComboBox(self._picture_row)
        self._picture_list.setToolTip(_("The pictures kept in this project"))
        self._picture_list.activated.connect(
            lambda _i: self._select_value(PREFIX + str(self._picture_list.currentData() or "")))
        picture_row.addWidget(self._picture_list, 1)
        add = QPushButton(_("Add…"), self._picture_row)
        add.setToolTip(_("Add an SVG, PNG or JPEG picture to the project and use it as the marker"))
        add.clicked.connect(self._add_picture)
        picture_row.addWidget(add)
        self._editors = {STANDARD: combo, CUSTOM: self._text, SYMBOL: self._symbols, PICTURE: self._picture_row}
        for index, (key, _label) in enumerate(choices):
            row.addWidget(self._editors[key], index, 1)
        for key, editor in self._editors.items():
            if key not in self._buttons:
                editor.setVisible(False)
        self._buttons[STANDARD].setChecked(True)
        self._show(STANDARD)

    def _show(self, key: str) -> None:
        for other, editor in self._editors.items():  # all shown, beside their choice; only the chosen one on
            editor.setEnabled(other == key)
        if key == PICTURE:
            self._fill_pictures()
            name = self._picture_list.currentData()
            if name and not self._value().startswith(PREFIX):
                self._select_value(PREFIX + str(name))
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

    def _repo(self) -> Any:
        return self._pictures() if callable(self._pictures) else None

    def _fill_pictures(self) -> None:
        repo = self._repo()
        names = repo.image_names() if repo is not None else []
        current = self._value()[len(PREFIX):] if self._value().startswith(PREFIX) else ""
        self._picture_list.clear()
        for name in names:
            self._picture_list.addItem(name, name)
        if current:
            index = self._picture_list.findData(current)
            if index < 0:
                self._picture_list.addItem(current + " " + _("(missing)"), current)
                index = self._picture_list.count() - 1
            self._picture_list.setCurrentIndex(index)

    def _add_picture(self) -> None:
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog, QMessageBox

        repo = self._repo()
        if repo is None:
            return
        path, _filter = QFileDialog.getOpenFileName(self, _("Add picture"), "",
                                                    _("Pictures (*.svg *.png *.jpg *.jpeg)"))
        if not path:
            return
        try:
            name = repo.add_image(Path(path).stem, Path(path).read_bytes(), source=Path(path).name)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, _("Add picture"), str(error))
            return
        self._fill_pictures()
        self._select_value(PREFIX + name)
        self._picture_list.setCurrentIndex(self._picture_list.findData(name))

    def sync(self) -> None:
        """Show the choice that holds what the combo has now (after a series is loaded)."""
        value = self._value()
        key = STANDARD
        if value.startswith(PREFIX) and PICTURE in self._buttons:
            key = PICTURE
        elif self._is_custom(value):
            key = SYMBOL if SYMBOL in self._buttons and self._symbols.findData(value) >= 0 else CUSTOM
        if key == CUSTOM:
            self._text.setText(value)
            self._text.setStyleSheet("")
        if key == SYMBOL:
            self._symbols.setCurrentIndex(self._symbols.findData(value))
        for button in self._buttons.values():
            button.blockSignals(True)
        self._buttons[key].setChecked(True)
        for button in self._buttons.values():
            button.blockSignals(False)
        self._show(key)
