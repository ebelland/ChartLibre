"""Searchable Matplotlib font-family selector with live previews."""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from matplotlib import font_manager
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter
from PySide6.QtWidgets import QComboBox, QCompleter, QWidget

from app.styles.style import create_hidpi_pixmap

_GENERIC_FAMILIES = ("sans-serif", "serif", "monospace", "cursive", "fantasy")
_FONT_ICON = QSize(42, 18)


@lru_cache(maxsize=1024)
def font_preview_icon(family: str) -> QIcon:
    """Render 'Aa' in the selected family for combo and tree previews."""
    if not family:
        return QIcon()
    pixmap = create_hidpi_pixmap(_FONT_ICON.width(), _FONT_ICON.height())
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = QFont(family)
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor("#3a3a3a"))
        painter.drawText(QRect(0, 0, _FONT_ICON.width(), _FONT_ICON.height()),
                         Qt.AlignmentFlag.AlignCenter, "Aa")
    finally:
        painter.end()
    return QIcon(pixmap)


class MatplotlibFontCombo(QComboBox):
    fontChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None, *, include_none: bool = True,
                 include_generic: bool = True, extra_fonts: Iterable[str] = ()) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setIconSize(_FONT_ICON)
        names = list(_GENERIC_FAMILIES if include_generic else ())
        names += sorted({e.name.strip() for e in font_manager.fontManager.ttflist
                         if e.name.strip() and not e.name.startswith(".")}, key=str.casefold)
        seen: set[str] = set()
        if include_none:
            self.addItem("", None)
        for value in (*names, *extra_fonts):
            name = str(value).strip()
            if name and name.casefold() not in seen:
                seen.add(name.casefold())
                self.addItem(font_preview_icon(name), name, name)
        completer = QCompleter(self.model(), self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.setCompleter(completer)
        self.currentTextChanged.connect(lambda *_: self.fontChanged.emit(self.current_font()))

    def current_font(self) -> str | None:
        return self.currentText().strip() or None

    def set_current_font(self, family: object) -> bool:
        text = "" if family is None else str(family).strip()
        index = self.findText(text, Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self.setCurrentIndex(index)
            return True
        self.setEditText(text)
        return False
