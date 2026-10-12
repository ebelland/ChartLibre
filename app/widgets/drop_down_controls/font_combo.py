"""Searchable Matplotlib font-family selector."""
from __future__ import annotations
from collections.abc import Iterable
from matplotlib import font_manager
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QCompleter, QWidget

_GENERIC_FAMILIES = ("sans-serif", "serif", "monospace", "cursive", "fantasy")

class MatplotlibFontCombo(QComboBox):
    fontChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None, *, include_none: bool = True,
                 include_generic: bool = True, extra_fonts: Iterable[str] = ()) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
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
                self.addItem(name, name)
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
