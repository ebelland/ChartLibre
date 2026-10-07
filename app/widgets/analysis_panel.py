"""The Analysis panel: the table operations, by category, each with its sentence.

Fit Model and the analyses to come work on a table's columns rather than on
a chart's series (see app/table_operations/). This panel lists them as the
scanner finds them; a click asks the window to open one on the table
selected in the Tables panel.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from app.scanners.table_operation_scanner import table_operations
from app.styles.style import icon_from_svg_source
from app.utils.i18n import _
from app.widgets.nav_bar import NavPanel


class AnalysisPanel(NavPanel):
    """A frame per category; a button and its description per operation."""

    #: An operation's discovery entry, to open.
    operation_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None, operations: list[dict] | None = None) -> None:
        super().__init__(parent)
        entries = list(table_operations if operations is None else operations)
        categories: dict[str, list[dict]] = {}
        for entry in entries:
            categories.setdefault(str(entry.get("category") or "Other"), []).append(entry)
        for category, items in categories.items():
            body = QWidget(self)
            layout = QVBoxLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
            for entry in items:
                button = QPushButton(_(str(entry.get("value") or entry.get("name"))), body)
                if entry.get("icon"):
                    button.setIcon(icon_from_svg_source(str(entry["icon"]), size=18))
                    button.setIconSize(QSize(18, 18))
                button.setToolTip(_(str(entry.get("description") or "")))
                button.clicked.connect(lambda _checked=False, e=entry: self.operation_requested.emit(e))
                layout.addWidget(button)
                description = QLabel(_(str(entry.get("description") or "")), body)
                description.setWordWrap(True)
                description.setProperty("muted", True)
                layout.addWidget(description)
            self.add_frame(_(category), body, object_name=f"analysis{category.replace(' ', '')}Frame")
