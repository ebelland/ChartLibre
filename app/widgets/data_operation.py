"""The Data Operations panel: New plot, then every operation, as a list.

A sectioned list in the manner of a macOS sidebar - Finder's, System
Settings' - and of the chart list in this window's own navigation rail:
small grey section titles, one row per operation with its icon and name,
the row under the pointer or the keyboard tinted, a click (or Return)
opening it. It replaced a grid of square tiles, which read as a page of
buttons rather than as the list of commands it is, and needed soft hyphens
to fit "Correzione linea di base" into a square.

The sections are the ones the layout proposal settled on - Plot on its
own, then Analysis, Statistics, Geometry, Signal Processing, Modeling -
because "I want a control chart" is one decision, not "which section is a
control chart in". Finding one by what it does is
the Guided analysis panel's search (and Cmd/Ctrl+K). The description of the
row under the pointer, or with keyboard focus, shows in a bar fixed at the
panel's bottom - not a tooltip, which vanishes the moment the pointer
moves and says nothing to someone using the keyboard.
"""
from __future__ import annotations

from html import unescape

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.scanners.data_operation_scanner import data_operations
from app.styles.style import icon_from_svg_source, load_icon
from app.utils.i18n import _, tr
from app.widgets.nav_bar import NavPanel

_ACCENT = "#2563EB"

#: The section of an operation that declares no Category. An operation names
#: its own section (DataOperationDialogBase.Category) and its place in it
#: (Order), so this panel knows no operation: one added or removed as a plugin
#: file changes the panel and nothing else. Titles are passed through _() in
#: _group_by_section, not here: this runs at import, before the saved language
#: is applied.
_FALLBACK_SECTION = "Other"


def _operation_action_id(operation: dict) -> str:
    return str(operation.get("value") or operation.get("name") or "")


def _group_by_section(operations: list[dict]) -> list[tuple[str, list[dict]]]:
    """Sort *operations* into sections by their Category, in Order.

    A section is placed by its lowest Order; operations without a Category
    form the last one, "Other".
    """
    def order(operation: dict) -> int:
        value = operation.get("order")
        return value if isinstance(value, int) else 0

    sections: dict[str, list[dict]] = {}
    for operation in operations:
        sections.setdefault(str(operation.get("category") or ""), []).append(operation)

    known = sorted(
        ((title, sorted(items, key=lambda op: (order(op), _operation_action_id(op)))) for title, items in sections.items() if title),
        key=lambda section: (min(order(op) for op in section[1]), section[0]),
    )
    grouped = [(_(title), items) for title, items in known]
    if "" in sections:
        grouped.append((_(_FALLBACK_SECTION), sections[""]))
    return grouped


#: Item data roles: the operation a row opens, and its description.
_OPERATION_ROLE = Qt.ItemDataRole.UserRole
_DESCRIPTION_ROLE = Qt.ItemDataRole.UserRole + 2

_ICON_SIZE = 18


class DataOperationWidget(NavPanel):
    """The Data operations panel: the operations, searchable, and the pointed one's description."""

    operation_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        page = QWidget(self)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget(page)
        self._list.setObjectName("dataOperationsList")
        self._list.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._list.setUniformItemSizes(False)
        self._list.setMouseTracking(True)
        self._list.itemClicked.connect(self._open_item)
        # A click opens - and only a click: itemActivated as well would open
        # twice on a double-click, or wherever one click activates. Return
        # is handled in eventFilter.
        self._list.itemEntered.connect(self._hint_for)
        self._list.currentItemChanged.connect(lambda current, _previous: self._hint_for(current))
        self._list.viewport().installEventFilter(self)
        self._list.installEventFilter(self)
        # Ignored vertically, like every list that fills a panel: it takes the
        # height left without asking the scrolling panel for its own.
        self._list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._list.setMinimumHeight(160)
        page_layout.addWidget(self._list, 1)
        self.add_frame(_("Operations"), page, object_name="dataOperationsPageCard", stretch=1)

        self._add_section(_("Plot"), [self.plot_operation()])
        for title, operations in _group_by_section(list(data_operations)):
            self._add_section(title, operations)

        # A frame of its own under the list: a description belongs somewhere
        # that stays put while it is read.
        self._hint_label = QLabel(_("Point at an operation for details"), self)
        self._hint_label.setObjectName("operationHint")
        self._hint_label.setProperty("muted", True)
        self._hint_label.setWordWrap(True)
        self.add_frame(_("Description"), self._hint_label, object_name="operationHintFrame")

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def _add_section(self, title: str, operations: list[dict]) -> None:
        header = QListWidgetItem(title, self._list)
        # Not selectable, not enabled: a title, read but never chosen - and
        # styled through :disabled in the sheets, like a sidebar heading.
        header.setFlags(Qt.ItemFlag.NoItemFlags)
        font = QFont(self._list.font())
        font.setBold(True)
        font.setPointSizeF(max(8.0, font.pointSizeF() * 0.85))
        header.setFont(font)
        header.setSizeHint(QSize(0, 26 if self._list.count() == 1 else 34))
        for operation in operations:
            name = tr(_operation_action_id(operation))
            description = tr(str(operation.get("description") or ""))
            item = QListWidgetItem(self.plugin_icon(operation), name, self._list)
            item.setData(_OPERATION_ROLE, operation)
            item.setData(_DESCRIPTION_ROLE, description)
            item.setToolTip(description)
            item.setSizeHint(QSize(0, 28))

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def operation_items(self) -> list[QListWidgetItem]:
        """The rows that open an operation, in order, hidden or not."""
        return [
            item for row in range(self._list.count())
            if (item := self._list.item(row)) is not None and item.data(_OPERATION_ROLE) is not None
        ]

    def _open_item(self, item: QListWidgetItem | None) -> None:
        operation = item.data(_OPERATION_ROLE) if item is not None else None
        if operation is None:
            return
        # A click opens; the row does not stay highlighted as if chosen.
        self._list.clearSelection()
        self.operation_requested.emit(operation)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            watched is self._list
            and event.type() == QEvent.Type.KeyPress
            and getattr(event, "key", lambda: None)() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            # Explicitly: on macOS a list view does not emit activated for
            # Return (Finder's Return renames), so it would do nothing here.
            self._open_item(self._list.currentItem())
            return True
        if watched is self._list.viewport() and event.type() == QEvent.Type.Leave:
            if not self._list.hasFocus():
                self._clear_hint()
        return super().eventFilter(watched, event)

    def _hint_for(self, item: QListWidgetItem | None) -> None:
        if item is None or item.data(_OPERATION_ROLE) is None:
            return
        self._show_hint(item.text(), str(item.data(_DESCRIPTION_ROLE) or ""))

    def _show_hint(self, title: str, description: str) -> None:
        self._hint_label.setProperty("muted", False)
        text = f"<b>{title}</b>"
        if description:
            text += f" — {description}"
        self._hint_label.setText(text)
        self._hint_label.style().unpolish(self._hint_label)
        self._hint_label.style().polish(self._hint_label)

    def _clear_hint(self) -> None:
        self._hint_label.setProperty("muted", True)
        self._hint_label.setText(_("Point at an operation for details"))
        self._hint_label.style().unpolish(self._hint_label)
        self._hint_label.style().polish(self._hint_label)

    @staticmethod
    def plot_operation() -> dict:
        return {
            "name": "NewPlotTabDialog",
            "value": "Plot",
            "description": getattr(NewPlotTabDialog, "Description", "Create a new plot"),
            "icon": NewPlotTabDialog.Icon,
            "builtin": True,
        }

    @staticmethod
    def plugin_icon(operation: dict) -> QIcon:
        if operation.get("action_icon"):
            # A catalogue action rather than a plugin with SVG artwork.
            return load_icon(str(operation["action_icon"]))
        svg_source = operation.get("icon") or operation.get("Icon") or ""
        svg_source = unescape(str(svg_source)).strip()
        if not svg_source:
            return QIcon()
        # The wrapping lives in style.icon_from_svg_source now; the accent
        # colour is this list's own, so it is passed rather than assumed.
        return icon_from_svg_source(svg_source, color=_ACCENT)
