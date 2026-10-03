"""The Series Operations panel: New plot, then every operation, as a list.

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
control chart in". A search field above filters the list by name and
description; Return in it opens the first match. The description of the
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.scanners.series_operation_scanner import series_operations
from app.styles.style import (
    CardFrame,
    icon_from_svg_source,
    load_icon,
    stdSizeAndlayout,
)
from app.utils.i18n import _, tr
from app.widgets.base_properties import BaseProperties

_ACCENT = "#2563EB"

#: Section name -> the operation Name strings (SeriesOperationDialogBase.Name)
#: it holds, in display order. An operation whose Name is not listed here
#: - a new plugin dropped in without this being updated - lands in
#: _FALLBACK_SECTION rather than disappearing, so being forgotten here costs
#: it a good home, not a listing at all.
#:
#: Titles are left untranslated here and passed through _() in
#: _group_by_section instead: this tuple is built once, at module import, so
#: a _() call made here would freeze the section headers in whatever
#: language was active at first import - English, since that runs before
#: the saved language preference is applied - and the app-wide language
#: switch would never reach them again.
_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Analysis", ("Peaks", "Roots", "Calculus")),
    (
        "Statistics",
        (
            "Statistics", "Outliers", "Clustering", "Control Chart",
            "Transform", "Decomposition",
        ),
    ),
    ("Geometry", ("Geometry",)),
    (
        "Signal Processing",
        ("Smoothing", "Spectral Analysis", "Filtering", "Baseline Correction"),
    ),
    ("Modeling", ("Fit", "Interpolation", "Function", "Regression", "GP Regression")),
)
_FALLBACK_SECTION = "Other"


def _operation_action_id(operation: dict) -> str:
    return str(operation.get("value") or operation.get("name") or "")


def _group_by_section(operations: list[dict]) -> list[tuple[str, list[dict]]]:
    """Sort *operations* into ``_SECTIONS``' order, dropping empty sections.

    A section with none of its operations discovered - every one of them
    failed to import, say - is left out rather than shown as an empty
    header with nothing under it.
    """
    by_name = {_operation_action_id(op): op for op in operations}
    grouped: list[tuple[str, list[dict]]] = []
    placed: set[str] = set()

    for title, names in _SECTIONS:
        items = [by_name[name] for name in names if name in by_name]
        placed.update(names)
        if items:
            grouped.append((_(title), items))

    leftover = [op for op in operations if _operation_action_id(op) not in placed]
    if leftover:
        grouped.append((_(_FALLBACK_SECTION), leftover))
    return grouped


def _operation_action_id(operation: dict) -> str:
    return str(operation.get("value") or operation.get("name") or "")


def _group_by_section(operations: list[dict]) -> list[tuple[str, list[dict]]]:
    """Sort *operations* into ``_SECTIONS``' order, dropping empty sections.

    A section with none of its operations discovered - every one of them
    failed to import, say - is left out rather than shown as an empty
    header with nothing under it.
    """
    by_name = {_operation_action_id(op): op for op in operations}
    grouped: list[tuple[str, list[dict]]] = []
    placed: set[str] = set()

    for title, names in _SECTIONS:
        items = [by_name[name] for name in names if name in by_name]
        placed.update(names)
        if items:
            grouped.append((_(title), items))

    leftover = [op for op in operations if _operation_action_id(op) not in placed]
    if leftover:
        grouped.append((_(_FALLBACK_SECTION), leftover))
    return grouped


#: Item data roles: the operation a row opens, and what the search matches.
_OPERATION_ROLE = Qt.ItemDataRole.UserRole
_SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1
_DESCRIPTION_ROLE = Qt.ItemDataRole.UserRole + 2

_ICON_SIZE = 18


class SeriesOperationWidget(BaseProperties):
    operation_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        root_layout = QVBoxLayout(self)
        stdSizeAndlayout(root_layout)

        page = CardFrame(self, "seriesOperationsPageCard", margins=(0, 0, 0, 0))
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root_layout.addWidget(page, 1)
        page_layout = page.layout()

        self._search = QLineEdit(page)
        self._search.setObjectName("seriesOperationsSearch")
        self._search.setPlaceholderText(_("Search operations"))
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter)
        self._search.returnPressed.connect(self._open_first_match)
        self._search.installEventFilter(self)
        page_layout.addWidget(self._search, 0)

        self._list = QListWidget(page)
        self._list.setObjectName("seriesOperationsList")
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
        page_layout.addWidget(self._list, 1)

        self._add_section(_("Plot"), [self.plot_operation()])
        self._add_section(_("Data"), [self.query_builder_operation()])
        for title, operations in _group_by_section(list(series_operations)):
            self._add_section(title, operations)

        # Fixed at the panel's bottom, outside the list: a description belongs
        # somewhere that stays put while it is read.
        self._hint_label = QLabel(_("Point at an operation for details"), page)
        self._hint_label.setObjectName("operationHint")
        self._hint_label.setProperty("muted", True)
        self._hint_label.setWordWrap(True)
        page_layout.addWidget(self._hint_label, 0)

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
        header.setData(_SEARCH_ROLE, None)
        for operation in operations:
            name = tr(_operation_action_id(operation))
            description = tr(str(operation.get("description") or ""))
            item = QListWidgetItem(self.plugin_icon(operation), name, self._list)
            item.setData(_OPERATION_ROLE, operation)
            item.setData(_SEARCH_ROLE, f"{name} {description}".casefold())
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

    def _filter(self, text: str) -> None:
        """Show the rows whose name or description has *text*, and their titles."""
        wanted = text.strip().casefold()
        header: QListWidgetItem | None = None
        header_used = False
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            haystack = item.data(_SEARCH_ROLE)
            if haystack is None:  # a section title
                if header is not None:
                    header.setHidden(not header_used)
                header, header_used = item, False
                continue
            shown = not wanted or wanted in str(haystack)
            item.setHidden(not shown)
            header_used = header_used or shown
        if header is not None:
            header.setHidden(not header_used)

    def _open_first_match(self) -> None:
        for item in self.operation_items():
            if not item.isHidden():
                self._open_item(item)
                return

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
        elif watched is self._search and event.type() == QEvent.Type.KeyPress:
            # Down from the search field goes into the list, at the first match.
            if getattr(event, "key", lambda: None)() == Qt.Key.Key_Down:
                visible = [item for item in self.operation_items() if not item.isHidden()]
                if visible:
                    self._list.setFocus()
                    self._list.setCurrentItem(visible[0])
                    return True
        return super().eventFilter(watched, event)

    def _reload_from_descriptor(self) -> None:
        """No-op: this panel lists operations, it does not edit a descriptor.

        Required by BaseProperties; nothing here reads from a connected
        figure, so there is nothing to reload when one changes.
        """

    def _set_enabled_state(self, enabled: bool) -> None:
        """No-op for the same reason: every row stays usable regardless.

        Whether an operation can actually run is decided where it is opened
        (main_window._open_series_operation refuses without a current
        chart), not by disabling the row that asks for one.
        """

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
    def query_builder_operation() -> dict:
        """Query Builder, listed with the operations: the main window opens it."""
        return {
            "name": "QueryBuilderDialog",
            "value": "Query Builder",
            "description": "Write, validate and save SQL queries",
            "action_icon": "query_builder",
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
