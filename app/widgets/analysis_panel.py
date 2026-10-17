"""The Analysis panel: every analysis and data operation, a titled frame per category, as JMP's Analyze menu.

One panel for what were two: the data operations (on a chart's series or a
table's columns) and the table analyses (Distribution, Fit Y by X, Fit
Model...), with New chart and the design of experiments. Each plugin names
its own category and its place (Category, Order on the plugin class), so the
panel knows no operation: a plugin file added or removed changes the panel
and nothing else. The categories follow JMP's where there is one - Basic
Analysis, Fitting Linear Models, Predictive and Specialized Modeling,
Multivariate Methods, Screening, Quality and Process, Design of Experiments -
and are ordered by their lowest Order. The quick answers are the Guided
analysis panel's.

Each frame is a short list in the manner of a sidebar: the operation's icon
and name, the row under the pointer tinted, a click (or Return) opening it,
its sentence as the tooltip.
"""
from __future__ import annotations

from html import unescape

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QAbstractItemView, QFrame, QListWidget, QListWidgetItem, QSizePolicy, QWidget

from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.scanners.data_operation_scanner import data_operations
from app.scanners.table_operation_scanner import table_operations
from app.styles.style import icon_from_svg_source, load_icon
from app.utils.i18n import _, tr
from app.widgets.nav_bar import NavPanel

_ACCENT = "#2563EB"

#: The kinds of entry: what a row opens.
DATA, TABLE, TOOL = "data", "table", "tool"
#: The category of an entry that declares none. Translated where it is shown.
_FALLBACK_SECTION = "Other"
#: Item data roles: the entry a row opens.
_ENTRY_ROLE = Qt.ItemDataRole.UserRole
_ICON_SIZE = 18
_ROW_HEIGHT = 28

#: The panel's own entries, placed among the plugins by the same Category and Order.
_PLOT_CATEGORY, _PLOT_ORDER = "Graph", 0
_DOE_CATEGORY, _DOE_ORDER = "Design of Experiments", 100


def entry_name(entry: dict) -> str:
    return str(entry.get("value") or entry.get("name") or "")


def _order(entry: dict) -> int:
    value = entry.get("order")
    return value if isinstance(value, int) else 0


def group_by_category(entries: list[dict]) -> list[tuple[str, list[dict]]]:
    """(category, entries) in the order of each category's lowest Order; uncategorised last."""
    sections: dict[str, list[dict]] = {}
    for entry in entries:
        sections.setdefault(str(entry.get("category") or ""), []).append(entry)
    known = sorted(
        ((title, sorted(items, key=lambda e: (_order(e), entry_name(e)))) for title, items in sections.items() if title),
        key=lambda section: (min(_order(e) for e in section[1]), section[0]),
    )
    if "" in sections:
        known.append((_FALLBACK_SECTION, sections[""]))
    return known


def plot_operation() -> dict:
    """New chart, as an entry like the operations'."""
    return {
        "value": "Plot", "name": NewPlotTabDialog.__name__, "kind": DATA, "builtin": True,
        "category": _PLOT_CATEGORY, "order": _PLOT_ORDER, "icon": NewPlotTabDialog.Icon,
        "description": unescape(str(getattr(NewPlotTabDialog, "Description", "Create a new plot"))),
    }


def doe_tool() -> dict:
    return {
        "value": "Create a table from a DOE", "kind": TOOL, "tool": "doe",
        # Drawn like the plugins' icons, in the panel's accent: a grid of runs.
        "icon": '<rect x="4" y="4" width="16" height="16" rx="2"/><circle cx="8" cy="8" r="1.4"/>'
                '<circle cx="16" cy="8" r="1.4"/><circle cx="8" cy="16" r="1.4"/><circle cx="16" cy="16" r="1.4"/>'
                '<circle cx="12" cy="12" r="1.4"/>',
        "category": _DOE_CATEGORY, "order": _DOE_ORDER,
        "description": "Generate an experiment matrix with factors and responses; analyse it with Fit Model.",
    }


def plugin_icon(entry: dict) -> QIcon:
    """The entry's own SVG icon (in the panel's accent), or a catalogue action's."""
    if entry.get("action_icon"):
        return load_icon(str(entry["action_icon"]))
    source = unescape(str(entry.get("icon") or entry.get("Icon") or "")).strip()
    if not source:
        return QIcon()
    return icon_from_svg_source(source, color=_ACCENT)


def panel_entries() -> list[dict]:
    """Everything the panel lists: New chart, the data operations, the table analyses, the DOE."""
    entries = [plot_operation()]
    entries += [{**operation, "kind": DATA} for operation in data_operations]
    entries += [{**operation, "kind": TABLE} for operation in table_operations
                if operation.get("category") != "Quick answers"]
    entries.append(doe_tool())
    return entries


class AnalysisPanel(NavPanel):
    """A frame per category; a row per analysis or operation."""

    #: A data operation's discovery entry (New chart included), to open.
    data_operation_requested = Signal(dict)
    #: A table analysis' discovery entry, to open on the selected table.
    table_operation_requested = Signal(dict)
    #: A tool's key ("doe").
    tool_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None, entries: list[dict] | None = None) -> None:
        super().__init__(parent)
        self._lists: list[QListWidget] = []
        for category, items in group_by_category(panel_entries() if entries is None else entries):
            view = self._section_list(items)
            object_name = "analysis" + "".join(word.capitalize() for word in category.split()) + "Frame"
            self.add_frame(_(category), view, object_name=object_name)

    def _section_list(self, entries: list[dict]) -> QListWidget:
        view = QListWidget(self)
        # The sidebar look the Data operations list had (the sheets' #dataOperationsList).
        view.setObjectName("dataOperationsList")
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setTextElideMode(Qt.TextElideMode.ElideRight)
        view.setMouseTracking(True)
        for entry in entries:
            name = tr(entry_name(entry))
            item = QListWidgetItem(plugin_icon(entry), name, view)
            item.setData(_ENTRY_ROLE, entry)
            item.setToolTip(tr(str(entry.get("description") or "")))
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
        # As tall as its rows: the panel scrolls, not each frame.
        view.setFixedHeight(len(entries) * _ROW_HEIGHT + 4)
        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # A click opens - and only a click: itemActivated would open twice on a double-click.
        view.itemClicked.connect(self._open_item)
        view.installEventFilter(self)
        self._lists.append(view)
        return view

    # ------------------------------------------------------------------
    def items(self) -> list[QListWidgetItem]:
        """Every row, frame by frame."""
        return [view.item(i) for view in self._lists for i in range(view.count())]

    def categories(self) -> list[str]:
        """The frames' titles, in order."""
        return [self.frame_title(index) for index in range(len(self._lists))]

    def frame_title(self, index: int) -> str:
        frame = self._lists[index].parentWidget()
        while frame is not None and not hasattr(frame, "title_label"):
            frame = frame.parentWidget()
        return frame.title_label.text() if frame is not None else ""

    def _open_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        entry = item.data(_ENTRY_ROLE)
        item.listWidget().clearSelection()
        if not isinstance(entry, dict):
            return
        if entry.get("kind") == TABLE:
            self.table_operation_requested.emit(entry)
        elif entry.get("kind") == TOOL:
            self.tool_requested.emit(str(entry.get("tool")))
        else:
            self.data_operation_requested.emit(entry)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (isinstance(watched, QListWidget) and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):  # type: ignore[attr-defined]
            self._open_item(watched.currentItem())
            return True
        return super().eventFilter(watched, event)
