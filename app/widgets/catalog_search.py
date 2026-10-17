"""Search everything the application can do: the rail's Search panel, and Cmd/Ctrl+K.

A field and the matches under it - analyses, data operations, chart types,
table tools (app.analysis.catalog) - each with its kind, the pointed one's
sentence below. Return opens the best match (or the one the arrows chose);
choosing emits :attr:`CatalogSearch.entry_chosen` and the window opens it.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon
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

from app.analysis import catalog
from app.analysis.catalog import CatalogEntry
from app.styles.style import icon_from_svg_source, load_icon
from app.utils.i18n import tr
from app.utils.config import get_constant

_ENTRY_ROLE = Qt.ItemDataRole.UserRole
_ICON_SIZE = 18
#: Matches listed at most: past these the words typed are too few to choose by.
_LIMIT = get_constant("search_results_limit", 12)

#: The table tools' icons, by catalogue key.
_TOOL_ICONS = {
    "doe": "new", "import": "import", "import_clipboard": "paste", "import_web": "web_sources", "import_database": "import", "query": "query",
    "new_table": "new", "edit_table": "document-edit",
}


def entry_icon(entry: CatalogEntry) -> QIcon:
    """The icon the entry has where it lives: its own SVG, the chart's, the tool's."""
    if entry.kind in (catalog.ANALYSIS, catalog.SERIES) and isinstance(entry.payload, dict):
        source = str(entry.payload.get("icon") or "")
        if source:
            return icon_from_svg_source(source, size=_ICON_SIZE)
        return load_icon("new_plot")
    if entry.kind == catalog.CHART:
        return load_icon("new_plot")
    return load_icon(_TOOL_ICONS.get(str(entry.payload), "table"))


class CatalogSearch(QWidget):
    """A search field and its matches; *collapsed* hides the list while the field is empty."""

    entry_chosen = Signal(object)
    #: Escape in the field (the palette closes on it).
    dismissed = Signal()

    def __init__(self, parent: QWidget | None = None, entries: list[CatalogEntry] | None = None, *,
                 collapsed: bool = False, compact: bool = False) -> None:
        super().__init__(parent)
        #: For the narrow rail: names only, no hint line, the field cleared once something is chosen.
        self._compact = compact
        if compact:
            # In the rail it is only as tall as what it shows.
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._entries = catalog.build_catalog() if entries is None else list(entries)
        self._collapsed = collapsed
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.field = QLineEdit(self)
        self.field.setObjectName("catalogSearchField")
        self.field.setPlaceholderText(tr("Search…") if compact else tr("Search analyses, operations, charts…"))
        self.field.setClearButtonEnabled(True)
        self.field.addAction(load_icon("search_functions"), QLineEdit.ActionPosition.LeadingPosition)
        self.field.textChanged.connect(self._update)
        self.field.returnPressed.connect(self._open_current)
        self.field.installEventFilter(self)
        layout.addWidget(self.field)

        self.results = QListWidget(self)
        self.results.setObjectName("catalogSearchResults")
        self.results.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self.results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results.setMouseTracking(True)
        self.results.itemClicked.connect(self._open_item)
        self.results.itemEntered.connect(self._describe)
        self.results.currentItemChanged.connect(lambda current, _previous: self._describe(current))
        self.results.installEventFilter(self)
        layout.addWidget(self.results, 1)

        self.hint = QLabel(self)
        self.hint.setObjectName("catalogSearchHint")
        self.hint.setProperty("muted", True)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self._update("")

    # ------------------------------------------------------------------

    def matches(self) -> list[CatalogEntry]:
        """The entries listed, in order."""
        return [self.results.item(row).data(_ENTRY_ROLE) for row in range(self.results.count())]

    def _update(self, text: str) -> None:
        self.results.clear()
        found = catalog.search(self._entries, text, tr, limit=_LIMIT)
        for entry in found:
            kind = tr(catalog.KIND_LABELS[entry.kind])
            label = tr(entry.name) if self._compact else f"{tr(entry.name)}  ·  {kind}"
            item = QListWidgetItem(entry_icon(entry), label, self.results)
            item.setData(_ENTRY_ROLE, entry)
            item.setToolTip(f"{kind}: {tr(entry.description)}" if self._compact else tr(entry.description))
            item.setSizeHint(QSize(0, 28))
        if found:
            self.results.setCurrentRow(0)
        typed = bool(text.strip())
        self.results.setVisible(bool(found) or not self._collapsed)
        self.hint.setVisible(not self._compact and (typed or not self._collapsed))
        if typed and not found:
            self.hint.setText(tr("Nothing found. Try another word - in English or Italian."))
        elif not typed:
            self.hint.setText(tr("Type what you want to do: \"anova\", \"outliers\", \"peaks\", \"tree\"…"))
        if self._collapsed and not found:
            self.results.setFixedHeight(0)
        if self._collapsed and found:
            # As tall as its rows, up to the limit: the frames below move down, not under it.
            self.results.setFixedHeight(min(len(found), 8) * 28 + 6)
        self.updateGeometry()

    def _describe(self, item: QListWidgetItem | None) -> None:
        entry = item.data(_ENTRY_ROLE) if item is not None else None
        if entry is not None:
            self.hint.setText(f"<b>{tr(entry.name)}</b> — {tr(entry.description)}")

    def _open_item(self, item: QListWidgetItem | None) -> None:
        entry = item.data(_ENTRY_ROLE) if item is not None else None
        if entry is not None:
            self.entry_chosen.emit(entry)
            if self._compact:
                self.field.clear()

    def _open_current(self) -> None:
        self._open_item(self.results.currentItem() or self.results.item(0))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if watched is self.field and key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.results.count():
                # The arrows choose among the matches while the typing stays in the field.
                row = self.results.currentRow() + (1 if key == Qt.Key.Key_Down else -1)
                self.results.setCurrentRow(max(0, min(row, self.results.count() - 1)))
                return True
            if key == Qt.Key.Key_Escape:
                if self.field.text():
                    self.field.clear()
                else:
                    self.dismissed.emit()
                return True
            if watched is self.results and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._open_current()
                return True
        return super().eventFilter(watched, event)
