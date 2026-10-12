"""The Guided analysis panel (navigation rail): search, goals, suggestions.

For someone who knows what they want to find out but not the name of the
function that does it: a search over everything the application does (the
same as Cmd/Ctrl+K), then the goals in everyday words
(app.analysis.catalog.GOALS) with what the response is, and what fits them
(catalog.GUIDE), best first, each with why and an Open button.

Whatever is chosen - a search result or a suggestion - is emitted by
:attr:`GuidedAnalysisPanel.entry_requested`; the main window opens it.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.analysis import catalog
from app.analysis.catalog import CatalogEntry
from app.styles.style import CardFrame, stdSizeAndlayout
from app.utils.i18n import _
from app.widgets.catalog_search import CatalogSearch, entry_icon
from app.widgets.nav_bar import NavPanel

_GOAL_ROLE = Qt.ItemDataRole.UserRole
_ROW_HEIGHT = 24


class GuidedAnalysisPanel(NavPanel):
    """Search, then goal and response type, then the suggestions."""

    #: A catalogue entry to open (app.analysis.catalog.CatalogEntry).
    entry_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None, entries: list[CatalogEntry] | None = None) -> None:
        super().__init__(parent)
        self._entries = catalog.build_catalog() if entries is None else list(entries)

        self.search = CatalogSearch(self, self._entries, collapsed=True)
        self.search.entry_chosen.connect(self.entry_requested)
        self.add_frame(_("Search"), self.search, object_name="guidedSearchFrame")

        # The questions a non-statistician asks, each a button that opens its quick answer.
        quick = [entry for entry in self._entries if entry.kind == catalog.ANALYSIS
                 and isinstance(entry.payload, dict) and entry.payload.get("category") == "Quick answers"]
        if quick:
            box = QWidget(self)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(0, 0, 0, 0)
            for entry in quick:
                button = QPushButton(entry_icon(entry), _(entry.name), box)
                button.setObjectName("quickAnswerButton")
                button.setToolTip(_(entry.description))
                button.clicked.connect(lambda _checked=False, e=entry: self.entry_requested.emit(e))
                box_layout.addWidget(button)
            self.add_frame(_("Quick answers"), box, object_name="guidedAnalysisCard")

        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        self.goals = QListWidget(body)
        self.goals.setObjectName("guidedGoals")
        for key, title, sentence in catalog.GOALS:
            item = QListWidgetItem(_(title), self.goals)
            item.setData(_GOAL_ROLE, key)
            item.setToolTip(_(sentence))
        # Every goal in sight: twelve short rows, no scrolling to find one.
        self.goals.setFixedHeight(len(catalog.GOALS) * _ROW_HEIGHT + 8)
        self.goals.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.goals.currentItemChanged.connect(lambda *_args: self._show())
        layout.addWidget(self.goals)
        form = QFormLayout()
        stdSizeAndlayout(form)
        self._response = QComboBox(body)
        for value, label in ((catalog.ANY, "Either"), (catalog.NUMERIC, "A number"),
                             (catalog.CATEGORICAL, "A category")):
            self._response.addItem(_(label), value)
        self._response.currentIndexChanged.connect(lambda _index: self._show())
        form.addRow(_("The response (Y) is"), self._response)
        layout.addLayout(form)
        self.add_frame(_("What do you want to do?"), body, object_name="guidedGoalsFrame")

        page = QWidget(self)
        self._list_layout = QVBoxLayout(page)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._goal_sentence = QLabel(page)
        self._goal_sentence.setWordWrap(True)
        self._goal_sentence.setProperty("muted", True)
        self._list_layout.addWidget(self._goal_sentence)
        self.add_frame(_("Suggestions"), page, object_name="guidedSuggestionsFrame")
        self.goals.setCurrentRow(0)

    def response(self) -> str:
        return str(self._response.currentData() or catalog.ANY)

    def shown(self) -> list[CatalogEntry]:
        """The suggestions listed now, in order."""
        return [card.property("entry") for card in self._cards()]

    def _cards(self) -> list[QWidget]:
        found = []
        for index in range(self._list_layout.count()):
            widget = self._list_layout.itemAt(index).widget()
            if widget is not None and widget.property("entry") is not None:
                found.append(widget)
        return found

    def _show(self) -> None:
        item = self.goals.currentItem()
        goal = str(item.data(_GOAL_ROLE)) if item is not None else ""
        sentences = {key: sentence for key, _title, sentence in catalog.GOALS}
        self._goal_sentence.setText(_(sentences.get(goal, "")))
        while self._list_layout.count() > 1:
            child = self._list_layout.takeAt(1).widget()
            if child is not None:
                # Hidden now: deleteLater waits for the event loop, and would
                # leave the last goal's cards painted under the new ones.
                child.hide()
                child.deleteLater()
        found = catalog.suggestions(self._entries, goal, self.response())
        for entry, why in found:
            self._list_layout.addWidget(self._card(entry, why))
        if not found:
            empty = QLabel(_("Nothing for this kind of response: choose Either."))
            empty.setProperty("muted", True)
            empty.setWordWrap(True)
            self._list_layout.addWidget(empty)

    def _card(self, entry: CatalogEntry, why: str) -> QWidget:
        card = CardFrame(None, "guidedSuggestion")
        card.setProperty("entry", entry)
        layout = card.layout()
        top = QHBoxLayout()
        icon = QLabel(card)
        icon.setPixmap(entry_icon(entry).pixmap(18, 18))
        top.addWidget(icon)
        title = QLabel(f"<b>{_(entry.name)}</b><br><span style='color:gray'>{_(catalog.KIND_LABELS[entry.kind])}"
                       "</span>", card)
        top.addWidget(title, 1)
        button = QPushButton(_("Open"), card)
        button.clicked.connect(lambda _checked=False, e=entry: self.entry_requested.emit(e))
        top.addWidget(button)
        layout.addLayout(top)
        text = QLabel(_(why), card)
        text.setWordWrap(True)
        layout.addWidget(text)
        return card
