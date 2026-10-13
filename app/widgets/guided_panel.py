"""The Guided analysis panel (navigation rail): search, goals, suggestions.

For someone who knows what they want to find out but not the name of the
function that does it: a search over everything the application does (the
same as Cmd/Ctrl+K), then the goals in everyday words
(app.analysis.catalog.GOALS) with what the response is, and what fits them
(catalog.GUIDE), best first, each with why and an Open button.

Whatever is chosen - a search result or a suggestion - is emitted by
:attr:`GuidedAnalysisPanel.entry_requested`; the main window opens it.

It follows the table selected in the Tables panel (set_table): it says what
the table holds, asks concrete questions of it - each opening its quick
answer with the columns already cast (:attr:`question_requested`) - and puts
the goals and suggestions the table can feed first, the others greyed with
what they lack (app.table_operations.table_profile).
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShowEvent
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
from app.logs.logger import applogger
from app.styles.style import CardFrame, stdSizeAndlayout
from app.table_operations import table_profile
from app.table_operations.table_profile import TableProfile
from app.utils.i18n import _
from app.widgets.catalog_search import CatalogSearch, entry_icon
from app.widgets.nav_bar import NavPanel

_GOAL_ROLE = Qt.ItemDataRole.UserRole
_ROW_HEIGHT = 24


class GuidedAnalysisPanel(NavPanel):
    """Search, then goal and response type, then the suggestions."""

    #: A catalogue entry to open (app.analysis.catalog.CatalogEntry).
    entry_requested = Signal(object)
    #: A concrete question: (catalogue entry, role -> columns, column -> type).
    question_requested = Signal(object, dict, dict)

    def __init__(self, parent: QWidget | None = None, entries: list[CatalogEntry] | None = None) -> None:
        super().__init__(parent)
        self._entries = catalog.build_catalog() if entries is None else list(entries)
        self._profile: TableProfile | None = None
        self._table: str | None = None
        self._loader: Callable[[str], tuple[pd.DataFrame, dict[str, str]]] | None = None
        self._stale = False

        self._table_label = QLabel(_("No table selected: the suggestions are general."), self)
        self._table_label.setWordWrap(True)
        self._table_label.setProperty("muted", True)
        self.add_frame(_("Table"), self._table_label, object_name="guidedTableFrame")

        self.search = CatalogSearch(self, self._entries, collapsed=True)
        self.search.entry_chosen.connect(self.entry_requested)
        self.add_frame(_("Search"), self.search, object_name="guidedSearchFrame")

        # The questions a non-statistician asks, each a button that opens its quick answer.
        quick = [entry for entry in self._entries if entry.kind == catalog.ANALYSIS
                 and isinstance(entry.payload, dict) and entry.payload.get("category") == "Quick answers"]
        box = QWidget(self)
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        # The table's own questions first, then the general ones.
        self._questions = QWidget(box)
        self._questions_layout = QVBoxLayout(self._questions)
        self._questions_layout.setContentsMargins(0, 0, 0, 0)
        self._questions.hide()
        box_layout.addWidget(self._questions)
        if quick:
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
        self._fill_goals()
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

    # ------------------------------------------------------------------
    # The selected table
    # ------------------------------------------------------------------

    def set_loader(self, loader: Callable[[str], tuple[pd.DataFrame, dict[str, str]]]) -> None:
        """How to read a table: its visible rows and its columns' declared SQL types."""
        self._loader = loader

    def set_table(self, table: str | None) -> None:
        """Follow the table selected in the Tables panel; read it when the panel is seen."""
        self._table = table or None
        self._stale = True
        if self.isVisible():
            self._read_table()

    def profile(self) -> TableProfile | None:
        if self._stale:
            self._read_table()
        return self._profile

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._stale:
            self._read_table()

    def _read_table(self) -> None:
        self._stale = False
        self._profile = None
        if self._table and self._loader is not None:
            try:
                frame, declared = self._loader(self._table)
                self._profile = table_profile.profile(self._table, frame, declared)
            except Exception:  # noqa: BLE001 - a table that cannot be read: general suggestions
                applogger.exception("Guided analysis: could not read table %s", self._table)
        self._show_table()
        self._fill_goals()
        self._show()

    def _show_table(self) -> None:
        table = self._profile
        self._table_label.setProperty("muted", table is None)
        self._table_label.style().unpolish(self._table_label)
        self._table_label.style().polish(self._table_label)
        if table is None:
            self._table_label.setText(_("No table selected: the suggestions are general."))
        else:
            counts = table.counts()
            self._table_label.setText(
                f"<b>{table.table}</b> · " + _("{rows} rows").format(rows=table.rows) + "<br>"
                + _("Numeric: {measures} · Categories: {groups} · Dates: {times}").format(**counts)
            )
        while self._questions_layout.count():
            child = self._questions_layout.takeAt(0).widget()
            if child is not None:
                child.hide()
                child.deleteLater()
        found = table_profile.questions(table) if table is not None else []
        by_key = {entry.key: entry for entry in self._entries}
        for question in found:
            entry = by_key.get(question.key)
            if entry is None:
                continue
            button = QPushButton(entry_icon(entry), _(question.template).format(**question.values), self._questions)
            button.setObjectName("quickAnswerButton")
            button.setToolTip(_(entry.description))
            button.clicked.connect(
                lambda _checked=False, e=entry, q=question: self.question_requested.emit(e, q.roles, q.kinds))
            self._questions_layout.addWidget(button)
        self._questions.setVisible(bool(found))

    def _fill_goals(self) -> None:
        """The goals, those the table can answer first; the others greyed, saying what they lack."""
        current = self.goals.currentItem()
        selected = str(current.data(_GOAL_ROLE)) if current is not None else None
        blocked = self.goals.blockSignals(True)
        self.goals.clear()
        ordered = sorted(catalog.GOALS, key=lambda goal: self._goal_missing(goal[0]) is not None)
        for key, title, sentence in ordered:
            item = QListWidgetItem(_(title), self.goals)
            item.setData(_GOAL_ROLE, key)
            missing = self._goal_missing(key)
            item.setToolTip(_(sentence) + ("\n" + _(missing) if missing else ""))
            if missing:
                item.setForeground(self.palette().placeholderText())
            if key == selected:
                self.goals.setCurrentItem(item)
        self.goals.blockSignals(blocked)

    def _goal_missing(self, goal: str) -> str | None:
        return table_profile.goal_missing(self._profile, goal) if self._profile is not None else None

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
        suits = {(key, why): kind for key, why, kind in catalog.GUIDE.get(goal, ())}
        missing = [
            table_profile.suggestion_missing(self._profile, entry.key, suits.get((entry.key, why), catalog.ANY))
            if self._profile is not None else None
            for entry, why in found
        ]
        # What the table can feed first; the rest after, greyed, saying what it lacks.
        for (entry, why), lack in sorted(zip(found, missing), key=lambda pair: pair[1] is not None):
            self._list_layout.addWidget(self._card(entry, why, lack))
        if not found:
            empty = QLabel(_("Nothing for this kind of response: choose Either."))
            empty.setProperty("muted", True)
            empty.setWordWrap(True)
            self._list_layout.addWidget(empty)

    def _card(self, entry: CatalogEntry, why: str, missing: str | None = None) -> QWidget:
        card = CardFrame(None, "guidedSuggestion")
        card.setProperty("entry", entry)
        card.setProperty("missing", missing or "")
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
        if missing:
            note = QLabel(_("This table: {missing}").format(missing=_(missing)), card)
            note.setWordWrap(True)
            note.setProperty("muted", True)
            layout.addWidget(note)
            for label in (title, text):
                label.setEnabled(False)
        return card
