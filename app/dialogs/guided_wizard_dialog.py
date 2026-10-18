"""The guided analysis wizard: pick columns, then see them and ask of them (G-02).

Columns first, for someone who knows their data but not the name of the
function they need:

1. *Which columns?* - the table's columns, each with its type, and the
   first rows of those picked (one to three).
2. *What you can do with them* - "See it": charts sketched from the rows,
   each made at once with Create chart; "Ask a question": the analyses that
   answer a question phrased with the columns' names, opened with them cast.
   Only what the columns allow is offered (guided_plan). When two measures
   could be the result, which one is asked here - nowhere else.

"Not sure where to start?" leads to the goals in everyday words and what
fits each (catalog.GUIDE), as the rail's panel used to show. The columns
picked last are remembered per table.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.analysis import catalog
from app.analysis.catalog import CatalogEntry
from app.dialogs.two_panels_dialog_base import PanelDialog
from app.plugin_api.table_operations import guided_plan, table_profile
from app.plugin_api.table_operations.column_roles import CONTINUOUS, NOMINAL, ORDINAL
from app.plugin_api.table_operations.table_profile import TableProfile
from app.styles.style import CardFrame, TitledCard, load_icon, mark_editor_panel
from app.utils.config import get_section, set_section
from app.utils.i18n import _
from app.widgets.catalog_search import entry_icon
from app.widgets.guided_thumbnails import thumbnail
from app.widgets.table_preview import DataFrameTableModel

#: Where the columns picked last are kept, per table.
STATE_SECTION = "guided_wizard"
#: The most columns picked at once.
MAX_COLUMNS = 3
#: The rows shown under the columns.
PREVIEW_ROWS = 8

_COLUMNS, _IDEAS, _GOALS = range(3)
_NAME_ROLE = Qt.ItemDataRole.UserRole


def kind_mark(profile: TableProfile, column: str) -> str:
    """The column's type as JMP marks it: ▲ continuous, ● nominal, ◐ ordinal - and a clock for a date."""
    if column in profile.times:
        return "◷"
    return {CONTINUOUS: "▲", NOMINAL: "●", ORDINAL: "◐"}.get(profile.kinds.get(column, ""), "·")


class GuidedWizardDialog(PanelDialog):
    """Pick columns, then a chart to see them or a question to ask of them."""

    #: Make this chart: (table, guided_plan.ChartIdea).
    chart_requested = Signal(str, object)
    #: Open this analysis with the columns cast: (catalogue entry, role -> columns, column -> type).
    question_requested = Signal(object, dict, dict)
    #: Open this catalogue entry as it is (a goal's suggestion).
    entry_requested = Signal(object)

    def __init__(self, table: str, frame: pd.DataFrame, profile: TableProfile, *,
                 entries: list[CatalogEntry] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent, title=_("Help me choose"), icon=load_icon("nav_guided"), size=None)
        self.resize(QSize(820, 560))
        self._table = table
        self._frame = frame
        self._profile = profile
        self._entries = catalog.build_catalog() if entries is None else list(entries)
        self._by_key = {entry.key: entry for entry in self._entries}

        self.pages = QStackedWidget(self)
        self.body_layout.addWidget(self.pages, 1)
        self.pages.addWidget(self._build_columns_page())
        self.pages.addWidget(self._build_ideas_page())
        self.pages.addWidget(self._build_goals_page())

        self._back = self.add_action("back", self._go_back, left=True)
        self._next = self.add_action("next", self._go_next, default=True)
        self.add_action("close", self.reject)
        self._restore_columns()
        self._show_page(_COLUMNS)

    # ------------------------------------------------------------------
    # 1. Which columns?
    # ------------------------------------------------------------------
    def _build_columns_page(self) -> QWidget:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        left = TitledCard(page, _("Which columns?"), "wizardColumnsCard")
        intro = QLabel(_("Pick one to three columns of “{table}”.").format(table=self._table), left.card)
        intro.setWordWrap(True)
        left.card.layout().addWidget(intro)
        self.columns = QListWidget(left.card)
        self.columns.setObjectName("wizardColumns")
        mark_editor_panel(self.columns)
        for name in self._profile.kinds:
            item = QListWidgetItem(f"{kind_mark(self._profile, name)}  {name}", self.columns)
            item.setData(_NAME_ROLE, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.columns.itemChanged.connect(self._on_column_checked)
        left.card.layout().addWidget(self.columns, 1)
        legend = QLabel(_("▲ number · ● category · ◐ ordered category · ◷ date"), left.card)
        legend.setProperty("muted", True)
        left.card.layout().addWidget(legend)
        unsure = QLabel(f'<a href="goals">{_("Not sure where to start? Start from what you want to find out")}</a>',
                        left.card)
        unsure.setWordWrap(True)
        unsure.linkActivated.connect(lambda _link: self._show_page(_GOALS))
        left.card.layout().addWidget(unsure)
        layout.addWidget(left, 2)

        right = TitledCard(page, _("First rows"), "wizardPreviewCard")
        self.preview = QTableView(right.card)
        mark_editor_panel(self.preview)
        self.preview.verticalHeader().setVisible(False)
        right.card.layout().addWidget(self.preview, 1)
        layout.addWidget(right, 3)
        right.layout().setStretchFactor(right.card, 1)
        left.layout().setStretchFactor(left.card, 1)
        return page

    def chosen(self) -> list[str]:
        """The columns picked, in the order of the table."""
        return [str(self.columns.item(i).data(_NAME_ROLE)) for i in range(self.columns.count())
                if self.columns.item(i).checkState() == Qt.CheckState.Checked]

    def choose(self, columns: list[str]) -> None:
        """Pick exactly *columns* (those of the table, up to MAX_COLUMNS)."""
        wanted = set(columns[:MAX_COLUMNS])
        blocked = self.columns.blockSignals(True)
        for i in range(self.columns.count()):
            item = self.columns.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(_NAME_ROLE) in wanted else Qt.CheckState.Unchecked)
        self.columns.blockSignals(blocked)
        self._on_columns_changed()

    def _on_column_checked(self, item: QListWidgetItem) -> None:
        if item.checkState() == Qt.CheckState.Checked and len(self.chosen()) > MAX_COLUMNS:
            blocked = self.columns.blockSignals(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.columns.blockSignals(blocked)
        self._on_columns_changed()

    def _on_columns_changed(self) -> None:
        columns = self.chosen()
        shown = self._frame[columns] if columns else self._frame
        self.preview.setModel(DataFrameTableModel(shown.head(PREVIEW_ROWS), parent=self.preview))
        self.preview.resizeColumnsToContents()
        self._next.setEnabled(bool(columns))

    def _restore_columns(self) -> None:
        remembered = get_section(STATE_SECTION).get(self._table)
        if isinstance(remembered, list):
            self.choose([str(c) for c in remembered if str(c) in self._profile.kinds])
        else:
            self._on_columns_changed()

    def _remember_columns(self) -> None:
        state = dict(get_section(STATE_SECTION))
        state[self._table] = self.chosen()
        set_section(STATE_SECTION, state)

    # ------------------------------------------------------------------
    # 2. What you can do with them
    # ------------------------------------------------------------------
    def _build_ideas_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heading = QLabel(page)
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        row = QHBoxLayout()
        self._result_label = QLabel(_("Which one is the result?"), page)
        self.result = QComboBox(page)
        self.result.setToolTip(_("The column the others might explain or predict."))
        self.result.currentIndexChanged.connect(lambda _index: self._show_ideas())
        row.addWidget(self._result_label)
        row.addWidget(self.result)
        row.addStretch(1)
        layout.addLayout(row)

        see = TitledCard(page, _("See it"), "wizardChartsCard")
        self._charts_row = QHBoxLayout()
        see.card.layout().addLayout(self._charts_row)
        layout.addWidget(see)
        ask = TitledCard(page, _("Ask a question"), "wizardQuestionsCard")
        self._questions_box = QVBoxLayout()
        ask.card.layout().addLayout(self._questions_box)
        ask.card.layout().addStretch(1)
        layout.addWidget(ask, 1)
        ask.layout().setStretchFactor(ask.card, 1)
        return page

    def plan(self) -> guided_plan.Plan:
        return guided_plan.plan(self._profile, self.chosen(), self.result.currentData())

    def _fill_results(self) -> None:
        results = guided_plan.plan(self._profile, self.chosen()).results
        blocked = self.result.blockSignals(True)
        self.result.clear()
        for name in results:
            self.result.addItem(name, name)
        if results:
            self.result.setCurrentIndex(len(results) - 1)
        self.result.blockSignals(blocked)
        self.result.setVisible(bool(results))
        self._result_label.setVisible(bool(results))

    def _show_ideas(self) -> None:
        plan = self.plan()
        names = ", ".join(f"“{c}”" for c in self.chosen())
        self.heading.setText(f"<b>{_('What you can do with {names}').format(names=names)}</b>")
        for box in (self._charts_row, self._questions_box):
            while box.count():
                widget = box.takeAt(0).widget()
                if widget is not None:
                    widget.hide()
                    widget.deleteLater()
        for idea in plan.charts:
            self._charts_row.addWidget(self._chart_card(idea))
        self._charts_row.addStretch(1)
        for question in plan.questions:
            entry = self._by_key.get(question.key)
            if entry is not None:
                self._questions_box.addWidget(self._question_card(entry, question))
        if not plan.charts and not plan.questions:
            note = QLabel(_("Nothing to offer for these columns alone: add a number to them."))
            note.setProperty("muted", True)
            self._questions_box.addWidget(note)

    def _chart_card(self, idea: guided_plan.ChartIdea) -> QWidget:
        card = CardFrame(None, "wizardChart")
        card.setProperty("chart_type", idea.chart_type)
        card.setFixedWidth(200)
        picture = QLabel(card)
        picture.setPixmap(thumbnail(idea, self._frame))
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card.layout().addWidget(picture)
        title = QLabel(f"<b>{_(idea.chart_type)}</b><br>{_(idea.title).format(**idea.values)}", card)
        title.setWordWrap(True)
        card.layout().addWidget(title, 1)
        button = QPushButton(_("Create chart"), card)
        button.clicked.connect(lambda _checked=False, i=idea: self._finish(self.chart_requested, self._table, i))
        card.layout().addWidget(button)
        return card

    def _question_card(self, entry: CatalogEntry, question: guided_plan.QuestionIdea) -> QWidget:
        card = CardFrame(None, "wizardQuestion")
        card.setProperty("entry", entry)
        layout = QHBoxLayout()
        icon = QLabel(card)
        icon.setPixmap(entry_icon(entry).pixmap(20, 20))
        layout.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        text = QLabel(f"<b>{_(question.question).format(**question.values)}</b><br>"
                      f"<span style='color:gray'>{_(entry.name)} · {_(question.why)}</span>", card)
        text.setWordWrap(True)
        layout.addWidget(text, 1)
        button = QPushButton(_("Open"), card)
        button.clicked.connect(lambda _checked=False, e=entry, q=question:
                               self._finish(self.question_requested, e, q.roles, q.kinds))
        layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
        card.layout().addLayout(layout)
        return card

    # ------------------------------------------------------------------
    # Not sure where to start: the goals
    # ------------------------------------------------------------------
    def _build_goals_page(self) -> QWidget:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        left = TitledCard(page, _("What do you want to find out?"), "wizardGoalsCard")
        self.goals = QListWidget(left.card)
        self.goals.setObjectName("guidedGoals")
        self.goals.setSpacing(3)
        for key, title, sentence in sorted(
            catalog.GOALS, key=lambda goal: table_profile.goal_missing(self._profile, goal[0]) is not None
        ):
            item = QListWidgetItem(_(title), self.goals)
            item.setData(_NAME_ROLE, key)
            missing = table_profile.goal_missing(self._profile, key)
            item.setToolTip(_(sentence) + ("\n" + _(missing) if missing else ""))
            if missing:
                item.setForeground(self.palette().placeholderText())
        self.goals.currentItemChanged.connect(lambda *_args: self._show_suggestions())
        left.card.layout().addWidget(self.goals, 1)
        layout.addWidget(left, 2)
        left.layout().setStretchFactor(left.card, 1)

        right = TitledCard(page, _("Suggestions"), "wizardSuggestionsCard")
        scroll = QScrollArea(right.card)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget(scroll)
        self._suggestions = QVBoxLayout(holder)
        self._suggestions.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(holder)
        right.card.layout().addWidget(scroll, 1)
        layout.addWidget(right, 3)
        right.layout().setStretchFactor(right.card, 1)
        self.goals.setCurrentRow(0)
        return page

    def _show_suggestions(self) -> None:
        while self._suggestions.count():
            widget = self._suggestions.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        item = self.goals.currentItem()
        goal = str(item.data(_NAME_ROLE)) if item is not None else ""
        sentences = {key: sentence for key, _title, sentence in catalog.GOALS}
        sentence = QLabel(_(sentences.get(goal, "")))
        sentence.setProperty("muted", True)
        sentence.setWordWrap(True)
        self._suggestions.addWidget(sentence)
        suits = {(key, why): kind for key, why, kind in catalog.GUIDE.get(goal, ())}
        found = catalog.suggestions(self._entries, goal)
        missing = [table_profile.suggestion_missing(self._profile, entry.key, suits.get((entry.key, why), catalog.ANY))
                   for entry, why in found]
        for (entry, why), lack in sorted(zip(found, missing), key=lambda pair: pair[1] is not None):
            self._suggestions.addWidget(self._suggestion_card(entry, why, lack))
        self._suggestions.addStretch(1)

    def _suggestion_card(self, entry: CatalogEntry, why: str, missing: str | None) -> QWidget:
        card = CardFrame(None, "guidedSuggestion")
        card.setProperty("entry", entry)
        row = QHBoxLayout()
        icon = QLabel(card)
        icon.setPixmap(entry_icon(entry).pixmap(18, 18))
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        text = QLabel(f"<b>{_(entry.name)}</b> · <span style='color:gray'>{_(catalog.KIND_LABELS[entry.kind])}"
                      f"</span><br>{_(why)}" + (f"<br><i>{_(missing)}</i>" if missing else ""), card)
        text.setWordWrap(True)
        text.setEnabled(not missing)
        row.addWidget(text, 1)
        button = QPushButton(_("Open"), card)
        button.clicked.connect(lambda _checked=False, e=entry: self._finish(self.entry_requested, e))
        row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
        card.layout().addLayout(row)
        return card

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _show_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self._back.setVisible(index != _COLUMNS)
        self._next.setVisible(index == _COLUMNS)

    def _go_next(self) -> None:
        if not self.chosen():
            return
        self._remember_columns()
        self._fill_results()
        self._show_ideas()
        self._show_page(_IDEAS)

    def _go_back(self) -> None:
        self._show_page(_COLUMNS)

    def _finish(self, signal: Any, *args: Any) -> None:
        """Hand the choice to the main window, and close."""
        self._remember_columns()
        self.accept()
        signal.emit(*args)
