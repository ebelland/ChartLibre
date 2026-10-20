"""The Guided analysis panel (navigation rail): the table, its quick answers, the common charts, the wizard.

For someone who knows what they want to find out but not the name of the
function that does it. Light on purpose (G-02): the table selected in the
Tables panel in one line, concrete questions of it - each opening its quick
answer with the columns already cast (:attr:`question_requested`) - the
general quick answers (:attr:`entry_requested`), and "Help me choose…",
which opens the wizard (app.dialogs.guided_wizard_dialog): columns first,
then the charts and the questions they allow, or the goals in everyday
words (:attr:`wizard_requested`). The main window opens whatever is chosen.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from app.analysis import catalog
from app.analysis.catalog import CatalogEntry
from app.logs.logger import applogger
from app.plugin_api.table_operations import table_profile
from app.plugin_api.table_operations.table_profile import TableProfile
from app.utils.i18n import _
from app.widgets.catalog_search import entry_icon
from app.widgets.nav_bar import NavPanel


#: The common charts, by the chart type and what each shows - the bars, the scatter, the distribution...
COMMON_CHARTS: tuple[tuple[str, str], ...] = (
    ("Bar Chart", "Bars: a value per category"),
    ("Scatter Plot", "Scatter: one measure against another"),
    ("Histogram", "Distribution: how the values spread"),
    ("Box Plot", "Box plot: groups compared"),
    ("Time Series", "Line over time"),
    ("Pie Chart", "Pie: shares of a whole"),
)


class GuidedAnalysisPanel(NavPanel):
    """The table in one line, its quick answers, and Help me choose."""

    #: A catalogue entry to open (app.analysis.catalog.CatalogEntry).
    entry_requested = Signal(object)
    #: A concrete question: (catalogue entry, role -> columns, column -> type).
    question_requested = Signal(object, dict, dict)
    #: Help me choose: the wizard, on the selected table (None: no table).
    wizard_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None, entries: list[CatalogEntry] | None = None) -> None:
        super().__init__(parent)
        self._entries = catalog.build_catalog() if entries is None else list(entries)
        self._profile: TableProfile | None = None
        self._table: str | None = None
        self._loader: Callable[[str], tuple[pd.DataFrame, dict[str, str]]] | None = None
        self._stale = False

        self._table_label = QLabel(_("No table selected."), self)
        self._table_label.setWordWrap(True)
        self._table_label.setProperty("muted", True)
        self.add_frame(_("Table"), self._table_label, object_name="guidedTableFrame")

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
        for entry in quick:
            button = QPushButton(entry_icon(entry), _(entry.name), box)
            button.setObjectName("quickAnswerButton")
            button.setToolTip(_(entry.description))
            button.clicked.connect(lambda _checked=False, e=entry: self.entry_requested.emit(e))
            box_layout.addWidget(button)
        self.add_frame(_("Quick answers"), box, object_name="guidedAnalysisCard")

        # The everyday charts, by what they show: each opens New chart on its type.
        by_key = {entry.key: entry for entry in self._entries}
        charts = QWidget(self)
        charts_layout = QVBoxLayout(charts)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        for chart, label in COMMON_CHARTS:
            entry = by_key.get(f"chart:{chart}")
            if entry is None:  # a chart type removed (they are plugins)
                continue
            button = QPushButton(entry_icon(entry), _(label), charts)
            button.setObjectName("quickAnswerButton")
            button.setToolTip(_(entry.description))
            button.clicked.connect(lambda _checked=False, e=entry: self.entry_requested.emit(e))
            charts_layout.addWidget(button)
        self.add_frame(_("Common charts"), charts, object_name="guidedChartsCard")

        choose = QWidget(self)
        choose_layout = QVBoxLayout(choose)
        choose_layout.setContentsMargins(0, 0, 0, 0)
        self.help_button = QPushButton(_("Help me choose…"), choose)
        self.help_button.setProperty("primary", True)
        self.help_button.setToolTip(_("Pick the columns: see the charts and the questions they allow. "
                                      "Or start from what you want to find out."))
        self.help_button.clicked.connect(lambda _checked=False: self.wizard_requested.emit(self._table))
        choose_layout.addWidget(self.help_button)
        sentence = QLabel(_("Columns first: the charts that show them and the analyses that answer "
                            "questions about them."), choose)
        sentence.setWordWrap(True)
        sentence.setProperty("muted", True)
        choose_layout.addWidget(sentence)
        self.add_frame(_("Not sure which analysis?"), choose, object_name="guidedChooseFrame")

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
            except Exception:  # noqa: BLE001 - a table that cannot be read: general answers only
                applogger.exception("Guided analysis: could not read table %s", self._table)
        self._show_table()

    def _show_table(self) -> None:
        table = self._profile
        self._table_label.setProperty("muted", table is None)
        self._table_label.style().unpolish(self._table_label)
        self._table_label.style().polish(self._table_label)
        if table is None:
            self._table_label.setText(_("No table selected."))
        else:
            counts = table.counts()
            self._table_label.setText(
                f"<b>{table.table}</b> · " + _("{rows} rows").format(rows=table.rows) + " · "
                + _("{measures} numbers, {groups} categories, {times} dates").format(**counts)
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
