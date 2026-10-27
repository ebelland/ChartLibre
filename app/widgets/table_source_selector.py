"""A table, its X column and its Y columns: a data operation's data straight from a table.

The other half of the Axis / Series page (see ``DataOperationDialogBase``):
instead of the series a chart already draws, the columns of a table. The
dialog draws them first, as a chart of their own, and then runs on it like on
any other - so every operation reads a table without knowing it does.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.styles.style import apply_fusion_for_item_view_styling, mark_editor_panel, stdSizeAndlayout
from app.utils.i18n import _

#: The columns the application keeps on every table, never data.
_APPLICATION_COLUMNS = frozenset({"Hide", "Selected"})
#: X as the row's position, for a table with no column to plot against.
ROW_NUMBER = ""
#: The charts of one variable: y is drawn as their value (Box Plot, Histogram, Violin Plot).
VALUE_CHARTS = frozenset({"Box Plot", "Histogram", "Violin Plot"})


class TableSourceSelector(QWidget):
    """Pick one table, one X column and the Y columns to run on."""

    #: The table or a column changed.
    changed = Signal()

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None,
                 charts: tuple[str, ...] = ("Scatter Plot",)) -> None:
        super().__init__(parent)
        self._repo = repo
        #: The chart the columns are drawn as; offered when there is a choice.
        self.chart_combo = QComboBox(self)
        for chart in charts or ("Scatter Plot",):
            self.chart_combo.addItem(_(chart), chart)
        self.chart_combo.setToolTip(_("What the columns are drawn as, before the operation runs on them."))

        self.table_combo = QComboBox(self)
        self.x_combo = QComboBox(self)
        self.x_combo.setToolTip(_("The column the operation reads as x; the row number when there is none."))
        for combo in (self.table_combo, self.x_combo):
            # As wide as the page: a table's name is the one thing to read here.
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.y_list = QListWidget(self)
        self.y_list.setToolTip(_("The numeric columns to run on, one series each."))
        mark_editor_panel(self.y_list)
        apply_fusion_for_item_view_styling(self.y_list)
        # About eight columns' worth; more scroll.
        self.y_list.setMaximumHeight(8 * max(self.y_list.fontMetrics().height() + 6, 20) + 8)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for row, (text, widget) in enumerate(((_("Table:"), self.table_combo), (_("X:"), self.x_combo))):
            grid.addWidget(QLabel(text, self), row, 0)
            grid.addWidget(widget, row, 1)
        grid.addWidget(QLabel(_("Y:"), self), 2, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self.y_list, 2, 1)
        if self.chart_combo.count() > 1:
            grid.addWidget(QLabel(_("Chart:"), self), 3, 0)
            grid.addWidget(self.chart_combo, 3, 1)
        else:
            self.chart_combo.hide()
        grid.setColumnStretch(1, 1)
        note = QLabel(_("The columns are drawn as a chart of their own, which the operation then runs on."), self)
        note.setProperty("muted", True)
        note.setWordWrap(True)

        layout = QVBoxLayout(self)
        stdSizeAndlayout(layout)
        layout.addLayout(grid)
        layout.addWidget(note)
        layout.addStretch(1)

        self.table_combo.currentIndexChanged.connect(lambda _index: self._load_columns())
        self.x_combo.currentIndexChanged.connect(lambda _index: self.changed.emit())
        self.y_list.itemChanged.connect(lambda _item: self.changed.emit())
        self.chart_combo.currentIndexChanged.connect(lambda _index: self.changed.emit())
        self.reload()

    def reload(self) -> None:
        """List the project's tables again, keeping the one chosen."""
        current = self.table_combo.currentText()
        self.table_combo.blockSignals(True)
        self.table_combo.clear()
        self.table_combo.addItems([str(name) for name in self._repo.list_user_tables()["Table"]])
        index = self.table_combo.findText(current)
        self.table_combo.setCurrentIndex(max(index, 0))
        self.table_combo.blockSignals(False)
        self._load_columns()

    def _load_columns(self) -> None:
        table = self.table_combo.currentText()
        sample = pd.DataFrame()
        if table:
            quoted = '"' + table.replace('"', '""') + '"'
            sample = self._repo.query_df(f"SELECT * FROM {quoted} LIMIT 200")
        columns = [str(c) for c in sample.columns if str(c) not in _APPLICATION_COLUMNS]
        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(sample[c])]

        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItem(_("(row number)"), ROW_NUMBER)
        for column in columns:
            self.x_combo.addItem(column, column)
        # The first numeric column is the usual x: a time, a position.
        self.x_combo.setCurrentIndex(1 if numeric and len(numeric) > 1 else 0)
        self.x_combo.blockSignals(False)

        self.y_list.blockSignals(True)
        self.y_list.clear()
        x = self.x_column()
        for column in numeric:
            item = QListWidgetItem(column, self.y_list)
            item.setData(Qt.ItemDataRole.UserRole, column)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # The first column that is not x, ticked: something to run on at once.
            ticked = column != x and not any(
                self.y_list.item(i).checkState() == Qt.CheckState.Checked for i in range(self.y_list.count() - 1)
            )
            item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
        self.y_list.blockSignals(False)
        self.changed.emit()

    # -- What was chosen ----------------------------------------------------

    def table(self) -> str:
        return self.table_combo.currentText()

    def x_column(self) -> str:
        return str(self.x_combo.currentData() or ROW_NUMBER)

    def y_columns(self) -> list[str]:
        return [
            str(self.y_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.y_list.count())
            if self.y_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def chart(self) -> str:
        """The chart the columns are drawn as."""
        return str(self.chart_combo.currentData() or "Scatter Plot")

    def set_chart(self, chart: str) -> None:
        self.chart_combo.setCurrentIndex(max(0, self.chart_combo.findData(chart)))

    def spec(self) -> tuple[str, str, tuple[str, ...]] | None:
        """``(table, x, ys)``, or None until a table and a Y are chosen."""
        ys = tuple(self.y_columns())
        if not self.table() or not ys:
            return None
        return self.table(), self.x_column(), ys

    def set_spec(self, table: str, x: str, ys: Any) -> None:
        """Choose *table*, *x* and *ys* (a test, or a window remembering them)."""
        index = self.table_combo.findText(table)
        if index >= 0:
            self.table_combo.setCurrentIndex(index)
        x_index = self.x_combo.findData(x)
        if x_index >= 0:
            self.x_combo.setCurrentIndex(x_index)
        wanted = set(ys)
        for row in range(self.y_list.count()):
            item = self.y_list.item(row)
            item.setCheckState(
                Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in wanted else Qt.CheckState.Unchecked
            )

    @classmethod
    def chart_series(cls, chart: str, table: str, x: str, y: str) -> tuple[str, dict[str, str]]:
        """A series' query and roles for one Y drawn as *chart*.

        x and y always, which the operation reads; a chart of one variable gets y as its value too.
        """
        sql = cls.series_sql(table, x, y)
        if chart not in VALUE_CHARTS:
            return sql, {"x": "x", "y": "y"}
        return sql.replace(" AS y FROM ", " AS y, " + cls._quoted(y) + " AS value FROM ", 1), \
            {"x": "x", "y": "y", "value": "value"}

    @staticmethod
    def _quoted(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    @staticmethod
    def series_sql(table: str, x: str, y: str) -> str:
        """One Y against X, each under its role's name, as every series reads its data."""
        def quoted(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        x_expr = "rowid" if x == ROW_NUMBER else quoted(x)
        return (
            f"SELECT {x_expr} AS x, {quoted(y)} AS y FROM {quoted(table)} "
            f"WHERE {x_expr} IS NOT NULL AND {quoted(y)} IS NOT NULL ORDER BY {x_expr}"
        )
