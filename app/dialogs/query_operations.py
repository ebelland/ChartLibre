"""The Query Builder's Build frame: a form per Tables operation that writes its SQL (AN-11).

Join, Union, Stack, Split, Transpose, Recode, Bin, Group: each form collects
the choices JMP's Tables menu asks for and returns the statement
app.data.sql_builders writes, which the Query Builder puts in its editor and
runs. Where the statement needs values from the data - split's and
transpose's labels, recode's values, the bins' edges - the form reads them
here, through the repository, as plain SQL.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from app.data import sql_builders as sb
from app.data.repo.table_tools import AGGREGATES
from app.data.sqlite_repo import SqliteRepo
from app.styles.style import apply_fusion_for_item_view_styling, stdSizeAndlayout
from app.utils.i18n import _

#: The most distinct values Recode lists.
RECODE_LIMIT = 200
_NONE = "(none)"


def _check_list(parent: QWidget, height: int = 96) -> QListWidget:
    view = QListWidget(parent)
    # Fusion draws the check boxes; the native macOS style hid the unchecked ones.
    apply_fusion_for_item_view_styling(view)
    view.setMaximumHeight(height)
    return view


def _fill_checks(view: QListWidget, names: Sequence[str], checked: set[str]) -> None:
    view.clear()
    for name in names:
        item = QListWidgetItem(name, view)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)


def _checked(view: QListWidget) -> list[str]:
    return [view.item(i).text() for i in range(view.count())
            if view.item(i).checkState() == Qt.CheckState.Checked]


class OperationForm(QWidget):
    """One operation: its controls in a form, and the SQL they make."""

    #: Shown in the operation combo, and the sentence under it.
    NAME = ""
    DESCRIPTION = ""

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self.form = QFormLayout(self)
        stdSizeAndlayout(self.form)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    # -- helpers ----------------------------------------------------------
    def tables(self) -> list[str]:
        return list(self._repo.list_table_names())

    def columns(self, table: str) -> list[str]:
        return [c for c in self._repo.get_columns(table) if c != "Hide"] if table else []

    def numeric_columns(self, table: str) -> list[str]:
        numeric = []
        for row in self._repo.table_info(table) if table else []:
            declared = str(row[2]).upper()
            if any(token in declared for token in ("INT", "REAL", "NUM", "DEC", "FLOA", "DOUB")):
                numeric.append(str(row[1]))
        return [c for c in numeric if c != "Hide"]

    def table_combo(self, on_change=None) -> QComboBox:
        combo = QComboBox(self)
        combo.addItems(self.tables())
        if on_change is not None:
            combo.currentTextChanged.connect(on_change)
        return combo

    def values(self, table: str, column: str, *, limit: int | None = None) -> list[object]:
        """A column's distinct values, in the order they first appear."""
        sql = (f"SELECT {sb.q(column)} AS v FROM {sb.q(table)} WHERE {sb.q(column)} IS NOT NULL "
               f"GROUP BY {sb.q(column)} ORDER BY MIN(rowid)") + (f" LIMIT {int(limit)}" if limit else "")
        return self._repo.query_df(sql)["v"].tolist()

    def sql(self) -> str:
        raise NotImplementedError


class JoinForm(OperationForm):
    NAME = "Join"
    DESCRIPTION = "Rows of two tables matched on one or more columns: inner, left, right or full outer."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.left = self.table_combo()
        self.right = self.table_combo()
        if self.right.count() > 1:
            self.right.setCurrentIndex(1)
        self.kind = QComboBox(self)
        for kind, label in zip(sb.JOIN_KINDS, (_("Inner: matched rows only"), _("Left: every left row"),
                                               _("Right: every right row"), _("Full outer: every row"))):
            self.kind.addItem(label, kind)
        self.form.addRow(_("Left:"), self.left)
        self.form.addRow(_("Right:"), self.right)
        self.form.addRow(_("Keep:"), self.kind)
        self.pairs: list[tuple[QComboBox, QComboBox]] = []
        for index in range(2):
            pair = (QComboBox(self), QComboBox(self))
            for combo in pair:
                # Two to a row: each asks for little, or the row squeezes the form's labels.
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(4)
            self.pairs.append(pair)
            row = QWidget(self)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(pair[0], 1)
            layout.addWidget(QLabel("=", row))
            layout.addWidget(pair[1], 1)
            self.form.addRow(_("Match:") if index == 0 else "", row)
        self.match = QCheckBox(_("Add a Match column"), self)
        self.match.setToolTip(_("Each row's origin: Both, Left only or Right only."))
        self.form.addRow("", self.match)
        self.left.currentTextChanged.connect(self._reload)
        self.right.currentTextChanged.connect(self._reload)
        self._reload()

    def _reload(self, *_args: object) -> None:
        left, right = self.columns(self.left.currentText()), self.columns(self.right.currentText())
        shared = [c for c in left if c in right]
        for index, (a, b) in enumerate(self.pairs):
            for combo, names in ((a, left), (b, right)):
                combo.clear()
                if index:
                    combo.addItem(_NONE)
                combo.addItems(names)
            # The first pair starts on a column both tables have, the way a person would match them.
            if index == 0 and shared:
                a.setCurrentText(shared[0])
                b.setCurrentText(shared[0])

    def sql(self) -> str:
        left, right = self.left.currentText(), self.right.currentText()
        pairs = [(a.currentText(), b.currentText()) for a, b in self.pairs
                 if a.currentText() not in ("", _NONE) and b.currentText() not in ("", _NONE)]
        return sb.join(left, self.columns(left), right, self.columns(right), pairs,
                       kind=str(self.kind.currentData()), match_column=self.match.isChecked())


class UnionForm(OperationForm):
    NAME = "Union"
    DESCRIPTION = "Tables one under the other (concatenate), columns matched by name."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.tables_list = _check_list(self, 120)
        _fill_checks(self.tables_list, self.tables(), set(self.tables()[:2]))
        self.source = QCheckBox(_("Add a Source column with each row's table"), self)
        self.source.setChecked(True)
        self.distinct = QCheckBox(_("Drop repeated rows (UNION instead of UNION ALL)"), self)
        self.form.addRow(_("Tables:"), self.tables_list)
        self.form.addRow("", self.source)
        self.form.addRow("", self.distinct)

    def sql(self) -> str:
        chosen = _checked(self.tables_list)
        return sb.union({name: self.columns(name) for name in chosen},
                        source_column="Source" if self.source.isChecked() else None,
                        distinct=self.distinct.isChecked())


class StackForm(OperationForm):
    NAME = "Stack"
    DESCRIPTION = "Wide to long: the chosen columns become rows of Label and Data, the others repeated."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.table = self.table_combo(self._reload)
        self.stacked = _check_list(self)
        self.label = QLineEdit("Label", self)
        self.value = QLineEdit("Data", self)
        self.drop = QCheckBox(_("Leave out empty values"), self)
        self.form.addRow(_("Table:"), self.table)
        self.form.addRow(_("Stack:"), self.stacked)
        self.form.addRow(_("Label column:"), self.label)
        self.form.addRow(_("Data column:"), self.value)
        self.form.addRow("", self.drop)
        self._reload()

    def _reload(self, *_args: object) -> None:
        table = self.table.currentText()
        _fill_checks(self.stacked, self.columns(table), set(self.numeric_columns(table)))

    def sql(self) -> str:
        table = self.table.currentText()
        stacked = _checked(self.stacked)
        keep = [c for c in self.columns(table) if c not in stacked]
        return sb.stack(table, stacked, keep, label=self.label.text().strip() or "Label",
                        value=self.value.text().strip() or "Data", drop_missing=self.drop.isChecked())


class SplitForm(OperationForm):
    NAME = "Split"
    DESCRIPTION = "Long to wide: one column's values become columns, filled from another."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.table = self.table_combo(self._reload)
        self.label = QComboBox(self)
        self.value = QComboBox(self)
        self.groups = _check_list(self)
        self.groups.setToolTip(_("None checked: the n-th value of every label goes in row n."))
        self.aggregate = QComboBox(self)
        self.aggregate.addItems(sb.SPLIT_AGGREGATES)
        self.aggregate.setToolTip(_("How two rows for the same group and label are combined."))
        self.form.addRow(_("Table:"), self.table)
        self.form.addRow(_("Split by:"), self.label)
        self.form.addRow(_("Values:"), self.value)
        self.form.addRow(_("Group by:"), self.groups)
        self.form.addRow(_("Combine:"), self.aggregate)
        self._reload()

    def _reload(self, *_args: object) -> None:
        names = self.columns(self.table.currentText())
        for combo in (self.label, self.value):
            combo.clear()
            combo.addItems(names)
        if len(names) > 1:
            self.value.setCurrentIndex(len(names) - 1)
        _fill_checks(self.groups, names, set())

    def sql(self) -> str:
        table, label = self.table.currentText(), self.label.currentText()
        labels = self.values(table, label, limit=sb.MAX_NEW_COLUMNS + 1)
        return sb.split(table, label, self.value.currentText(), labels, _checked(self.groups),
                        aggregate=self.aggregate.currentText())


class TransposeForm(OperationForm):
    NAME = "Transpose"
    DESCRIPTION = "Columns become rows and rows columns, named by a label column or numbered."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.table = self.table_combo(self._reload)
        self.label = QComboBox(self)
        self.chosen = _check_list(self)
        self.form.addRow(_("Table:"), self.table)
        self.form.addRow(_("Name columns by:"), self.label)
        self.form.addRow(_("Transpose:"), self.chosen)
        self._reload()

    def _reload(self, *_args: object) -> None:
        table = self.table.currentText()
        self.label.clear()
        self.label.addItem(_("(row numbers)"), "")
        for column in self.columns(table):
            self.label.addItem(column, column)
        _fill_checks(self.chosen, self.columns(table), set(self.numeric_columns(table)))

    def sql(self) -> str:
        table, label = self.table.currentText(), str(self.label.currentData() or "")
        columns = [c for c in _checked(self.chosen) if c != label]
        if label:
            labels = self._repo.query_df(f"SELECT {sb.q(label)} AS v FROM {sb.q(table)} ORDER BY rowid")["v"].tolist()
            return sb.transpose(table, columns, label_column=label, labels=labels)
        rows = int(self._repo.query_df(f"SELECT COUNT(*) AS n FROM {sb.q(table)}")["n"].iloc[0])
        return sb.transpose(table, columns, rows=rows)


class RecodeForm(OperationForm):
    NAME = "Recode"
    DESCRIPTION = "Replace a column's values with new ones; the values not given stay as they are."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.table = self.table_combo(self._reload)
        self.column = QComboBox(self)
        self.column.currentTextChanged.connect(self._fill_values)
        self.mapping = QTableWidget(0, 2, self)
        self.mapping.setHorizontalHeaderLabels([_("Value"), _("New value")])
        self.mapping.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.mapping.verticalHeader().setVisible(False)
        self.mapping.setMaximumHeight(170)
        self.mapping.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.new_column = QLineEdit(self)
        self.new_column.setPlaceholderText(_("empty: replace the column"))
        self.form.addRow(_("Table:"), self.table)
        self.form.addRow(_("Column:"), self.column)
        self.form.addRow(self.mapping)
        self.form.addRow(_("New column:"), self.new_column)
        self._reload()

    def _reload(self, *_args: object) -> None:
        self.column.clear()
        self.column.addItems(self.columns(self.table.currentText()))

    def _fill_values(self, column: str) -> None:
        self.mapping.setRowCount(0)
        if not column:
            return
        self.new_column.setText(f"{column}_recoded")
        for value in self.values(self.table.currentText(), column, limit=RECODE_LIMIT):
            row = self.mapping.rowCount()
            self.mapping.insertRow(row)
            text = f"{value:g}" if isinstance(value, float) else str(value)
            old = QTableWidgetItem(text)
            old.setFlags(old.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping.setItem(row, 0, old)
            self.mapping.setItem(row, 1, QTableWidgetItem(""))

    def sql(self) -> str:
        mapping = {}
        for row in range(self.mapping.rowCount()):
            new = (self.mapping.item(row, 1).text() if self.mapping.item(row, 1) else "").strip()
            if new:
                mapping[self.mapping.item(row, 0).text()] = new
        table = self.table.currentText()
        return sb.recode(table, self.columns(table), self.column.currentText(), mapping,
                         new_column=self.new_column.text().strip() or None)


class BinForm(OperationForm):
    NAME = "Bin"
    DESCRIPTION = "A numeric column cut into intervals: equal width, equal counts (quantiles) or your edges."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.table = self.table_combo(self._reload)
        self.column = QComboBox(self)
        self.column.currentTextChanged.connect(lambda name: self.new_column.setText(f"{name}_bin" if name else ""))
        self.method = QComboBox(self)
        for method, label in (("equal width", _("Equal width")), ("quantiles", _("Equal counts (quantiles)")),
                              ("edges", _("My edges"))):
            self.method.addItem(label, method)
        self.count = QSpinBox(self)
        self.count.setRange(1, 100)
        self.count.setValue(5)
        self.edges = QLineEdit(self)
        self.edges.setPlaceholderText("0, 10, 20, 50")
        self.labels = QComboBox(self)
        for style, label in (("intervals", _("Intervals [a, b)")), ("numbers", _("Numbers 1, 2, 3...")),
                             ("midpoints", _("Midpoints"))):
            self.labels.addItem(label, style)
        self.new_column = QLineEdit(self)
        self.form.addRow(_("Table:"), self.table)
        self.form.addRow(_("Column:"), self.column)
        self.form.addRow(_("Bins:"), self.method)
        self.form.addRow(_("How many:"), self.count)
        self.form.addRow(_("Edges:"), self.edges)
        self.form.addRow(_("Labels:"), self.labels)
        self.form.addRow(_("New column:"), self.new_column)
        self.method.currentIndexChanged.connect(self._show_rows)
        self._reload()
        self._show_rows()

    def _show_rows(self, *_args: object) -> None:
        edges = self.method.currentData() == "edges"
        self.form.setRowVisible(self.count, not edges)
        self.form.setRowVisible(self.edges, edges)

    def _reload(self, *_args: object) -> None:
        self.column.clear()
        self.column.addItems(self.numeric_columns(self.table.currentText()))

    def sql(self) -> str:
        table, column = self.table.currentText(), self.column.currentText()
        if not column:
            raise ValueError("Choose a numeric column.")
        values = self._repo.query_df(f"SELECT CAST({sb.q(column)} AS REAL) AS v FROM {sb.q(table)}")["v"]
        edges = [part for part in self.edges.text().replace(";", ",").split(",") if part.strip()]
        try:
            numbers = [float(part) for part in edges]
        except ValueError as exc:
            raise ValueError("The edges must be numbers separated by commas.") from exc
        cuts = sb.bin_edges(values.to_numpy(float), str(self.method.currentData()), self.count.value(), numbers)
        return sb.bin_column(table, self.columns(table), column, list(np.asarray(cuts, dtype=float)),
                             new_column=self.new_column.text().strip() or f"{column}_bin",
                             style=str(self.labels.currentData()))


class GroupForm(OperationForm):
    NAME = "Group"
    DESCRIPTION = "One row per group: counts, sums, means... of the chosen columns (summary table)."

    def __init__(self, repo: SqliteRepo, parent: QWidget | None = None) -> None:
        super().__init__(repo, parent)
        self.table = self.table_combo(self._reload)
        self.groups = _check_list(self)
        self.aggregate = QComboBox(self)
        self.aggregate.addItems(AGGREGATES)
        self.aggregate.setCurrentText("AVG")
        self.measured = _check_list(self)
        self.form.addRow(_("Table:"), self.table)
        self.form.addRow(_("Group by:"), self.groups)
        self.form.addRow(_("Statistic:"), self.aggregate)
        self.form.addRow(_("Of:"), self.measured)
        self._reload()

    def _reload(self, *_args: object) -> None:
        table = self.table.currentText()
        _fill_checks(self.groups, self.columns(table), set())
        _fill_checks(self.measured, self.columns(table), set(self.numeric_columns(table)))

    def sql(self) -> str:
        measures = [("COUNT", None)] + [(self.aggregate.currentText(), c) for c in _checked(self.measured)]
        return self._repo._group_select(self.table.currentText(), _checked(self.groups), measures)


FORMS: tuple[type[OperationForm], ...] = (JoinForm, UnionForm, StackForm, SplitForm, TransposeForm, RecodeForm,
                                          BinForm, GroupForm)
