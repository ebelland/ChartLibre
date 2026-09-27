"""Forms for the table tools: group and aggregate, computed column, fill, find and replace.

Each one only collects choices and hands them back; the work itself is done
by the repository (``app/data/repo/table_tools.py``), so it is the same with
or without a dialog in front of it.
"""
from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.data.repo._common import _is_ident, _quote_ident
from app.data.repo.table_tools import AGGREGATES
from app.styles.style import (
    CardFrame,
    action_presentation,
    apply_dialog_shell,
    apply_fusion_for_item_view_styling,
    create_action_button,
)
from app.utils.i18n import _
from app.widgets.table_preview import DataFrameTableModel


def _buttons(dialog: QDialog, layout: QVBoxLayout) -> None:
    box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog
    )
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    layout.addWidget(box)


def _aggregate_label(func: str) -> str:
    """How an aggregate reads in the form; the SQL name is the item's data."""
    return {
        "COUNT": _("Count"),
        "SUM": _("Sum"),
        "AVG": _("Mean"),
        "MIN": _("Minimum"),
        "MAX": _("Maximum"),
        "COUNT_DISTINCT": _("Distinct values"),
    }.get(func, func)


def _error_label(parent: QWidget) -> QLabel:
    label = QLabel(parent)
    label.setWordWrap(True)
    label.setProperty("muted", True)
    return label


def _preview_view(parent: QWidget) -> QTableView:
    view = QTableView(parent)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.setAlternatingRowColors(True)
    view.verticalHeader().setVisible(False)
    apply_fusion_for_item_view_styling(view)
    return view


class _MeasureRow(QWidget):
    """One measure: an aggregate of a column, with a button to remove it."""

    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, columns: Sequence[str], numeric: set[str], parent: QWidget) -> None:
        super().__init__(parent)
        self._numeric = numeric
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.aggregate = QComboBox(self)
        for func in AGGREGATES:
            self.aggregate.addItem(_aggregate_label(func), func)
        self.column = QComboBox(self)
        self.column.addItem(_("(rows)"), None)
        for column in columns:
            self.column.addItem(column, column)
        remove = QToolButton(self)
        icon, text, tooltip = action_presentation("delete")
        remove.setIcon(icon)
        remove.setToolTip(_("Remove this measure"))
        remove.setAccessibleName(text)
        remove.setAutoRaise(True)
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row.addWidget(self.aggregate, 1)
        row.addWidget(QLabel(_("of"), self), 0)
        row.addWidget(self.column, 2)
        row.addWidget(remove, 0)
        self.aggregate.currentIndexChanged.connect(self._on_aggregate)
        self.column.currentIndexChanged.connect(lambda _i: self.changed.emit())

    def _on_aggregate(self, _index: int) -> None:
        # "(rows)" only means something to Count: anything else needs a
        # column, and a number is what Sum or Mean can use.
        if self.aggregate.currentData() != "COUNT" and self.column.currentData() is None:
            for i in range(1, self.column.count()):
                if self.column.itemData(i) in self._numeric:
                    self.column.setCurrentIndex(i)
                    break
            else:
                if self.column.count() > 1:
                    self.column.setCurrentIndex(1)
        self.changed.emit()

    def set_measure(self, func: str, column: str | None) -> None:
        self.aggregate.setCurrentIndex(max(0, self.aggregate.findData(func)))
        index = self.column.findData(column) if column is not None else 0
        self.column.setCurrentIndex(max(0, index))

    def value(self) -> tuple[str, str | None]:
        return str(self.aggregate.currentData()), self.column.currentData()


class GroupAggregateDialog(QDialog):
    """Summarise a table into a new one: grouping columns, measures, a preview.

    The form only collects the choices; the repository writes the table
    (``group_aggregate``) and computes the preview from the same SELECT, so
    what is shown is what will be written.
    """

    def __init__(
        self,
        repo: Any,
        table: str,
        parent: QWidget | None = None,
        *,
        initial_group: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._repo = repo
        self._table = str(table)
        self.setWindowTitle(_("Group and aggregate '{table}'").format(table=self._table))
        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        schema = [(name, kind) for name, kind in repo._schema(self._table) if name != "Hide"]
        self._columns = [name for name, _kind in schema]
        self._numeric = {
            name for name, kind in schema
            if any(word in kind.upper() for word in ("INT", "REAL", "FLOA", "DOUB", "NUM"))
        }
        self._has_hide = "Hide" in {name for name, _kind in repo._schema(self._table)}

        top = QHBoxLayout()
        root.addLayout(top, 0)

        groups_card = CardFrame(self, "groupByCard")
        groups_card.layout().addWidget(QLabel(_("Group by:"), groups_card))
        self.groups = QListWidget(groups_card)
        # Fusion draws the check boxes; the native macOS style drew the
        # unchecked ones as nothing at all, and the checked ones barely.
        apply_fusion_for_item_view_styling(self.groups)
        for column in self._columns:
            item = QListWidgetItem(column, self.groups)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if column == initial_group else Qt.CheckState.Unchecked
            )
        self.groups.setMaximumHeight(170)
        self.groups.itemChanged.connect(lambda _item: self._schedule())
        groups_card.layout().addWidget(self.groups)
        hint = QLabel(_("One row per distinct combination. None checked: one row of totals."), groups_card)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        groups_card.layout().addWidget(hint)
        top.addWidget(groups_card, 1)

        measures_card = CardFrame(self, "measuresCard")
        measures_card.layout().addWidget(QLabel(_("Measures:"), measures_card))
        self._measure_box = QVBoxLayout()
        measures_card.layout().addLayout(self._measure_box)
        add_row = QHBoxLayout()
        add = create_action_button(
            parent=measures_card,
            action_id="add",
            action=lambda: self._add_measure(),
            presentation=(action_presentation("add")[0], _("Add measure"), _("Add another column to the summary")),
        )
        add_row.addWidget(add)
        add_row.addStretch(1)
        measures_card.layout().addLayout(add_row)
        measures_card.layout().addStretch(1)
        top.addWidget(measures_card, 1)
        self._measures: list[_MeasureRow] = []
        self._add_measure("COUNT", None)

        # A plain row, not a QFormLayout: on macOS a form centres itself
        # and leaves the name box a few characters wide.
        options = QHBoxLayout()
        root.addLayout(options, 0)
        options.addWidget(QLabel(_("New table name:"), self), 0)
        self.name = QLineEdit(self)
        self.name.textChanged.connect(lambda _t: self._schedule())
        options.addWidget(self.name, 1)
        self.include_hidden = QCheckBox(_("Include hidden rows"), self)
        self.include_hidden.setToolTip(_("Rows marked Hide are left out unless this is checked, as they are from the charts."))
        self.include_hidden.setVisible(self._has_hide)
        self.include_hidden.toggled.connect(lambda _c: self._schedule())
        options.addSpacing(12)
        options.addWidget(self.include_hidden, 0)

        root.addWidget(QLabel(_("Preview:"), self), 0)
        self._preview = _preview_view(self)
        root.addWidget(self._preview, 1)
        self._status = _error_label(self)
        root.addWidget(self._status, 0)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._ok = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText(_("Create table"))
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        root.addWidget(box, 0)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh_preview)
        self._refresh_preview()

    # -- measures -----------------------------------------------------------
    def _add_measure(self, func: str | None = None, column: str | None = None) -> None:
        row = _MeasureRow(self._columns, self._numeric, self)
        if func is None:
            # A second measure is usually a number summarised: the mean of
            # the first numeric column that is not a grouping column.
            groups = set(self.chosen_groups())
            first = next((c for c in self._columns if c in self._numeric and c not in groups), None)
            func, column = ("AVG", first) if first else ("COUNT", None)
        row.set_measure(func, column)
        row.changed.connect(self._schedule)
        row.remove_requested.connect(self._remove_measure)
        self._measures.append(row)
        self._measure_box.addWidget(row)
        self._schedule()

    def _remove_measure(self, row: _MeasureRow) -> None:
        if len(self._measures) <= 1:
            return
        self._measures.remove(row)
        row.deleteLater()
        self._schedule()

    # -- results ------------------------------------------------------------
    def chosen_groups(self) -> list[str]:
        return [
            self.groups.item(i).text()
            for i in range(self.groups.count())
            if self.groups.item(i).checkState() == Qt.CheckState.Checked
        ]

    def measures(self) -> list[tuple[str, str | None]]:
        return [row.value() for row in self._measures]

    def table_name(self) -> str:
        return self.name.text().strip() or self._repo.group_table_name(self._table, self.chosen_groups())

    # -- preview ------------------------------------------------------------
    def _schedule(self) -> None:
        if hasattr(self, "_timer"):
            self._timer.start()

    def _refresh_preview(self) -> None:
        self.name.setPlaceholderText(self._repo.group_table_name(self._table, self.chosen_groups()))
        problem = self._name_problem()
        try:
            frame, total = self._repo.group_aggregate_preview(
                self._table,
                self.chosen_groups(),
                self.measures(),
                include_hidden=self.include_hidden.isChecked(),
            )
        except Exception as exc:
            self._preview.setModel(None)
            self._status.setText(str(exc))
            self._ok.setEnabled(False)
            return
        old = self._preview.model()
        self._preview.setModel(DataFrameTableModel(frame, self._preview))
        if old is not None:
            old.deleteLater()
        self._preview.resizeColumnsToContents()
        shown = _("{count} row(s) in the new table.").format(count=total)
        if total > len(frame):
            shown += " " + _("The first {count} are shown.").format(count=len(frame))
        self._status.setText(problem or shown)
        self._ok.setEnabled(problem is None)

    def _name_problem(self) -> str | None:
        name = self.name.text().strip()
        if not name:
            return None
        if self._repo.free_table_name(name) != name:
            return _("A table or query named '{name}' already exists.").format(name=name)
        return None


class ComputedColumnDialog(QDialog):
    """A new column computed per row from a SQL expression, with a preview."""

    def __init__(self, repo: Any, table: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self._table = str(table)
        self.setWindowTitle(_("Add computed column"))
        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")

        form = QFormLayout()
        root.addLayout(form, 0)
        self.name = QLineEdit(self)
        self.name.setPlaceholderText("new_column")
        form.addRow(_("Name of the new column:"), self.name)
        self.kind = QComboBox(self)
        self.kind.addItem(_("Automatic"), None)
        for kind in ("REAL", "INTEGER", "TEXT"):
            self.kind.addItem(kind, kind)
        self.kind.setToolTip(_("Automatic keeps each value as the expression gives it."))
        form.addRow(_("Type:"), self.kind)

        middle = QHBoxLayout()
        root.addLayout(middle, 1)
        left = QVBoxLayout()
        left.addWidget(QLabel(_("Expression:"), self))
        self.expression = QPlainTextEdit(self)
        self.expression.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.expression.setPlaceholderText("2 * salary_eur\nage / 10.0\nCASE WHEN x > 0 THEN 'up' ELSE 'down' END")
        self.expression.setTabChangesFocus(True)
        left.addWidget(self.expression, 1)
        middle.addLayout(left, 2)

        right = QVBoxLayout()
        right.addWidget(QLabel(_("Columns:"), self))
        self.columns = QListWidget(self)
        self.columns.setToolTip(_("Double-click a column to insert it at the cursor."))
        for name in repo.get_columns(self._table):
            self.columns.addItem(str(name))
        self.columns.itemDoubleClicked.connect(self._insert_column)
        right.addWidget(self.columns, 1)
        middle.addLayout(right, 1)

        root.addWidget(QLabel(_("Preview (first rows):"), self), 0)
        self._preview = _preview_view(self)
        self._preview.setMaximumHeight(160)
        root.addWidget(self._preview, 0)
        self._status = _error_label(self)
        root.addWidget(self._status, 0)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._ok = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText(_("Add column"))
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        root.addWidget(box, 0)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh_preview)
        self.expression.textChanged.connect(self._timer.start)
        self.name.textChanged.connect(lambda _t: self._timer.start())
        self._refresh_preview()

    def _insert_column(self, item: QListWidgetItem) -> None:
        name = item.text()
        self.expression.insertPlainText(name if _is_ident(name) else _quote_ident(name))
        self.expression.setFocus()

    def column_name(self) -> str:
        return self.name.text().strip()

    def column_type(self) -> str | None:
        return self.kind.currentData()

    def expression_text(self) -> str:
        return self.expression.toPlainText().strip()

    def _refresh_preview(self) -> None:
        name, expression = self.column_name(), self.expression_text()
        old = self._preview.model()
        self._preview.setModel(None)
        if old is not None:
            old.deleteLater()
        if not expression:
            self._status.setText(_("Type an expression; it is evaluated once for every row."))
            self._ok.setEnabled(False)
            return
        try:
            values = self._repo.preview_expression(self._table, expression)
        except Exception as exc:
            self._status.setText(str(exc))
            self._ok.setEnabled(False)
            return
        self._preview.setModel(
            DataFrameTableModel(pd.DataFrame({name or "new_column": values}), self._preview)
        )
        self._preview.horizontalHeader().setStretchLastSection(True)
        problem = None
        if not name:
            problem = _("Type a name for the new column first.")
        elif name in {self.columns.item(i).text() for i in range(self.columns.count())}:
            problem = _("'{name}' is already a column of this table.").format(name=name)
        self._status.setText(problem or "")
        self._ok.setEnabled(problem is None)


#: Fill methods as shown, paired with the repository's own names.
FILL_CHOICES: tuple[tuple[str, str], ...] = (
    ("A value", "constant"),
    ("The column mean", "mean"),
    ("The column median", "median"),
    ("The previous value", "previous"),
    ("Linear interpolation", "linear"),
)


class FillMissingDialog(QDialog):
    """How to fill one column's empty cells."""

    def __init__(self, column: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Fill empty cells in '{column}'").format(column=column))
        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size=None)
        form = QFormLayout()
        root.addLayout(form)
        self.method = QComboBox(self)
        for label, key in FILL_CHOICES:
            self.method.addItem(_(label), key)
        form.addRow(_("Fill with:"), self.method)
        self.value = QLineEdit(self)
        form.addRow(_("Value:"), self.value)
        self.method.currentIndexChanged.connect(
            lambda _i: self.value.setEnabled(self.method.currentData() == "constant")
        )
        _buttons(self, root)


class FindReplaceDialog(QDialog):
    """What to find, what to put instead, and where."""

    def __init__(self, column: str | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Find and replace"))
        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size=None)
        form = QFormLayout()
        root.addLayout(form)
        self.find = QLineEdit(self)
        form.addRow(_("Find:"), self.find)
        self.replace = QLineEdit(self)
        form.addRow(_("Replace with:"), self.replace)
        self.scope = QComboBox(self)
        if column:
            self.scope.addItem(_("Column '{column}'").format(column=column), column)
        self.scope.addItem(_("Whole table"), None)
        form.addRow(_("In:"), self.scope)
        self.whole_cell = QCheckBox(_("Only cells that match exactly"), self)
        form.addRow("", self.whole_cell)
        _buttons(self, root)
