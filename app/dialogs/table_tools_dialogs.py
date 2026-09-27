"""Small forms for the table tools: group and aggregate, fill, find and replace.

Each one only collects choices and hands them back; the work itself is done
by the repository (``app/data/repo/table_tools.py``), so it is the same with
or without a dialog in front of it.
"""
from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.repo.table_tools import AGGREGATES
from app.styles.style import apply_dialog_shell
from app.utils.i18n import _


def _buttons(dialog: QDialog, layout: QVBoxLayout) -> None:
    box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog
    )
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    layout.addWidget(box)


class GroupAggregateDialog(QDialog):
    """Choose the columns to group by, the aggregate, and the new table's name."""

    def __init__(self, table: str, columns: Sequence[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Group and aggregate '{table}'").format(table=table))
        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")
        form = QFormLayout()
        root.addLayout(form)

        self.groups = QListWidget(self)
        for column in columns:
            item = QListWidgetItem(column, self.groups)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
        form.addRow(_("Group by:"), self.groups)

        self.aggregate = QComboBox(self)
        for name in AGGREGATES:
            self.aggregate.addItem(name, name)
        form.addRow(_("Aggregate:"), self.aggregate)

        self.value = QComboBox(self)
        self.value.addItem(_("(rows)"), None)
        for column in columns:
            self.value.addItem(column, column)
        form.addRow(_("Of column:"), self.value)

        self.name = QLineEdit(self)
        self.name.setPlaceholderText(_("Automatic"))
        form.addRow(_("New table name:"), self.name)
        _buttons(self, root)

    def chosen_groups(self) -> list[str]:
        return [
            self.groups.item(i).text()
            for i in range(self.groups.count())
            if self.groups.item(i).checkState() == Qt.CheckState.Checked
        ]


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
