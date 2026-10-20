"""Column Info: one column's name, type and description, as JMP's Column Info dialog (R-08).

Opened from a column's context menu in the table preview. What it changes:

* the name (the column renamed, its description following it);
* the data type - REAL, INTEGER, TEXT - its values converted (Cast);
* the modelling type forced - Continuous, Ordinal, Nominal - which every
  analysis then takes for the column, whatever its values suggest;
* label, unit, notes;
* missing value codes, which the analyses read as empty;
* value labels, which the analyses show in place of the codes.

The description is kept with the table (app.data.column_metadata); OK is
one undo step for all of it.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from app.data import column_metadata as cm
from app.data.repo.table_tools import CAST_TYPES
from app.dialogs.two_panels_dialog_base import PanelDialog
from app.logs.logger import applogger
from app.styles.style import stdSizeAndlayout
from app.utils.i18n import _
from app.utils.messages import show_message

#: The modelling types offered, "" first: the values decide.
MODELLING_TYPES: tuple[tuple[str, str], ...] = (
    ("", "From the values"), ("continuous", "Continuous"), ("ordinal", "Ordinal"), ("nominal", "Nominal"),
)
#: A column with more distinct values than this is not offered "Fill from the data".
MAX_LABELLED_VALUES = 100


class ColumnInfoDialog(PanelDialog):
    """Edit one column of *table*: its name, types and description."""

    def __init__(self, repo: Any, table: str, column: str, parent: QWidget | None = None) -> None:
        super().__init__(parent, title=_("Column Info: {column}").format(column=column), size=None)
        self._repo, self._table, self._column = repo, table, column
        self._info = cm.column_info(repo, table).get(column, cm.ColumnInfo())
        declared = {str(row[1]): str(row[2] or "").upper() for row in repo.table_info(table)}
        self._declared = declared.get(column, "")
        values = repo.query_df(f'SELECT "{column.replace(chr(34), chr(34) * 2)}" AS v FROM '
                               f'"{table.replace(chr(34), chr(34) * 2)}"')["v"]
        self._values = values

        # -- The column: name and types.
        form = QFormLayout()
        stdSizeAndlayout(form)
        self.add_card(_("Column"), "columnInfoCard").layout().addLayout(form)
        self.name = QLineEdit(column, self)
        form.addRow(_("Name:"), self.name)
        self.data_type = QComboBox(self)
        for kind in CAST_TYPES:
            self.data_type.addItem(kind, kind)
        if self._declared and self._declared not in CAST_TYPES:
            self.data_type.insertItem(0, self._declared, self._declared)
        self.data_type.setCurrentIndex(max(0, self.data_type.findData(self._declared or "TEXT")))
        self.data_type.setToolTip(_("How the values are stored. Changing it converts them; a value that cannot "
                                    "be read in the new type is left empty."))
        form.addRow(_("Data type:"), self.data_type)
        self.modelling = QComboBox(self)
        for key, label in MODELLING_TYPES:
            self.modelling.addItem(_(label), key)
        self.modelling.setCurrentIndex(max(0, self.modelling.findData(self._info.kind)))
        self.modelling.setToolTip(_("How the analyses take the column: as measurements (continuous), ordered "
                                    "levels (ordinal) or categories (nominal). Forced here, it is the column's "
                                    "type in every analysis."))
        form.addRow(_("Modelling type:"), self.modelling)
        self.label = QLineEdit(self._info.label, self)
        form.addRow(_("Label:"), self.label)
        self.unit = QLineEdit(self._info.unit or self._imported_unit(), self)
        self.unit.setPlaceholderText(_("mm, °C, mV..."))
        form.addRow(_("Unit:"), self.unit)
        self.notes = QPlainTextEdit(self._info.notes, self)
        self.notes.setFixedHeight(self.notes.fontMetrics().height() * 3 + 12)
        form.addRow(_("Notes:"), self.notes)

        # -- The values: what counts as missing, and their labels.
        values_card = self.add_card(_("Values"), "columnValuesCard", stretch=1)
        summary = QLabel(self._summary(), values_card)
        summary.setProperty("muted", True)
        values_card.layout().addWidget(summary)
        missing_form = QFormLayout()
        stdSizeAndlayout(missing_form)
        self.missing = QLineEdit(", ".join(self._info.missing), self)
        self.missing.setPlaceholderText(_("-999, NA, n/d"))
        self.missing.setToolTip(_("Values that mean \"no value\", separated by commas: the analyses read them "
                                  "as empty."))
        missing_form.addRow(_("Missing value codes:"), self.missing)
        values_card.layout().addLayout(missing_form)
        self.labels = QTableWidget(0, 2, values_card)
        self.labels.setHorizontalHeaderLabels([_("Value"), _("Label")])
        self.labels.verticalHeader().setVisible(False)
        self.labels.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.labels.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.labels.setMinimumHeight(self.labels.verticalHeader().defaultSectionSize() * 4 + 30)
        self.labels.setToolTip(_("Value labels: the analyses show the label in place of the value (1 -> male)."))
        for value, text in self._info.value_labels.items():
            self._add_label_row(value, text)
        values_card.layout().addWidget(self.labels, 1)
        row = QHBoxLayout()
        stdSizeAndlayout(row)
        fill = QPushButton(_("Fill from the data"), values_card)
        fill.setToolTip(_("A row for each distinct value not listed yet (up to {n}).").format(n=MAX_LABELLED_VALUES))
        fill.clicked.connect(self._fill_from_data)
        add = QPushButton(_("Add"), values_card)
        add.clicked.connect(lambda: self._add_label_row("", ""))
        remove = QPushButton(_("Remove"), values_card)
        remove.clicked.connect(self._remove_label_rows)
        for button in (fill, add, remove):
            row.addWidget(button)
        row.addStretch(1)
        values_card.layout().addLayout(row)

        self.add_action("apply", self._apply, default=True)
        self.add_action("close", self.reject)

    # ------------------------------------------------------------------
    def _imported_unit(self) -> str:
        """The unit an importer found for the column, offered until one is written here."""
        source = self._repo.get_table_info(self._table).get("source_metadata") or {}
        return str(dict(source.get("units") or {}).get(self._column, ""))

    def _summary(self) -> str:
        present = self._values.dropna()
        present = present[present.astype(str).str.strip() != ""]
        return _("{count} values, {empty} empty, {distinct} distinct").format(
            count=len(present), empty=len(self._values) - len(present), distinct=present.astype(str).nunique())

    def _add_label_row(self, value: str, text: str) -> None:
        position = self.labels.rowCount()
        self.labels.insertRow(position)
        self.labels.setItem(position, 0, QTableWidgetItem(str(value)))
        self.labels.setItem(position, 1, QTableWidgetItem(str(text)))

    def _fill_from_data(self) -> None:
        present = self._values.dropna()
        present = present[present.astype(str).str.strip() != ""]
        listed = {self.labels.item(r, 0).text().strip() for r in range(self.labels.rowCount())
                  if self.labels.item(r, 0)}
        distinct = pd.unique(present.map(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)))
        if len(distinct) > MAX_LABELLED_VALUES:
            show_message(self, "column_info.too_many_values", count=len(distinct), limit=MAX_LABELLED_VALUES)
            return
        for value in sorted(distinct, key=lambda v: (pd.to_numeric(v, errors="coerce"), v)):
            if value not in listed:
                self._add_label_row(value, "")

    def _remove_label_rows(self) -> None:
        for position in sorted({index.row() for index in self.labels.selectedIndexes()}, reverse=True):
            self.labels.removeRow(position)

    def info(self) -> cm.ColumnInfo:
        """The description as the dialog holds it now."""
        labels: dict[str, str] = {}
        for position in range(self.labels.rowCount()):
            value, text = self.labels.item(position, 0), self.labels.item(position, 1)
            if value and text and value.text().strip() and text.text().strip():
                labels[value.text().strip()] = text.text().strip()
        return cm.ColumnInfo(label=self.label.text().strip(), unit=self.unit.text().strip(),
                             notes=self.notes.toPlainText().strip(), kind=str(self.modelling.currentData() or ""),
                             missing=[code.strip() for code in self.missing.text().split(",") if code.strip()],
                             value_labels=labels)

    def _apply(self) -> None:
        name = self.name.text().strip()
        if not name:
            show_message(self, "column_info.no_name")
            return
        repo, table, column = self._repo, self._table, self._column
        new_type = str(self.data_type.currentData() or "")
        try:
            entry = repo.snapshot_for_undo([table, "__table_descriptors__"],
                                           label=f"Column Info of '{column}' in '{table}'")
            failed = 0
            if new_type != self._declared and new_type in CAST_TYPES:
                failed = repo.cast_column(table, column, new_type, undo_entry=entry)
            if name != column:
                repo.rename_table_column(table, column, name)
                cm.rename_column_info(repo, table, column, name)
            cm.set_column_info(repo, table, name, self.info())
        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the dialog stays open
            applogger.exception("Column Info of %s.%s could not be saved", table, column)
            show_message(self, "column_info.failed", error=exc)
            return
        if failed:
            show_message(self, "column_info.cast_failed", count=failed, kind=new_type)
        self.column_name = name
        self.accept()
