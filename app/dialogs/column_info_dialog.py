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
one undo step for all of it. The arrows at the top go to the previous or next
column without closing (P-01): the changes to this one are saved first, as OK
saves them.
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from app.data import column_metadata as cm
from app.data.sqlite_repo import CAST_TYPES
from app.dialogs.two_panels_dialog_base import Column, ColumnsDialog
from app.logs.logger import applogger
from app.styles.style import stdSizeAndlayout
from app.utils.i18n import _
from app.utils.messages import show_message
from app.widgets.chart_jump_bar import jump_arrow

#: The modelling types offered, "" first: the values decide.
MODELLING_TYPES: tuple[tuple[str, str], ...] = (
    ("", "From the values"), ("continuous", "Continuous"), ("ordinal", "Ordinal"), ("nominal", "Nominal"),
)
#: A column with more distinct values than this is not offered "Fill from the data".
MAX_LABELLED_VALUES = 100
#: The columns the application keeps on a table for itself: the arrows pass them by.
APPLICATION_COLUMNS = frozenset({"Hide", "Selected", "_id"})


class ColumnInfoDialog(ColumnsDialog):
    """Edit one column of *table*: its name, types and description."""

    def __init__(self, repo: Any, table: str, column: str, parent: QWidget | None = None) -> None:
        # Two columns, the column's description and its values side by side: one tall column was taller than
        # a laptop's screen.
        super().__init__(parent, title=_("Column Info: {column}").format(column=column), size=None,
                         columns=(Column(), Column()))
        self._repo, self._table, self._column = repo, table, column
        #: Whether a column was saved while moving between them: Close then says so, as OK would.
        self._saved = False
        #: The table's columns in order, the application's own left out: the arrows go through them.
        self._table_columns = [str(row[1]) for row in repo.table_info(table) if str(row[1]) not in APPLICATION_COLUMNS]

        # -- Which column: the previous or the next one, without closing (P-01) - at the left of the bottom row,
        # where they take no height of their own.
        # The arrows the charts' bar has: one look for "the previous one, the next one".
        self.previous_button = jump_arrow(self, "chart_previous", lambda: self._go(-1), tooltip=_(
            "The previous column - the changes to this one are saved first"))
        self.next_button = jump_arrow(self, "chart_next", lambda: self._go(1), tooltip=_(
            "The next column - the changes to this one are saved first"))
        self.position = QLabel("", self)
        self.position.setProperty("muted", True)
        for position, widget in enumerate((self.previous_button, self.next_button, self.position)):
            self.action_row.insertWidget(position, widget)

        # -- The column: name and types.
        form = QFormLayout()
        stdSizeAndlayout(form)
        self.add_column_card(0, _("Column"), "columnInfoCard").layout().addLayout(form)
        self.name = QLineEdit(self)
        form.addRow(_("Name:"), self.name)
        self.data_type = QComboBox(self)
        self.data_type.setToolTip(_("How the values are stored. Changing it converts them; a value that cannot "
                                    "be read in the new type is left empty."))
        form.addRow(_("Data type:"), self.data_type)
        self.modelling = QComboBox(self)
        for key, label in MODELLING_TYPES:
            self.modelling.addItem(_(label), key)
        self.modelling.setToolTip(_("How the analyses take the column: as measurements (continuous), ordered "
                                    "levels (ordinal) or categories (nominal). Forced here, it is the column's "
                                    "type in every analysis."))
        form.addRow(_("Modelling type:"), self.modelling)
        self.label = QLineEdit(self)
        form.addRow(_("Label:"), self.label)
        self.unit = QLineEdit(self)
        self.unit.setPlaceholderText(_("mm, °C, mV..."))
        form.addRow(_("Unit:"), self.unit)
        self.notes = QPlainTextEdit(self)
        self.notes.setFixedHeight(self.notes.fontMetrics().height() * 3 + 12)
        form.addRow(_("Notes:"), self.notes)

        # How the values are shown - in the table preview, for now; the values stay as stored.
        self.format = QComboBox(self)
        for key, label in cm.FORMATS:
            self.format.addItem(_(label), key)
        self.format.setToolTip(_("How the values are shown in the table: the values themselves do not change."))
        self.decimals = QSpinBox(self)
        self.decimals.setRange(0, 15)
        self.decimals.setSuffix(_(" dec."))
        format_row = QWidget(self)
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.addWidget(self.format, 1)
        format_layout.addWidget(self.decimals)
        form.addRow(_("Format:"), format_row)
        self.pattern = QLineEdit(self)
        self.pattern.setToolTip(_("strftime codes: %d day, %m month, %Y year, %H:%M:%S time, %b month name."))
        form.addRow(_("Pattern:"), self.pattern)
        self._pattern_label = form.labelForField(self.pattern)
        self.example = QLabel("", self)
        self.example.setProperty("muted", True)
        form.addRow("", self.example)
        for signal in (self.format.currentIndexChanged, self.decimals.valueChanged, self.pattern.textChanged):
            signal.connect(lambda *_args: self._show_format())

        # -- The values: what counts as missing, and their labels.
        values_card = self.add_column_card(1, _("Values"), "columnValuesCard", stretch=1)
        self.summary = QLabel("", values_card)
        self.summary.setProperty("muted", True)
        values_card.layout().addWidget(self.summary)
        missing_form = QFormLayout()
        stdSizeAndlayout(missing_form)
        self.missing = QLineEdit(self)
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
        self._load(column)

    # ------------------------------------------------------------------
    # Which column
    # ------------------------------------------------------------------

    def _load(self, column: str) -> None:
        """Show *column*: its name, types and description, its values, and where it is among the table's."""
        repo, table = self._repo, self._table
        self._column = column
        self.setWindowTitle(_("Column Info: {column}").format(column=column))
        self._info = cm.column_info(repo, table).get(column, cm.ColumnInfo())
        declared = {str(row[1]): str(row[2] or "").upper() for row in repo.table_info(table)}
        self._declared = declared.get(column, "")
        self._values = repo.query_df(f'SELECT "{column.replace(chr(34), chr(34) * 2)}" AS v FROM '
                                     f'"{table.replace(chr(34), chr(34) * 2)}"')["v"]
        info = self._info
        self.name.setText(column)
        self.data_type.clear()
        for kind in CAST_TYPES:
            self.data_type.addItem(kind, kind)
        if self._declared and self._declared not in CAST_TYPES:
            self.data_type.insertItem(0, self._declared, self._declared)
        self.data_type.setCurrentIndex(max(0, self.data_type.findData(self._declared or "TEXT")))
        self.modelling.setCurrentIndex(max(0, self.modelling.findData(info.kind)))
        self.label.setText(info.label)
        self.unit.setText(info.unit or self._imported_unit())
        self.notes.setPlainText(info.notes)
        self.format.setCurrentIndex(max(0, self.format.findData(info.format)))
        self.decimals.setValue(info.decimals)
        self.pattern.setText(info.pattern)
        self._show_format()
        self.summary.setText(self._summary())
        self.missing.setText(", ".join(info.missing))
        self.labels.setRowCount(0)
        for value, text in info.value_labels.items():
            self._add_label_row(value, text)
        index = self._table_columns.index(column) if column in self._table_columns else -1
        self.position.setText(_("Column {number} of {count}").format(number=index + 1, count=len(self._table_columns))
                              if index >= 0 else "")
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(0 <= index < len(self._table_columns) - 1)
        self._shown = self._state()

    def _state(self) -> tuple[Any, ...]:
        """What the dialog would save: compared with what it showed, whether anything changed."""
        return self.name.text().strip(), str(self.data_type.currentData() or ""), self.info()

    def _go(self, step: int) -> None:
        """The previous (-1) or next (+1) column; this one's changes saved first - a failed save stays here."""
        if self._state() != self._shown and not self._save():
            return
        index = self._table_columns.index(self._column) + step if self._column in self._table_columns else -1
        if 0 <= index < len(self._table_columns):
            self._load(self._table_columns[index])

    def reject(self) -> None:
        """Close: a column saved on the way (the arrows) counts as OK, so that what shows the table reloads."""
        if self._saved:
            self.accept()
        else:
            super().reject()

    # ------------------------------------------------------------------
    def _imported_unit(self) -> str:
        """The unit an importer found for the column, offered until one is written here."""
        source = self._repo.get_table_info(self._table).get("source_metadata") or {}
        return str(dict(source.get("units") or {}).get(self._column, ""))

    def _show_format(self) -> None:
        """Decimals for a number format, the pattern for dates, and the first value as it will look."""
        kind = str(self.format.currentData() or "")
        self.decimals.setVisible(kind in ("fixed", "percent", "scientific", "thousands"))
        self.pattern.setVisible(kind == "datetime")
        if self._pattern_label is not None:
            self._pattern_label.setVisible(kind == "datetime")
        first = next((value for value in self._values if value is not None and str(value).strip() != ""), None)
        info = cm.ColumnInfo(format=kind, decimals=self.decimals.value(), pattern=self.pattern.text())
        self.example.setText(_("{stored} is shown {shown}").format(stored=first, shown=cm.format_value(first, info))
                             if kind and first is not None else "")

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
                             value_labels=labels, format=str(self.format.currentData() or ""),
                             decimals=self.decimals.value(),
                             pattern=self.pattern.text().strip() or cm.DEFAULT_DATE_PATTERN)

    def _apply(self) -> None:
        """OK: this column saved, the dialog closed."""
        if self._save():
            self.accept()

    def _save(self) -> bool:
        """Write this column's name, type and description - one undo step; False, and told why, when it fails."""
        name = self.name.text().strip()
        if not name:
            show_message(self, "column_info.no_name")
            return False
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
            return False
        if failed:
            show_message(self, "column_info.cast_failed", count=failed, kind=new_type)
        if column in self._table_columns:
            self._table_columns[self._table_columns.index(column)] = name
        self._column = self.column_name = name
        self._saved = True
        return True
