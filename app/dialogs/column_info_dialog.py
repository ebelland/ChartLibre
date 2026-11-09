"""Column Info: one column's name, type and description, as JMP's Column Info dialog (R-08).

Opened from a column's context menu in the table preview. What it changes:

* the name (the column renamed, its description following it);
* the data type - REAL, INTEGER, TEXT - its values converted (Cast);
* the modelling type forced - Continuous, Ordinal, Nominal - which every
  analysis then takes for the column, whatever its values suggest;
* label, unit, notes;

The description is kept with the table (app.data.column_metadata); OK is
one undo step for all of it. The arrows at the top go to the previous or next
column without closing (P-01): the changes to this one are saved first, as OK
saves them.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
)

from app.data import column_metadata as cm
from app.data.repo.project import CAST_TYPES
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
#: The columns the application keeps on a table for itself: the arrows pass them by.
APPLICATION_COLUMNS = frozenset({"Hide", "Selected", "_id"})


class ColumnInfoDialog(ColumnsDialog):
    """Edit one column of *table*: its name, types and description."""

    def __init__(self, repo: Any, table: str, column: str, parent: QWidget | None = None) -> None:
        # Two columns, the column's description and its values side by side: one tall column was taller than
        # a laptop's screen.
        super().__init__(parent, title=_("Column Info: {column}").format(column=column), size=None,
                         columns=(Column(),))  # one column: the Values card is gone
        self._repo, self._table, self._column = repo, table, column
        #: Whether a column was saved while moving between them: Close then says so, as OK would.
        self._saved = False
        #: The table's columns in order, the application's own left out: the arrows go through them.
        # A saved query or a stored function is described as a table is, but its columns are its SQL's or its
        # formula's: their names and types are read, not changed here (_read_only).
        source = repo.get_data_source(table)
        self._read_only = bool(source is not None and source.is_query)
        self._from = source.from_clause() if self._read_only else '"' + table.replace('"', '""') + '"'
        if self._read_only:
            self._table_columns = [str(name) for name in repo.query_df(f"SELECT * FROM {self._from} LIMIT 0").columns]
        else:
            self._table_columns = [str(row[1]) for row in repo.table_info(table)
                                   if str(row[1]) not in APPLICATION_COLUMNS]

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
        self._values = repo.query_df(f'SELECT "{column.replace(chr(34), chr(34) * 2)}" AS v FROM {self._from}')["v"]
        if self._read_only:
            # Its type as the query returns it: SQLite's REAL, INTEGER or TEXT.
            dtype = self._values.dtype
            self._declared = ("INTEGER" if pd.api.types.is_integer_dtype(dtype) else
                              "REAL" if pd.api.types.is_float_dtype(dtype) else "TEXT")
        else:
            declared = {str(row[1]): str(row[2] or "").upper() for row in repo.table_info(table)}
            self._declared = declared.get(column, "")
        info = self._info
        self.name.setText(column)
        for field in (self.name, self.data_type):
            field.setEnabled(not self._read_only)
            if self._read_only:
                field.setToolTip(_("A query's or a function's column: named and typed by its SQL or its formula."))
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

    def info(self) -> cm.ColumnInfo:
        """The description as the dialog holds it now."""
        return cm.ColumnInfo(label=self.label.text().strip(), unit=self.unit.text().strip(),
                             notes=self.notes.toPlainText().strip(), kind=str(self.modelling.currentData() or ""),
                             format=str(self.format.currentData() or ""),
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
        if self._read_only:
            # A query's or a function's column: its description only - the name and the type are its SQL's.
            name, new_type = column, self._declared
        try:
            entry = repo.snapshot_for_undo(([] if self._read_only else [table]) + ["__table_descriptors__"],
                                           label=f"Column Info of '{column}' in '{table}'")
            failed = 0
            if new_type != self._declared and new_type in CAST_TYPES:
                failed = repo.cast_column(table, column, new_type, undo_entry=entry)
            if name != column:
                repo.rename_table_column(table, column, name)
                cm.rename_column_info(repo, table, column, name)
            cm.set_column_info(repo, table, name, self.info())
        except Exception as exc:
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
