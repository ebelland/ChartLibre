"""What a series operation hands back: one result per source series.

Every operation's result class derives from :class:`OperationResult`, which
says what any result can be asked:

* ``model_name``, ``parameters`` and ``series`` - which model ran, with what
  settings, on which series. The defaults read the fields most results
  already have (``model``, ``metadata``, ``source_name``); a result that
  names them otherwise overrides the property.
* ``to_df()`` - the numbers, as the table Apply saves.
* ``to_html()`` - a short report of the three above.
* ``preview(dialog, axis_id)`` and ``apply(dialog, axis_id)`` - put the
  result on the chart, temporarily or for good.

Almost every result is a table plus the series that read it; that is
:class:`TableResult`, which implements the last two once. A result that is
drawn some other way - Geometry's query, Statistics' report - implements
them itself, and the dialog base no longer needs to know which is which.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pandas as pd

from app.logs.logger import applogger
from app.utils import report_html
from app.utils.i18n import _

if TYPE_CHECKING:
    from app.series_operations.dialog_base import SeriesOperationDialogBase


class OperationResult(ABC):
    """One result of one series operation."""

    # Results are slotted dataclasses; an empty __slots__ here keeps them so.
    __slots__ = ()

    @property
    def model_name(self) -> str:
        """The model or method that produced this result."""
        return str(getattr(self, "model", None) or getattr(self, "method", None) or "")

    @property
    def parameters(self) -> dict[str, Any]:
        """The settings the model ran with, as shown in reports."""
        return dict(getattr(self, "metadata", None) or {})

    @property
    def series(self) -> tuple[str, ...]:
        """The names of the series this result was computed from."""
        name = getattr(self, "source_name", None)
        return (str(name),) if name else ()

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        """The result as the table Apply saves."""

    def to_html(self) -> str:
        """A summary: model, series and parameters, in the house style."""
        rows: list[tuple[str, Any]] = [
            (_("Model"), self.model_name),
            (_("Series"), ", ".join(self.series)),
        ]
        rows.extend((str(key), value) for key, value in self.parameters.items())
        return report_html.summary_table(rows)

    @abstractmethod
    def preview(self, dialog: SeriesOperationDialogBase, axis_id: int) -> None:
        """Draw this result on *axis_id* until the dialog closes or applies."""

    @abstractmethod
    def apply(self, dialog: SeriesOperationDialogBase, axis_id: int) -> None:
        """Save this result and draw it on *axis_id* for good."""


class TableResult(OperationResult):
    """A result saved as a table, drawn by the series that read it."""

    __slots__ = ()

    def preview(self, dialog: SeriesOperationDialogBase, axis_id: int) -> None:
        table_name = dialog.preview_table_name(axis_id, self)
        dialog.write_result_table(table_name, self)
        dialog.remember_preview_table(table_name)
        dialog.create_preview_series(axis_id, table_name, self)

    def apply(self, dialog: SeriesOperationDialogBase, axis_id: int) -> None:
        table_name = dialog.result_table_name(axis_id, self)
        dialog.write_result_table(table_name, self)
        applogger.info(f"Saved result table: {table_name}")
        dialog.create_result_series(axis_id, table_name, self)
