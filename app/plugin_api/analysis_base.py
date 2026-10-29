"""The base of every analysis window, whichever data it reads.

A data operation reads a chart's series (or a table's columns chosen as series); a table analysis reads a
table's columns cast into roles. Both are the same window (OperationShell); DATA_PICKER says which kind of
data picker the window has, so the two bases can be told apart by a declaration rather than by their names.
"""
from __future__ import annotations

from typing import ClassVar

from app.plugin_api.operation_shell import OperationShell


class AnalysisDialogBase(OperationShell):
    """The window of an analysis: its shell, and the kind of data it picks."""

    #: "series": the chart's series, or the table's columns chosen as series (data operations).
    #: "roles": the table's columns cast into roles (table analyses).
    DATA_PICKER: ClassVar[str] = ""
