"""How a table came to be: the operations recorded in the project (todo R-03).

Every Apply of a series operation is kept in ``__operations__`` (see
``app/data/repo/operations.py``); this shows the ones that read or wrote one
table - or, without a table, every one in the project - oldest first - what was run, on which series, with which entries, and
the report it gave - in the house report style, so it can be copied like
any other report.
"""
from __future__ import annotations

import html
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from PySide6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from app.data.repo.operations import OperationRecord
from app.data.sqlite_repo import SqliteRepo
from app.styles.style import (
    CardFrame,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
from app.utils import report_html
from app.utils.i18n import _
from app.widgets.html_results import HtmlResultsView, looks_like_html, plain_to_html


class OperationHistoryDialog(QDialog):
    """The recorded operations that read or wrote *table*; all of them when None."""

    def __init__(self, repo: SqliteRepo, table: str | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("History of {table}").format(table=table) if table else _("Project history"))
        self.setWindowIcon(load_icon("operation_history"))

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")

        card = CardFrame(self, "operationHistoryCard")
        card_layout = card.layout()
        card_layout.addWidget(create_section_title(_("History"), card))
        self._view = HtmlResultsView(card)
        self._view.setContent(history_html(table, repo.operations(table=table)))
        card_layout.addWidget(self._view, 1)
        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(parent=self, action_id="dismiss", action=self.reject, layout=action_row)
        root.addLayout(action_row, 0)


def _local_time(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stamp


def _value(value: Any) -> str:
    if isinstance(value, float):
        return report_html.format_number(value)
    if isinstance(value, (list, tuple)):
        return html.escape(", ".join(str(item) for item in value))
    return html.escape(str(value))


def _record_section(record: OperationRecord) -> str:
    summary = report_html.summary_table(
        [
            (_("Applied"), _local_time(record.applied_at)),
            (_("Read"), ", ".join(str(source.get("name", "")) for source in record.sources)),
            (_("Wrote"), ", ".join(str(result.get("table", "")) for result in record.results)),
            (_("ChartLibre version"), record.app_version),
        ]
    )
    # The declared parameters when the operation has them; otherwise its
    # other entries (a model combo, a checkbox) are what it was run with.
    settings = record.parameters or record.entries
    parameters = report_html.table(
        [_("Setting"), _("Value")],
        [(html.escape(str(name)), _value(value)) for name, value in settings.items()],
        align=["left", "left"],
        empty_message=_("No settings recorded."),
    )
    report = ""
    if record.report.strip():
        report = report_html.raw_note(
            record.report if looks_like_html(record.report) else plain_to_html(record.report)
        )
    return report_html.section(f"{record.id}. {record.operation}", summary, parameters, report)


def history_html(table: str | None, records: Sequence[OperationRecord]) -> str:
    """The history page for *table*, or for the whole project when None."""
    title = _("History of {table}").format(table=table) if table else _("Project history")
    if not records:
        body = report_html.note(
            _("No operation has been applied to or from this table since the project began recording them.")
            if table else _("No operation has been applied in this project since it began recording them.")
        )
        return report_html.document(title, "", body)
    return report_html.document(
        title,
        _("{count} operation(s), oldest first").format(count=len(records)),
        *(_record_section(record) for record in records),
    )
