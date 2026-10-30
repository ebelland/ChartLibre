"""The base of every analysis window, whichever data it reads.

A data operation reads a chart's series (or a table's columns chosen as series); a table analysis reads a
table's columns cast into roles. Both are the same window (OperationShell); DATA_PICKER says which kind of
data picker the window has, so the two bases can be told apart by a declaration rather than by their names.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

from app.logs.logger import applogger
from app.plugin_api.operation_shell import OperationShell


class AnalysisDialogBase(OperationShell):
    """The window of an analysis: its shell, and the kind of data it picks."""

    #: "series": the chart's series, or the table's columns chosen as series (data operations).
    #: "roles": the table's columns cast into roles (table analyses).
    DATA_PICKER: ClassVar[str] = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: The figures the last Preview or OK drew (create_report_figure keeps it).
        self.created_figure_ids: list[int] = []
        #: The figure the main window shows, for "Draw on: new axis in the current figure"; None: no such choice.
        self.current_figure_id: int | None = None

    def begin_preview_transaction(self, name: str | None = None) -> None:
        """Open the SAVEPOINT that makes a Preview undoable.

        Why a database savepoint and not just artifact bookkeeping: removing the
        preview series and temp tables only undoes what the *generated-table*
        operations create.  Clustering writes a ClusterId column into the user's
        own source table and rewrites the selected series' SQL, and outlier
        removal flips hide flags on real rows - none of which the artifact
        cleanup can reach.  With every preview write inside one savepoint,
        Close rolls back whatever the operation did, whatever that was.

        SqliteRepo._commit() suppresses commits while the savepoint is open, so
        repository helpers that normally commit after each write stay inside it.
        """
        if self._repo.preview_savepoint_active:
            return
        self._repo.begin_preview_savepoint(name or f"preview_{self.__class__.__name__}")

    def rollback_preview_transaction(self) -> bool:
        """Undo every database change made since the preview started."""
        if not self._repo.preview_savepoint_active:
            return False
        try:
            return bool(self._repo.rollback_preview_savepoint())
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to roll back the preview savepoint")
            return False

    def commit_preview_transaction(self) -> None:
        """Make the preview changes permanent."""
        if not self._repo.preview_savepoint_active:
            return
        try:
            self._repo.release_preview_savepoint()
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to commit the preview savepoint")

    def draw_on(self) -> str:
        """The Charts tab's choice; a new figure for one in the current figure when there is none to add to."""
        combo = getattr(self, "destination_combo", None)
        chosen = str(combo.currentData() or self.DEST_NEW_FIGURE) if combo is not None else self.DEST_NEW_FIGURE
        if chosen in (self.DEST_NEW_AXIS, self.DEST_SAME_AXIS) and getattr(self, "current_figure_id", None) is None:
            return self.DEST_NEW_FIGURE
        return chosen

    def create_report_figure(self, name: str, notes_html: str, panels: Sequence[tuple]) -> int | None:
        """A figure of *panels*, two to a row, with *notes_html* in its notes; its id.

        Each panel is ``(chart type, title, x label, y label, series)``, and
        optionally the axis options last (annotations, reference lines); each
        series ``(name, sql, roles, style)`` - see series_select.
        """
        from app.scanners.axis_renderer_scanner import get_renderer
        from app.widgets.chart_panel import FIGURE_VIEW_OPTIONS_KEY

        # Chart types are plugins that can be removed: a panel whose chart type
        # is not installed is left out (and logged) - the report is still made.
        available = []
        for panel in panels:
            if get_renderer(str(panel[0])) is None:
                applogger.warning(
                    "%s: the chart type %r is not installed; the panel %r is left out.",
                    self.Name, panel[0], panel[1],
                )
                continue
            available.append(panel)
        # The panels whose chart type is cleared in the Charts tab are left out.
        panels = [panel for panel in available if self.chart_enabled(str(panel[0]))]

        repo = self._repo
        draw_on = self.draw_on()
        if draw_on == self.DEST_NONE or not panels:
            # The report is the result - no chart asked for, or every box cleared: no empty figure; the report
            # stays in Results and in the project's history.
            return None
        target = getattr(self, "current_figure_id", None) if draw_on != self.DEST_NEW_FIGURE else None
        descriptor = repo.load_figure_descriptor(int(target)) if target is not None else None
        onto = next((axis for axis in (descriptor.axes if descriptor else []) if axis.name == panels[0][0]), None) \
            if draw_on == self.DEST_SAME_AXIS and len(panels) == 1 else None
        if onto is not None:
            # Current axis: the one panel's series added to the figure's axis of its kind.
            for position, (series_name, sql, roles, style) in enumerate(panels[0][4], start=len(onto.series)):
                repo.create_series_descriptor(axis_id=int(onto.id), series_index=position, name=series_name,
                                              sql_query=sql, roles=roles, style=style)
            self.created_figure_ids.append(int(target))
            return int(target)
        if descriptor is not None:
            # New axes after the figure's own, its grid grown to show them.
            figure_id = int(target)
            columns = max(1, int(descriptor.ncols or 1))
            first = max((int(axis.index) for axis in descriptor.axes), default=-1) + 1
            rows = max(int(descriptor.nrows or 1), math.ceil((first + len(panels)) / columns))
            repo.set_figure_grid(figure_id, nrows=rows, ncols=columns)
        else:
            first = 0
            columns = 2 if len(panels) > 1 else 1
            rows = max(1, math.ceil(len(panels) / columns))
            figure_id = int(repo.create_figure_descriptor(
                name=name, nrows=rows, ncols=columns,
                options={FIGURE_VIEW_OPTIONS_KEY: {"notes_html": notes_html}},
            ))
        for index, (chart, title, x_label, y_label, series, *options) in enumerate(panels, start=first):
            axis_id = int(repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=index, chart_type=chart,
                title=title, x_label=x_label, y_label=y_label, options=dict(options[0]) if options else {},
            ))
            for position, (series_name, sql, roles, style) in enumerate(series):
                repo.create_series_descriptor(
                    axis_id=axis_id, series_index=position, name=series_name,
                    sql_query=sql, roles=roles, style=style,
                )
        self.created_figure_ids.append(figure_id)
        return figure_id
