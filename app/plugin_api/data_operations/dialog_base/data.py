"""What an operation reads and computes: the input series (validated and read), the result tables and their
formatting, and the job run here or on a worker thread."""
from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.data_source import parse_roles, resolve_role_column, row_value
from app.logs.logger import applogger
from app.plugin_api.charts.grids import pivot_to_grid
from app.plugin_api.data_operations.dialog_base.common import (
    OperationJob,
    ResultSeriesSpec,
    SeriesJob,
    SeriesOutcome,
    generated_table_name,
)
from app.plugin_api.data_operations import next_steps
from app.utils import report_html
from app.utils.coercion import coerce_axis, to_numbers, to_numeric_axis
from app.utils.i18n import _
from app.utils.series_validation import SeriesIssue, clean_xy, errors, validate_xy


def _xy_columns(frame: pd.DataFrame, roles: Mapping[str, Any]) -> tuple[str, str]:
    """The x and y columns of a series' query: the ones its roles name, else the first two numeric ones."""
    columns = [str(column) for column in frame.columns]
    numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    x_col = resolve_role_column(columns, roles, "x") or ""
    y_col = resolve_role_column(columns, roles, "y") or ""
    if x_col not in columns:
        x_col = numeric[0] if numeric else columns[0]
    if y_col not in columns:
        y_col = numeric[1] if len(numeric) > 1 else x_col
    return x_col, y_col


class DataMixin:
    """What an operation reads and computes. A part of DataOperationDialogBase."""

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input_xy(
        self,
        x: Any,
        y: Any,
        *,
        label: str = "",
        raise_on_error: bool = True,
        repairable: bool = False,
    ) -> list[SeriesIssue]:
        """Check one series against this operation's requirements.

        Warnings are logged and the caller continues; errors are logged and,
        by default, raised - so an operation that does not check the return
        value still cannot run on data that would give a wrong answer.

        Pass ``raise_on_error=False`` to collect problems across several series
        and report them together, which is what the multi-series dialogs want:
        one bad series should not hide the six good ones.
        """
        issues = validate_xy(
            x,
            y,
            minimum_points=self.INPUT_MINIMUM_POINTS,
            require_sorted_x=self.INPUT_REQUIRES_SORTED_X,
            require_unique_x=self.INPUT_REQUIRES_UNIQUE_X,
            require_uniform_x=self.INPUT_REQUIRES_UNIFORM_X,
            require_varying_y=self.INPUT_REQUIRES_VARYING_Y,
            repairable=repairable,
            label=label,
        )

        for issue in issues:
            if issue.severity == "warning":
                applogger.warning(
                    issue.message
                )

        blocking = errors(issues)
        if blocking and raise_on_error:
            # One message, not one per issue: a dialog that pops three boxes
            # for one bad series trains the user to dismiss them unread.
            applogger.error(
                " ".join(issue.message for issue in blocking),
                raise_error=True,
            )

        return issues

    def prepare_input_xy(
        self,
        x: Any,
        y: Any,
        *,
        label: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate, then return the series cleaned to this operation's needs.

        The repair follows the declared requirements: an operation that needs
        sorted x gets sorted x, one that needs unique x gets duplicates
        averaged, and every operation gets non-finite points dropped.  Anything
        changed is reported, because quietly discarding a user's points is its
        own kind of wrong answer.
        """
        # repairable=True: clean_xy below fixes exactly what the declared
        # requirements ask for, so a problem it is about to solve is a warning
        # rather than a reason to stop.
        self.validate_input_xy(x, y, label=label, repairable=True)

        x_clean, y_clean, report = clean_xy(
            x,
            y,
            sort_x=self.INPUT_REQUIRES_SORTED_X,
            merge_duplicate_x=self.INPUT_REQUIRES_UNIQUE_X,
        )

        if report.changed:
            prefix = f"{label}: " if label else ""
            applogger.warning(
                f"{prefix}{report.describe()}.",
            )

        return x_clean, y_clean

    def series_xy(self, row: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Run one selected series' SQL and return its x/y, cleaned.

        Byte-identical copies of this lived in the calculus, control-chart
        and peaks dialogs, and a fourth was about to be written for the root
        finder - see todo.txt P2-14. Every operation needs the same three
        steps and makes the same two decisions, so both live here:

        *Which columns.* The series' own roles first, since that is what the
        chart draws. A role naming a column the query no longer returns
        falls back to the first two numeric columns rather than raising: the
        query is edited far more often than the roles are, and the numbers
        are usually still there under different names.

        *What to do with the rows.* :meth:`prepare_input_xy`, which repairs
        exactly what this operation's INPUT_* flags declare it needs and
        reports whatever it changed.
        """
        frame, roles = self._series_frame(row)
        x_col, y_col = _xy_columns(frame, roles)
        return self.prepare_input_xy(
            self.numeric_x(frame[x_col], name),
            self.numeric_y(frame[y_col]),
            label=name,
        )

    def _series_frame(self, row: Any) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        """Run one series' SQL: its rows as a frame, and its roles. Raises when there is no query or no rows."""
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            raise ValueError("the series has no SQL query")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            raise ValueError("the series query returned no rows")
        return frame, parse_roles(row_value(row, "roles", default={}))

    def numeric_x(self, column: Any, name: str = "") -> np.ndarray:
        """Return an x column as floats, timestamps included, and remember.

        Not ``pd.to_numeric``: a timestamp column run through that becomes
        all-NaN, and the operation then reports "0 usable points" about a
        series of sixty perfectly good ones. Every operation but the
        outlier one did that to every dated series -
        ``app/utils/coercion`` was written for exactly this and only that
        one dialog was calling it.

        Whether the column *was* temporal is remembered against the series
        name, so :meth:`restore_temporal_x` can put the result back on the
        axis it came from rather than at x = 1 700 000 000.
        """
        if name:
            _coerced, is_temporal = coerce_axis(column)
            self._temporal_x_sources[name] = bool(is_temporal)
        return to_numeric_axis(column)

    def series_origin(self, row: Any) -> str:
        """Return a short, human caption for what shape one series' data is.

        Purely informational: no operation decides what it can compute from
        this string, only the roles it reads directly.  It exists so a user
        opening, say, the peaks or roots dialog on a series with a ``z`` role
        is told - before pressing anything - whether they are about to search
        a 2D curve, a 3D grid, an interpolated 3D surface, or a vector field.

        Deciding "on a grid" vs. "scattered" needs the actual data (a SQL
        query), which is why this is relatively expensive and is meant to be
        called only when the selection changes - see
        ``connect_common_signals`` - not on every redraw.
        """
        roles = parse_roles(row_value(row, "roles", default={}))
        has_z = bool(roles.get("z"))
        has_u = bool(roles.get("u"))
        has_v = bool(roles.get("v"))

        if has_u and has_v:
            # Informational only: nothing in this task actually computes
            # curl/divergence yet, this just tells the user what they see.
            return _("vector field (x, y, u, v)")

        if not has_z:
            return _("2D (x, y)")

        try:
            name = self._series_display_name(row)
            x, y, z = self.series_xyz(row, name)
            frame = pd.DataFrame({"x": x, "y": y, "z": z})
            grid = pivot_to_grid(frame)
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            # Any failure to read/pivot the data still leaves a true fact
            # standing: there is a z role, so this is 3D of some kind. Which
            # kind is a detail the label can afford to get wrong here - the
            # operation itself will raise a clear error if it matters.
            return _("3D scattered (x, y, z)")

        return _("3D on a grid (x, y, z)") if grid is not None else _("3D scattered (x, y, z)")

    def _update_origin_label(self) -> None:
        """Show ``series_origin`` for the current selection, if there is one.

        Wired next to ``mark_results_stale`` in ``connect_common_signals``:
        the same event (selection changed) that invalidates a preview is the
        one point where the caption needs recomputing.
        """
        selector = getattr(self, "series_selector", None)
        if selector is None or not hasattr(selector, "set_origin_text"):
            return
        rows = self.selected_series()
        if not rows:
            selector.set_origin_text("")
            return
        try:
            selector.set_origin_text(self.series_origin(rows[0]))
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to describe the selected series' data shape.")
            selector.set_origin_text("")

    def series_xyz(self, row: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one selected series' SQL and return its x/y/z, as parallel arrays.

        The scattered (not-yet-gridded) reading of a 3D series: one point per
        row, same idea as smoothing_dialog's ``_series_choice_from_row`` but
        centralised here so every 3D-aware operation reads its data the same
        way. x and y go through ``numeric_x``/``numeric_y`` (timestamp-aware,
        same as the 2D case); z is a computed/measured quantity, never a
        chart axis a result has to land back on, so it goes through
        ``to_numeric_axis`` directly.

        Raises a clear error when the series has no ``z`` role, or when that
        role no longer names a real column - callers are expected to have
        already decided (via ``series_origin`` or the roles themselves) that
        this series is 3D before calling this, not to find out from the
        exception.
        """
        frame, roles = self._series_frame(row)
        columns = [str(column) for column in frame.columns]
        z_col = str(roles.get("z") or "")
        if not z_col or z_col not in columns:
            raise ValueError("the series has no z role")

        x_col, y_col = _xy_columns(frame, roles)
        x_values = np.asarray(self.numeric_x(frame[x_col], name), dtype=float).reshape(-1)
        y_values = np.asarray(self.numeric_y(frame[y_col]), dtype=float).reshape(-1)
        z_values = np.asarray(to_numeric_axis(frame[z_col]), dtype=float).reshape(-1)
        return x_values, y_values, z_values

    def series_grid_xyz(
        self,
        row: Any,
        name: str,
        *,
        resolution: int = 120,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return ``(X, Y, Z, interpolated)`` grids for one 3D series.

        Preference order, and why it is this order:

        1. **Exact.** ``pivot_to_grid`` (``app/plugin_api/charts/grids.py``) recognises a
           complete Cartesian product of the distinct x and y values and
           returns it untouched - no smoothing, no invented values. This is
           always tried first and always preferred, because it is the only
           answer that is exactly what was measured.
        2. **Honest interpolation.** When the points are not a complete grid
           (which is the common case for anything sampled freehand, or
           jittered), a regular ``resolution`` x ``resolution`` grid is built
           over the data's own x/y span and filled with
           ``scipy.interpolate.griddata(..., method="linear")``. Linear
           interpolation inside the convex hull of the original points is a
           defensible estimate; *outside* it, ``griddata`` correctly refuses
           to guess and leaves ``NaN`` - which this function deliberately
           does not paper over. A caller that averages, contours or searches
           this grid has to skip those cells rather than be handed a number
           that was invented to fill a hole no data ever supported.

        Never a third option: there is no "reasonable default" for a cell
        with no support and no neighbours close enough to interpolate from -
        that would be fabricated precision, which is worse than leaving it
        as ``NaN`` and saying so via the returned ``interpolated`` flag.
        """
        x, y, z = self.series_xyz(row, name)
        frame = pd.DataFrame({"x": x, "y": y, "z": z})

        grid = pivot_to_grid(frame)
        if grid is not None:
            x_grid, y_grid, z_grid = grid
            return x_grid, y_grid, z_grid, False

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x_finite, y_finite, z_finite = x[finite], y[finite], z[finite]
        if x_finite.size < 4 or np.unique(x_finite).size < 2 or np.unique(y_finite).size < 2:
            raise ValueError("not enough points to interpolate a surface")

        x_lin = np.linspace(float(np.min(x_finite)), float(np.max(x_finite)), int(resolution))
        y_lin = np.linspace(float(np.min(y_finite)), float(np.max(y_finite)), int(resolution))
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)

        from scipy.interpolate import griddata  # here, not at import: 0.5 s of startup

        z_grid = griddata(
            (x_finite, y_finite),
            z_finite,
            (x_grid, y_grid),
            method="linear",
        )
        z_grid = np.asarray(z_grid, dtype=float)
        if not np.any(np.isfinite(z_grid)):
            raise ValueError("not enough points to interpolate a surface")

        return x_grid, y_grid, z_grid, True

    def numeric_y(self, column: Any) -> np.ndarray:
        """Return a y column as floats, timestamps included.

        Nothing to remember: a result's y is a computed quantity - a
        derivative, a residual, a count - and is written as the number it
        is even when the source was a duration or a time of day.
        """
        return to_numeric_axis(column)

    def result_to_frame(self, result: Any) -> Any:
        """Return the pandas DataFrame to save for one result: its to_df().

        ``to_frame()`` is still accepted from a user-written result that
        predates OperationResult.
        """
        for name in ("to_df", "to_frame"):
            method = getattr(result, name, None)
            if callable(method):
                return method()
        raise NotImplementedError(f"{type(result).__name__} has no to_df()")

    def result_style(self, result: Any, **overrides: Any) -> dict[str, Any]:
        """A generated series' style: this dialog's marks (generated_style_filter), the series and the model it
        came from, then *overrides*."""
        return {**self.generated_style_filter, "source_name": getattr(result, "source_name", ""),
                "model": getattr(result, "model", ""), **overrides}

    def curve_spec(self, table_name: str, result: Any, *, y: str = "y", name: str = "",
                   **style: Any) -> ResultSeriesSpec:
        """The result table's *y* column against x, in x order, as one series - named after the result unless
        *name* says otherwise, styled by result_style and *style*."""
        column = "y" if y == "y" else f"{y} AS y"
        return ResultSeriesSpec(
            name=name or result.result_name,
            sql_query=f'SELECT x, {column} FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style=self.result_style(result, **style),
        )

    def result_series_spec(self, axis_id: int, table_name: str, result: Any) -> ResultSeriesSpec:
        """The chart series of one saved result table: its y against x, drawn as RESULT_STYLE says."""
        del axis_id
        return self.curve_spec(table_name, result, **self.RESULT_STYLE)

    def result_series_specs(
        self,
        axis_id: int,
        table_name: str,
        result: Any,
    ) -> Sequence[ResultSeriesSpec]:
        """Return every series one result should draw.

        One by default, which is what almost every operation wants and what
        ``result_series_spec`` already answers. A control chart is the
        exception: its result table carries the plotted values, the centre
        line and the control limits, and drawing only the first of those
        leaves the chart without the lines that make it a control chart.

        Overriding this rather than calling create_series_descriptor directly
        keeps preview and apply on the same path - a spec listed here is
        previewed, cleaned up and applied exactly like any other.
        """
        return [self.result_series_spec(axis_id, table_name, result)]

    def result_table_name(self, axis_id: int, result: Any) -> str:
        """Return a safe table name for one result.

        From RESULT_TABLE_PREFIX/RESULT_TABLE_VARIANT when the operation sets
        a prefix; otherwise the class name and whatever model or method the
        result carries.
        """
        prefix = self.RESULT_TABLE_PREFIX
        if prefix:
            raw = f"{prefix}_axis{axis_id}_{result.source_name}"
            if self.RESULT_TABLE_VARIANT is not None:
                raw += f"_{getattr(result, self.RESULT_TABLE_VARIANT)}"
            return generated_table_name(raw, fallback=f"{prefix}_Result")
        source_name = str(getattr(result, "source_name", "Series") or "Series")
        # "model" on most results, "method" on the clustering one: a result
        # without either used to make Apply fail outright.
        variant = getattr(result, "model", None) or getattr(result, "method", None) or "result"
        raw = f"{self.__class__.__name__}_axis{axis_id}_{source_name}_{variant}"
        return generated_table_name(raw, fallback="Series_Result")

    def next_steps(self, result: Any) -> list[next_steps.NextStep]:
        """The questions this result leads to, as "What next?" links; none by default."""
        del result
        return []

    def what_next(self, result: Any) -> str:
        """The "What next?" section of *result*'s report, empty when turned off or there is none."""
        return next_steps.section_html(self.next_steps(result)) if next_steps.enabled() else ""

    def results_document(self, subtitle: str, *sections: str) -> str:
        """The Results pane's document, the same frame for every operation: its name, *subtitle*, *sections*."""
        return report_html.document(_(self.operation_label), subtitle, *sections)

    def format_results(self, results: Sequence[Any]) -> str:
        """The Results pane, framed by results_document: the RESULT_COLUMNS table, else a section per result.

        An operation says what it has to say through result_row or result_section; the frame, the empty
        case and the subtitle are the same for all.
        """
        if not results:
            return report_html.note(_("No results."))
        if self.RESULT_COLUMNS:
            body = report_html.section(_("Results"), report_html.table(
                tuple(_(heading) for heading, _align in self.RESULT_COLUMNS),
                [self.result_row(result) for result in results],
                align=tuple(align for _heading, align in self.RESULT_COLUMNS),
                empty_message=_("No results for this selection."),
            ))
            return self.results_document(self.results_subtitle(results), body)
        return self.results_document(self.results_subtitle(results),
                                     *(self.result_section(result) for result in results))

    def how_to_read(self, results: Sequence[Any]) -> list[str]:
        """The "How to read this" box (H-04): the numbers the report hinges on, each in two sentences
        (app.plugin_api.reading_guide); none by default."""
        del results
        return []

    def results_report(self, results: Sequence[Any]) -> str:
        """The Results pane: format_results, with the "How to read this" box under its first section."""
        from app.plugin_api import reading_guide

        report = self.format_results(results)
        if not report or not results or not reading_guide.enabled():
            return report
        try:
            box = report_html.reading_guide(_("How to read this"), self.how_to_read(results))
        except Exception:  # noqa: BLE001 - the explanation is never the reason a report is not shown
            applogger.exception("%s: could not explain the report.", type(self).__name__)
            return report
        return report_html.with_reading_guide(report, box)

    def results_subtitle(self, results: Sequence[Any]) -> str:
        """The line under the operation's name: the model chosen for a table of results, else the first's."""
        if self.RESULT_COLUMNS:
            return self.current_model()
        return str(getattr(results[0], "model_name", "") or "")

    def result_row(self, result: Any) -> Sequence[Any]:
        """The cells of *result*'s row in the RESULT_COLUMNS table (HTML)."""
        raise NotImplementedError

    def result_section(self, result: Any) -> str:
        """*result*'s section: its series, model and settings (OperationResult.to_html), and its metrics."""
        series = ", ".join(getattr(result, "series", ()) or ()) or str(getattr(result, "source_name", "") or "")
        body = result.to_html() if hasattr(result, "to_html") else report_html.note(str(result))
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, Mapping) and metrics:
            body += report_html.summary_table((str(key), value) for key, value in metrics.items())
        return report_html.section(series or _("Result"), body)

    @staticmethod
    def details(values: Mapping[str, Any], *, leave_out: Sequence[str] = ()) -> str:
        """``key: value, ...`` of *values* that say something - the Detail cell of a results table."""
        return ", ".join(f"{key}: {value}" for key, value in values.items()
                         if key not in leave_out and value not in (None, ""))

    def store_cached_results(self, results: Sequence[Any]) -> None:
        """Store latest results using the existing dialog convention."""
        if hasattr(self, "_last_results"):
            setattr(self, "_last_results", list(results))

    def write_result_table(self, table_name: str, result: Any) -> None:
        """Persist one operation result to a normal SQLite table."""
        frame = self.restore_temporal_x(self.result_to_frame(result), result)
        # An infinite value is no number to draw - a model overflowing far from its data (exp of a date in
        # seconds): written as missing, so the chart leaves it out instead of scaling to it.
        numeric = frame.select_dtypes(include="number").columns
        if len(numeric):
            frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        self._repo.import_dataframe(frame, table_name=table_name, normalize_columns=False)

    def restore_temporal_x(self, frame: Any, result: Any) -> Any:
        """Write a result's x back in the units its source was read in.

        An operation does its arithmetic on seconds since the epoch, because
        subtracting two timestamps is what every one of them needs. Writing
        that back would put the result on the chart at x = 1 700 000 000
        while the series it came from is drawn as dates - the right answer,
        plotted somewhere unreadable.

        So a result computed from a dated series gets dated x values back.
        The renderers read them the same way they read the source (see
        ``coercion.coerce_axis``), which is what puts the two on one axis.
        """
        # The series it was computed from: source_name, or the series every result names (a fit has
        # only that - its dated x was written as seconds, a chart of dates drawn at the year -215941).
        sources = {str(getattr(result, "source_name", "") or ""), *map(str, getattr(result, "series", ()) or ())}
        if not any(self._temporal_x_sources.get(source) for source in sources if source):
            return frame

        for column in ("x", "left_x", "right_x"):
            if column not in getattr(frame, "columns", []):
                continue
            try:
                frame[column] = pd.to_datetime(
                    to_numbers(frame[column]),
                    unit="s",
                    errors="coerce",
                )
            except (TypeError, ValueError):
                applogger.warning(
                    "Could not write %s back as timestamps; leaving it numeric.",
                    column,
                )
        return frame

    def _remove_series_styled(self, axis_id: int, expected: Mapping[str, Any]) -> int:
        """Delete the series of *axis_id* whose style has every key/value of *expected*; return how many."""
        removed = 0
        for row in self._repo.get_series(axis_id) or []:
            try:
                style = json.loads(str(row["style_json"] or "{}"))
            except (TypeError, ValueError):
                style = {}
            if all(style.get(key) == value for key, value in expected.items()):
                self._repo.delete_series(int(row["id"]))
                removed += 1
        return removed

    def remove_previous_generated_series(self, axis_id: int) -> int:
        """Delete generated descriptors for this operation on the selected axis."""
        removed = self._remove_series_styled(axis_id, self.generated_style_filter)
        if removed:
            applogger.info(f"Removed {removed} previous {self.operation_label} generated series.")
        return removed

    def create_result_series(self, axis_id: int, table_name: str, result: Any) -> int | None:
        """Create every chart-series descriptor for a saved result table.

        Returns the first series id, for callers that only ever make one.
        """
        first_id: int | None = None
        for spec in self.result_series_specs(axis_id, table_name, result):
            series_id = self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=spec.name,
                sql_query=spec.sql_query,
                roles=dict(spec.roles),
                style=dict(spec.style),
            )
            applogger.info(f"Created series: {spec.name}")
            if first_id is None and series_id is not None:
                first_id = int(series_id)
        return first_id

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """Return the axis the results should be written to.

        The selected axis by default: a smoothed series belongs next to the
        series it smoothed.  A transform that changes the meaning of the x
        axis - a spectrum's x is frequency, not the source's x - overrides this
        to build a chart of its own, because putting frequencies on a time axis
        produces a picture that is simply wrong.
        """
        del results
        if not self.DESTINATIONS:
            return selected_axis_id
        # "Draw on" (the Charts tab): a new axis or figure of the source axis' own kind.
        descriptor = self._repo.load_figure_descriptor(self._figure_id)
        chart_type = next((axis.name for axis in (descriptor.axes if descriptor else [])
                           if int(axis.id) == int(selected_axis_id)), "Scatter Plot")
        return self.resolve_destination_axis(selected_axis_id, chart_type=chart_type, title=self.operation_label)

    def create_result_axis(
        self,
        *,
        chart_type: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        options: Mapping[str, Any] | None = None,
        figure_id: int | None = None,
    ) -> int:
        """Add an axis to the current figure and return its id.

        For operations whose output cannot share the source's axes.  A spectrum
        turns a time axis into a frequency one, and a box plot of a sample has
        no x in common with the series it summarises - putting either on top of
        its input produces a picture that is simply wrong.

        A new axis rather than a new figure: the result belongs beside what it
        was computed from, in the same tab, and a chart tab per estimate would
        bury the original.  The figure's grid grows by one row to make room,
        because an axis outside the grid is never drawn.
        """
        figure_id = self._figure_id if figure_id is None else int(figure_id)
        axis_index = self._repo.next_axis_index(figure_id)

        descriptor = self._repo.load_figure_descriptor(figure_id=figure_id)
        rows = int(getattr(descriptor, "nrows", 1) or 1)
        cols = int(getattr(descriptor, "ncols", 1) or 1)
        if axis_index >= rows * cols:
            self._repo.set_figure_grid(figure_id, nrows=axis_index + 1, ncols=cols)

        axis_id = int(
            self._repo.create_axis_descriptor(
                figure_id=figure_id,
                axis_index=axis_index,
                chart_type=chart_type,
                title=title,
                x_label=x_label,
                y_label=y_label,
                options=dict(options or {"grid": True}),
            )
        )
        applogger.info(
            "%s: created axis %s (%s) on figure %s.",
            self.operation_label,
            axis_id,
            chart_type,
            figure_id,
        )
        return axis_id

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def selected_series(self) -> list[Any]:
        if self.reads_table() and not getattr(self, "_switching_figure", False):
            # The table's columns, drawn if they are not yet.
            try:
                self._prepare_table_source()
            except ValueError:
                return []
        return self.series_selector.selected_series()

    def _selected_series_row(self) -> Any:
        rows = self.selected_series()

        if not rows:
            message = "Select one source series in the Data panel."
            applogger.error(message)
            raise ValueError(message)

        if len(rows) > 1:
            # A warning, not a message box: this runs on every Preview and
            # every parameter tweak while two or more series are checked.
            # The status bar shows it without asking to be dismissed.
            name = row_value(rows[0], "name", default="first")
            applogger.warning(
                "%d series are selected; using only '%s'. Uncheck the "
                "others, or reorder them, to use a different one.",
                len(rows),
                name,
            )

        return rows[0]

    def set_results_text(self, text: str) -> None:
        """Show results in the results pane: markup rendered, plain text escaped (OperationShell.show_results)."""
        self.show_results(text)

    def results_widget(self) -> Any:
        """The Results pane, for OperationShell."""
        return self._results_label

    def set_results_html(self, markup: str) -> None:
        """Show markup in the results pane."""
        self.show_results(str(markup or ""))

    def publish_results(self, formatted: str) -> None:
        """Show ``format_results`` output: markup rendered, plain text escaped (OperationShell.show_results)."""
        self.show_results(formatted)

    def mark_results_stale(self, *_ignored: Any) -> None:
        """Invalidate any previewed results without recomputing anything.

        Wired wherever a control used to call refresh_results()/
        _queue_refresh_results() straight away. Preview and OK are the only
        paths left that actually run compute_results().
        """
        self.store_cached_results([])
        self.publish_results(_(self.PENDING_RESULTS_MESSAGE))

    # ------------------------------------------------------------------
    # Shared generated-series apply pipeline
    # ------------------------------------------------------------------
    @property
    def operation_label(self) -> str:
        """Human readable operation name used in messages/logs: the operation's Name."""
        return self.Name or self.windowTitle() or self.__class__.__name__

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        """Style key/value pairs identifying series generated by this dialog.

        Example::

            {"generated_smoothing": True, "smoothing_dialog": "series_smoothing"}

        ``remove_previous_generated_series`` deletes only descriptors whose
        ``style_json`` contains all these key/value pairs.

        Derived from the source file's name by default - smoothing_dialog.py
        gives the example above; an operation that is a package
        (fit_dialog/dialog.py) takes its folder's name. The file, not the
        module: the scanner loads a plugin under a module name of its own,
        different every session (_dynamic_data_operation_..._<hash>), and
        keys made from that were neither stable nor the ones the series were
        stored with. Four operations predate the rule and keep their own
        values, which are stored in existing projects.
        """
        path = Path(inspect.getfile(type(self)))
        name = path.stem if path.stem.endswith("_dialog") else path.parent.name
        stem = name.removesuffix("_dialog")
        return {f"generated_{stem}": True, f"{stem}_dialog": f"series_{stem}"}

    def compute_results(self) -> Sequence[Any]:
        """Return operation-specific result objects for the selected series.

        Preview and Apply call this, on the GUI thread, unless the operation
        runs in the background (see evaluate). An operation that implements
        prepare_job/finish_job - or, more simply, compute_series - gets it for
        free; one that has not been split that way yet overrides this instead.
        """
        job = self.prepare_job()
        if job is None:
            return []
        return self.finish_job(job, job.run())

    # ------------------------------------------------------------------
    # Evaluation: one way to compute, here or on a worker thread
    # ------------------------------------------------------------------

    # -- Per series: the common case ------------------------------------

    def series_settings(self) -> Any:
        """Everything compute_series needs besides the data, read on the GUI thread.

        ``(model, parameters)`` for an operation with MODELS, else the parameters.
        """
        if self.MODELS:
            return self.current_model(), self.parameter_values()
        return self.parameter_values()

    def read_series(self, row: Any, name: str) -> Any:
        """One selected series' data, read on the GUI thread; x and y by default."""
        if self.READS_SURFACES:
            return self.read_curve_or_surface(row, name)
        return self.series_xy(row, name)

    def read_curve_or_surface(self, row: Any, name: str) -> tuple[bool, Any]:
        """``(True, series_grid_xyz)`` for a series with a z role - a surface -
        else ``(False, series_xy)``: for the operations that handle both."""
        if parse_roles(row_value(row, "roles", default={})).get("z"):
            return True, self.series_grid_xyz(row, name)
        return False, self.series_xy(row, name)

    def compute_series(self, name: str, data: Any, settings: Any) -> Any:
        """The calculation for one series, on whatever thread runs the job.

        Must not touch a widget, the repository or the logger's dialogs:
        everything it needs arrives in *data* and *settings*. Whatever it
        returns goes to finish_series, back on the GUI thread.
        """
        raise NotImplementedError

    def finish_series(self, name: str, outcome: Any, settings: Any) -> Any:
        """Turn compute_series' outcome into a result, on the GUI thread (notes go to the log here)."""
        del name, settings
        return outcome

    def _computes_per_series(self) -> bool:
        from app.plugin_api.data_operations.dialog_base.dialog import (
            DataOperationDialogBase,  # the class this is part of
        )

        return type(self).compute_series is not DataOperationDialogBase.compute_series

    def prepare_job(self, **options: Any) -> OperationJob | None:
        """Read the widgets into a job, on the GUI thread; None when there is nothing to do.

        The job carries everything the calculation needs - data, parameters,
        and whatever names the result will need later - so the window can be
        changed while it runs without mixing two runs. *options* are the
        operation's own (the fit's ``optimise``, for one).

        An operation with compute_series gets a SeriesJob: the selected
        series are read here, one by one, and a series that cannot be read
        is reported with the ones that fail to compute.
        """
        if not self._computes_per_series():
            raise NotImplementedError(
                f"{type(self).__name__} implements neither prepare_job nor compute_results"
            )
        del options
        settings = self.series_settings()
        inputs: list[tuple[str, Any]] = []
        errors: list[str] = []
        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                inputs.append((name, self.read_series(row, name)))
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")
        if not inputs and not errors:
            return None
        return SeriesJob(inputs, settings, self.compute_series, errors)

    def finish_job(self, job: OperationJob, outcome: Any) -> Sequence[Any]:
        """Turn a job's outcome into this dialog's results, on the GUI thread."""
        if isinstance(job, SeriesJob) and isinstance(outcome, SeriesOutcome):
            results = [self.finish_series(name, value, job.settings) for name, value in outcome.outcomes]
            # One bad series should not hide the good ones; only when none
            # worked is it an error.
            if outcome.errors and not results:
                raise ValueError("; ".join(outcome.errors))
            for message in outcome.errors:
                applogger.warning(message)
            return results
        del job
        return list(outcome) if isinstance(outcome, (list, tuple)) else [outcome]

    @property
    def evaluating(self) -> bool:
        """A calculation is running (OperationShell.job_running)."""
        return self.job_running

    def evaluate(
        self,
        on_results: Callable[[Sequence[Any]], None],
        *,
        background: bool | None = None,
        **options: Any,
    ) -> None:
        """Compute the results and hand them to *on_results*, on the GUI thread.

        In the background when *background* says so (RUN_IN_BACKGROUND when it does not), through
        OperationShell.run_job: the series are counted on the progress bar, Stop is polled between them.
        *on_results* is not called when the run is stopped or fails; the reason is shown instead.
        """
        if self.job_running:
            return
        job = self.prepare_job(**options)
        if job is None:
            return
        if not (self.RUN_IN_BACKGROUND if background is None else background):
            on_results(self.finish_job(job, job.run()))
            return
        self.run_job(lambda cancel: job.run(should_stop=cancel.is_set),
                     lambda outcome: on_results(self.finish_job(job, outcome)),
                     progress=getattr(job, "progress", None))
