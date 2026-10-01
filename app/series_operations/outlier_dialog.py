"""Dialog for outlier detection: hide the outliers, or colour them.

The dialog only owns UI, preview and numeric outlier detection.  All database
management is delegated to SqliteRepo:
- creating/resetting the Hide column
- marking rows as hidden
- updating series SQL with the Hide filter
- or, to colour instead, writing a colour column and exposing it through the
  series' "color" projection (which the scatter renderer draws per point)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QFormLayout,
    QWidget,
)
from app.analysis.outliers import (
    OUTLIER_ELLIPTIC_ENVELOPE,
    OUTLIER_IQR,
    OUTLIER_ISOLATION_FOREST,
    OUTLIER_LOCAL_OUTLIER_FACTOR,
    OUTLIER_MAD,
    OUTLIER_ONE_CLASS_SVM,
    OUTLIER_ROLLING,
    OUTLIER_ZSCORE,
    OutlierSettings,
    outlier_mask,
)
from app.data.data_source import parse_roles, row_value
from app.data.select_sql import has_projection_alias, sql_insert_select_expression
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.parameter_spec import ChoiceParam, ColorParam, FloatParam, IntParam
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.logs.logger import applogger
from app.utils.coercion import to_numeric_axis

# Fewer than this and every estimator here is meaningless: a median absolute
# deviation over two points says nothing about either of them.
MIN_POINTS: int = 3
from app.utils.messages import show_message
from app.utils.i18n import _


# The method names are the engine's (app.analysis.outliers).

#: What is done with the outliers found.
ACTION_HIDE = "hide"
ACTION_COLOUR = "colour"

#: The column that holds the colour of each point in the source table, and
#: the projection that hands it to the renderer.
COLOUR_COLUMN = "OutlierColor"
COLOUR_PROJECTION = f'"{COLOUR_COLUMN}" AS "color"'


@dataclass(frozen=True, slots=True, kw_only=True)
class OutlierModel(OperationModel):
    #: The first four methods test y alone - a residual, a z-score, a
    #: spread. The shape-aware ones (scikit-learn) fit the joint (x, y)
    #: point cloud as a 2D feature space, which is what lets them catch an
    #: outlier a pure-y test misses - a point at a perfectly ordinary y value
    #: but far from the curve everything else traces at that x.
    shape_aware: bool = False


#: The methods offered, in combo order.
OUTLIER_METHODS: dict[str, OutlierModel] = {
    OUTLIER_ZSCORE: OutlierModel(
        doc_title="Z-score outlier detection",
        doc_url="https://en.wikipedia.org/wiki/Standard_score",
    ),
    OUTLIER_IQR: OutlierModel(
        doc_title="IQR outlier detection",
        doc_url="https://en.wikipedia.org/wiki/Interquartile_range",
    ),
    OUTLIER_MAD: OutlierModel(
        doc_title="Median absolute deviation",
        doc_url="https://en.wikipedia.org/wiki/Median_absolute_deviation",
    ),
    OUTLIER_ROLLING: OutlierModel(
        doc_title="Rolling median residuals",
        doc_url="https://en.wikipedia.org/wiki/Moving_average#Median_filter",
    ),
    OUTLIER_ISOLATION_FOREST: OutlierModel(
        doc_title="Isolation Forest",
        doc_url="https://en.wikipedia.org/wiki/Isolation_forest",
        shape_aware=True,
    ),
    OUTLIER_LOCAL_OUTLIER_FACTOR: OutlierModel(
        doc_title="Local Outlier Factor",
        doc_url="https://en.wikipedia.org/wiki/Local_outlier_factor",
        shape_aware=True,
    ),
    OUTLIER_ONE_CLASS_SVM: OutlierModel(
        doc_title="One-Class SVM",
        doc_url="https://scikit-learn.org/stable/modules/outlier_detection.html#one-class-svm",
        shape_aware=True,
    ),
    OUTLIER_ELLIPTIC_ENVELOPE: OutlierModel(
        doc_title="Elliptic Envelope (Minimum Covariance Determinant)",
        doc_url="https://scikit-learn.org/stable/modules/outlier_detection.html#fitting-an-elliptic-envelope",
        shape_aware=True,
    ),
}

@dataclass(slots=True)
class OutlierResult(TableResult):
    """Outlier detection result for one source series."""

    source_name: str
    result_name: str
    model: str
    source_table: str
    x_col: str
    y_col: str
    x: np.ndarray
    y: np.ndarray
    outlier_x: np.ndarray
    outlier_y: np.ndarray
    outlier_rowids: np.ndarray
    metadata: dict[str, Any]
    outlier_count: int
    message: str
    #: Every source row the series draws: what a re-run shows again before
    #: hiding its own outliers.
    rowids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def to_df(self) -> pd.DataFrame:
        """Return a small preview frame; not used for persistence."""
        return pd.DataFrame(
            {
                "source_name": self.source_name,
                "model": self.model,
                "x": self.x,
                "y": self.y,
                "hide": False,
            }
        )

class SeriesOutlierDialog(SeriesOperationDialogBase):
    """Dialog to detect outliers and mark source rows with Hide=True."""

    MODELS = OUTLIER_METHODS
    MODEL_TOOLTIP = "Choose the outlier detection method."
    Name: str = "Outliers"
    Description = "Detect anomalies"

    # The rolling-median detectors walk the series in order; an unsorted x
    # makes every window span an arbitrary set of points, so the residuals
    # it flags are not the outliers.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_MINIMUM_POINTS = 3

    # Replaces the three hand-built spin boxes, their three signal
    # connections, and _refresh_visibility's three set_row_visible calls. The
    # visibility rules that used to live in code are now the same data the
    # widgets are built from, so the two cannot drift apart.
    PARAMS = (
        ChoiceParam(
            "action",
            "Outliers:",
            tooltip=(
                "Hide takes the outliers off the chart (they stay in the table, "
                "flagged Hide). Colour keeps them on the chart and draws them "
                "in the colour below."
            ),
            choices=(("Hide them", ACTION_HIDE), ("Colour them", ACTION_COLOUR)),
            default_value=ACTION_HIDE,
        ),
        ColorParam(
            "colour",
            "Colour:",
            tooltip="The colour the outliers are drawn in.",
            default_value="#d62728",
            visible_for={"action": (ACTION_COLOUR,)},
        ),
        FloatParam(
            "threshold",
            "Threshold:",
            tooltip="Threshold multiplier applied to the method's spread estimate.",
            default_value=3.0,
            minimum=0.1,
            maximum=30.0,
            visible_for={"model": (OUTLIER_ZSCORE, OUTLIER_MAD, OUTLIER_ROLLING)},
        ),
        FloatParam(
            "iqr_factor",
            "IQR multiplier:",
            tooltip="Multiplier applied to IQR outlier bounds.",
            default_value=1.5,
            minimum=0.5,
            maximum=10.0,
            visible_for={"model": (OUTLIER_IQR,)},
        ),
        IntParam(
            "window",
            "Window size:",
            tooltip="Rolling window size, in points, used to compute the local median.",
            default_value=11,
            minimum=3,
            maximum=9999,
            odd_only=True,
            visible_for={"model": (OUTLIER_ROLLING,)},
        ),
        FloatParam(
            "contamination",
            "Contamination:",
            tooltip="Expected fraction of points that are outliers.",
            default_value=0.05,
            minimum=0.001,
            maximum=0.5,
            visible_for={
                "model": (
                    OUTLIER_ISOLATION_FOREST,
                    OUTLIER_LOCAL_OUTLIER_FACTOR,
                    OUTLIER_ELLIPTIC_ENVELOPE,
                )
            },
        ),
        IntParam(
            "n_neighbors",
            "Neighbours:",
            tooltip="Number of neighbours each point's local density is compared against.",
            default_value=20,
            minimum=1,
            maximum=9999,
            visible_for={"model": (OUTLIER_LOCAL_OUTLIER_FACTOR,)},
        ),
        FloatParam(
            "nu",
            "Nu:",
            tooltip=(
                "Upper bound on the fraction of training errors and lower "
                "bound on the fraction of support vectors."
            ),
            default_value=0.05,
            minimum=0.001,
            maximum=0.999,
            visible_for={"model": (OUTLIER_ONE_CLASS_SVM,)},
        ),
    )

    Icon = """
    <circle cx="7" cy="8" r="1.3"/>
    <circle cx="10.5" cy="11" r="1.3"/>
    <circle cx="8.5" cy="15" r="1.3"/>
    <circle cx="14" cy="9" r="1.3"/>
    <circle cx="16" cy="14" r="1.3"/>
    <circle cx="19" cy="5.5" r="1.5"/>
    <path d="M17.8 6.7l-2 2"/>
    """
    def __init__(self, *, repo: SqliteRepo, figure_id: int, parent: QWidget | None = None) -> None:
        if repo is None:
            applogger.error("SeriesOutlierDialog requires a repository instance.")

        self._last_results: list[OutlierResult] = []
        self._parameter_form: QFormLayout | None = None
        self._preview_active = False
        self._preview_hide_snapshots: dict[str, list[int]] = {}
        self._preview_series_sql: dict[int, str] = {}
        self._preview_state_tables: set[str] = set()
        self._preview_colour_snapshots: dict[str, dict[int, str] | None] = {}

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Outliers",
            parent=parent,
            width=720,
            height=640,
        )
        self.setModal(True)
        self.series_selector.set_series_filter(self._has_query)
        self._populate_axes()
        self._refresh_visibility()
        self.mark_results_stale()

    def init_operation_widgets(self) -> None:
        self._parameter_form = None

    def _populate_axes(self) -> None:
        self.series_selector.reload(select_all_series=True)


    def compute_results(self) -> list[OutlierResult]:
        """Detect outliers for each selected source series.

        No selection is a normal dialog startup state, so return an empty list
        instead of raising and showing an error in the preview pane.
        """
        selected = self.selected_series()
        if not selected:
            return []

        usable = self._report_series_without_xy(selected)
        if not usable:
            return []

        model = self.model_combo.currentText()
        params = self._params()
        results: list[OutlierResult] = []
        errors: list[str] = []

        for row in usable:
            try:
                results.append(self._detect_outliers(row, model, params))
            except Exception as exc:
                errors.append(f"{self._series_display_name(row)}: {exc}")

        if errors and not results:
            applogger.error("\n".join(errors))
        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )
        return results

    def _has_query(self, row: Any) -> bool:
        """Only expose source series backed by an SQL query."""
        return bool(row["sql_query"] != "")

    @staticmethod
    def _missing_xy_roles(row: Any) -> list[str]:
        """Return the x/y roles this series does not define.

        Outlier detection is a 1D operation over y ordered by x, and the rows it
        hides are identified by rowid in the source table, so both roles are
        structurally required - there is no sensible default for either.
        """
        roles = parse_roles(row_value(row, "roles"))
        return [
            role
            for role in ("x", "y")
            if not str(roles.get(role, "") or "").strip()
        ]

    def _report_series_without_xy(self, rows: Sequence[Any]) -> list[Any]:
        """Return only the series that can be processed, warning about the rest.

        Two reasons a series is rejected, and the user is told which.

        Without x and y the SQL built for detection is malformed, so the
        operation used to fail with a database error that said nothing about
        the actual cause.

        A series over a saved query has no table to write to: hiding an outlier
        sets a Hide flag on a real row, found by rowid, and a query result has
        neither.  The operation cannot be made to work there - the query would
        have to be materialised first - so it says so instead of failing later
        with "no such column: Hide".
        """
        usable: list[Any] = []
        rejected: list[str] = []
        query_backed: list[str] = []

        for row in rows:
            if not self._repo.is_table_backed_sql(str(row["sql_query"])):
                query_backed.append(f"• {self._series_display_name(row)}")
                continue

            missing = self._missing_xy_roles(row)
            if missing:
                rejected.append(
                    f"• {self._series_display_name(row)} "
                    f"(missing role{'s' if len(missing) > 1 else ''}: {', '.join(missing)})"
                )
            else:
                usable.append(row)

        if query_backed:
            show_message(
                self,
                "series.outliers_need_a_table",
                title=self.operation_label,
                series="\n".join(query_backed),
            )

        if rejected:
            show_message(
                self,
                "series.roles_missing",
                title=self.operation_label,
                series="\n".join(rejected),
            )

        return usable

    @staticmethod
    def _column_diagnostics(frame: pd.DataFrame) -> str:
        """Return compact x/y dtype/sample diagnostics for failure messages."""
        parts: list[str] = []
        for col in ("x", "y"):
            if col not in frame.columns:
                parts.append(f"{col}=<missing>")
                continue
            series = frame[col]
            samples = [str(v) for v in series.dropna().head(3).tolist()]
            parts.append(f"{col} dtype={series.dtype}, sample={samples}")
        return "; ".join(parts)

    def _detect_outliers(
        self,
        choice: Any,
        model: str,
        params: Mapping[str, Any],
    ) -> OutlierResult:
        """Load one source series, compute the outlier mask, and map rowids."""
        roles = parse_roles(choice["roles"])
        source_table = self._repo.query_source_table(choice["sql_query"])
        # The series' own rows, through its own SQL: see the repository.
        source_df = self._repo.query_series_frame_for_hide(sql_query=choice["sql_query"], roles=roles)

        # to_numeric alone would turn a timestamp column into all-NaN, and the
        # only symptom would be this method reporting "not enough points"
        # about a table with a million rows.  See utils.coercion.
        raw_x = to_numeric_axis(source_df["x"])
        raw_y = to_numeric_axis(source_df["y"])
        raw_rowids = source_df["__rowid__"].to_numpy(dtype=int)

        # Report only - never prepare_input_xy here. The mask this method
        # computes is mapped back to source rows through raw_rowids, so sorting
        # or merging x would move each mark onto a different row. Unsorted x is
        # a real problem for the rolling detectors, but the fix belongs in the
        # source data, and the warning says so.
        self.validate_input_xy(
            raw_x, raw_y, label=str(choice["name"]), raise_on_error=False
        )

        finite_mask = np.isfinite(raw_x) & np.isfinite(raw_y)
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count < MIN_POINTS:
            # Say what was actually found: "3 points required" against a full
            # table sends the user looking in the wrong place.
            diagnostics = self._column_diagnostics(source_df)
            applogger.error(
                "%s: %d row(s) read, %d usable X/Y pair(s), %d required. "
                "Check that roles map to the "
                "actual source columns and that X/Y are numeric or date-like. %s",
                self._series_display_name(choice),
                len(source_df),
                finite_count,
                MIN_POINTS,
                diagnostics,
            )
            raise ValueError(
                f"{self._series_display_name(choice)} has {finite_count} usable X/Y pair(s); "
                f"{MIN_POINTS} required. {diagnostics}"
            )

        x_values = raw_x[finite_mask]
        y_values = raw_y[finite_mask]
        rowids = raw_rowids[finite_mask]

        order = np.argsort(x_values)
        x_sorted = x_values[order]
        y_sorted = y_values[order]
        rowids_sorted = rowids[order]
        mask_sorted = self._outlier_mask(x_sorted, y_sorted, model, params)
        outlier_count = int(np.count_nonzero(mask_sorted))
        outlier_rowids = rowids_sorted[mask_sorted]

        x_out = x_sorted[~mask_sorted]
        y_out = y_sorted[~mask_sorted]
        message = f"Detected {outlier_count} outlier(s); apply marks source rows Hide=True"

        return OutlierResult(
            source_name=choice["name"],
            result_name=f"{choice['name']} - Outliers {model}",
            model=model,
            source_table=source_table,
            # The frame is aliased to x/y by query_series_frame_for_hide.
            x_col="x",
            y_col="y",
            x=np.asarray(x_out, dtype=float),
            y=np.asarray(y_out, dtype=float),
            outlier_x=np.asarray(x_sorted[mask_sorted], dtype=float),
            outlier_y=np.asarray(y_sorted[mask_sorted], dtype=float),
            outlier_rowids=np.asarray(outlier_rowids, dtype=int),
            metadata={
                "figure_id": self._figure_id,
                "source_series_id": choice["id"],
                "source_sql_query": choice["sql_query"],
                "model": model,
                "threshold": float(params.get("threshold", 3.0)),
                "iqr_factor": float(params.get("iqr_factor", 1.5)),
                "window": int(params.get("window", 11)),
                "contamination": float(params.get("contamination", 0.05)),
                "n_neighbors": int(params.get("n_neighbors", 20)),
                "nu": float(params.get("nu", 0.05)),
            },
            outlier_count=outlier_count,
            message=message,
            rowids=np.asarray(raw_rowids, dtype=int),
        )

    def _params(self) -> dict[str, Any]:
        return self.parameter_values()

    def _outlier_mask(
        self, x_data: np.ndarray, y_data: np.ndarray, model: str, params: Mapping[str, Any]
    ) -> np.ndarray:
        """The engine's verdict for each point (app.analysis.outliers)."""
        return outlier_mask(
            model,
            x_data,
            y_data,
            OutlierSettings(
                threshold=float(params.get("threshold", 3.0)),
                iqr_factor=float(params.get("iqr_factor", 1.5)),
                window=int(params.get("window", 11)),
                contamination=float(params.get("contamination", 0.05)),
                n_neighbors=int(params.get("n_neighbors", 20)),
                nu=float(params.get("nu", 0.05)),
            ),
        )


    @staticmethod
    def _format_results(results: Sequence[OutlierResult], colour: str | None = None) -> str:
        lines: list[str] = []
        for result in results:
            lines.append(result.source_name)
            if colour is None:
                lines.append(result.message)
            else:
                lines.append(f"Detected {result.outlier_count} outlier(s); apply colours them {colour}")
            lines.append(f"Source table: {result.source_table}")
            lines.append(f"Outliers: {result.outlier_count}")
            if colour is None:
                lines.append(f"Rows marked Hide=True on apply: {result.outlier_count}")
            else:
                lines.append(f"Points coloured on apply: {result.outlier_count}")
            lines.append("")
        return "\n".join(lines).strip()

    def result_series_spec(self, axis_id: int, table_name: str, result: OutlierResult) -> ResultSeriesSpec:
        del axis_id, table_name, result
        raise NotImplementedError("Outlier dialog marks source-table Hide flags and does not create generated series.")

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_outlier_removal": True, "outlier_dialog": "series_outliers"}

    def result_table_name(self, axis_id: int, result: OutlierResult) -> str:
        del axis_id, result
        return generated_table_name("Outlier_Hide_Flags")

    def format_results(self, results: Sequence[OutlierResult]) -> str:
        return self._format_results(results, self._colour_choice())

    def _colour_choice(self) -> str | None:
        """The colour the outliers will be drawn in, or None when they are to be hidden."""
        params = self.parameter_values()
        return str(params.get("colour", "")) if params.get("action") == ACTION_COLOUR else None

    def _ensure_preview_state_attrs(self) -> None:
        """Create preview bookkeeping attributes if an older instance lacks them."""
        if not hasattr(self, "_preview_hide_snapshots"):
            self._preview_hide_snapshots: dict[str, list[int]] = {}
        if not hasattr(self, "_preview_series_sql"):
            self._preview_series_sql: dict[int, str] = {}
        if not hasattr(self, "_preview_state_tables"):
            self._preview_state_tables: set[str] = set()
        if not hasattr(self, "_preview_colour_snapshots"):
            self._preview_colour_snapshots: dict[str, dict[int, str] | None] = {}

    def _hidden_rowids(self, table_name: str) -> list[int]:
        if not self._repo.is_open:
            return []
        return self._repo.hidden_rowids(table_name)

    def _snapshot_outlier_state(self, results: Sequence[OutlierResult]) -> None:
        self._ensure_preview_state_attrs()
        for result in results:
            if result.source_table not in self._preview_hide_snapshots:
                self._preview_hide_snapshots[result.source_table] = self._hidden_rowids(result.source_table)
            source_series_id = result.metadata.get("source_series_id")
            if source_series_id is not None:
                series_id = int(source_series_id)
                if series_id not in self._preview_series_sql:
                    self._preview_series_sql[series_id] = str(result.metadata.get("source_sql_query", ""))

    def _apply_hide_results(self, results: Sequence[OutlierResult]) -> None:
        # One marking per table: the rows of every series processed are shown
        # again, then every outlier any of them found is hidden. Series on
        # the same rows (a measurement and its deseasonalised twin) share
        # them, so marking series by series let the second one show again
        # what the first had just hidden. Rows of series not in this run keep
        # their flags.
        by_table: dict[str, tuple[set[int], set[int]]] = {}
        for result in results:
            scope, hidden = by_table.setdefault(str(result.source_table), (set(), set()))
            scope.update(int(rowid) for rowid in result.rowids)
            hidden.update(int(rowid) for rowid in result.outlier_rowids)
        for table_name, (scope, hidden) in by_table.items():
            self._repo.mark_hide_rowids(
                table_name=table_name, rowids=sorted(hidden), scope_rowids=sorted(scope)
            )
        for result in results:
            source_series_id = result.metadata.get("source_series_id")
            if source_series_id is not None:
                self._repo.update_series_hide_filter(
                    int(source_series_id),
                    str(result.metadata.get("source_sql_query", "")),
                )

    @staticmethod
    def _foreign_colour(result: OutlierResult) -> str:
        """What colours the series already, when it is not this operation's own doing.

        Empty when the series has no colour of its own - or has only the one a
        previous run of this dialog gave it, which a new run may replace.
        """
        sql = str(result.metadata.get("source_sql_query", ""))
        if has_projection_alias(sql, "color") and COLOUR_PROJECTION.lower() not in " ".join(sql.lower().split()):
            return _("a colour column in its query")
        return ""

    def _apply_colour_results(self, results: Sequence[OutlierResult], colour: str) -> None:
        """Colour the outliers' rows and make each series draw that column."""
        for result in results:
            table = result.source_table
            if table not in self._preview_colour_snapshots:
                self._preview_colour_snapshots[table] = self._repo.colour_snapshot(table, COLOUR_COLUMN)
            self._repo.set_row_colours(
                table, COLOUR_COLUMN, [int(rowid) for rowid in result.outlier_rowids], colour
            )
            series_id = result.metadata.get("source_series_id")
            if series_id is not None:
                self._repo.update_series_sql_query(
                    int(series_id),
                    sql_insert_select_expression(
                        str(result.metadata.get("source_sql_query", "")), COLOUR_PROJECTION
                    ),
                )

    def _restore_colour_state(self) -> None:
        """Undo the colouring a preview did: the column's old values, the series' old query."""
        for table, snapshot in self._preview_colour_snapshots.items():
            self._repo.restore_row_colours(table, COLOUR_COLUMN, snapshot)
        self._preview_colour_snapshots.clear()

    def _prepare_preview_state_columns(self, results: Sequence[OutlierResult]) -> None:
        """Snapshot Hide/ClusterId to _Hide/_ClusterId before mutating source tables."""
        self._ensure_preview_state_attrs()
        for result in results:
            table_name = str(result.source_table)
            if table_name not in self._preview_state_tables:
                self._repo.ensure_preview_state_columns(table_name)
                self._preview_state_tables.add(table_name)

    def preview(self) -> bool:
        """Temporarily apply the Hide flags (or the colour) so the chart updates, without closing."""
        self._ensure_preview_state_attrs()
        try:
            params = self.parameter_values()
            colour = str(params.get("colour", "")) if params.get("action") == ACTION_COLOUR else None

            # If Preview is clicked repeatedly, first restore the source table to
            # the pre-preview snapshot, but keep _Hide/_ClusterId as the original baseline.
            if self._preview_active:
                for table_name in list(self._preview_state_tables):
                    self._repo.restore_preview_state_columns(table_name)
                self._restore_colour_state()
                for series_id, sql_query in self._preview_series_sql.items():
                    self._repo.update_series_sql_query(int(series_id), str(sql_query))

            results = list(self.compute_results())
            if not results:
                show_message(self, "series.no_series_selected", title=self.operation_label)
                return False

            if colour is not None:
                for result in results:
                    existing = self._foreign_colour(result)
                    if existing:
                        # Nothing is applied: colouring would overwrite it.
                        show_message(
                            self, "outliers.colour_present",
                            series=result.source_name, existing=existing,
                        )
                        self._preview_active = False
                        self._preview_series_sql.clear()
                        return False

            self._snapshot_outlier_state(results)
            if colour is None:
                self._prepare_preview_state_columns(results)
                self._apply_hide_results(results)
            else:
                self._apply_colour_results(results, colour)
            self.store_cached_results(results)
            self._preview_active = True
            self.applied.emit()
            what = "Hide column updated" if colour is None else f"outliers coloured {colour}"
            message = f"Preview updated: {what} for {len(results)} series."
            detail = self.format_results(results)
            self.set_results_text(f"{detail}\n\n{message}" if detail else message)
            return True
        except Exception as exc:
            applogger.exception("Failed to preview outlier hide flags")
            show_message(self, "series.preview_failed", error=exc)
            return False

    def ok(self) -> None:
        """Accept previewed Hide flags, then remove _Hide/_ClusterId."""
        self._ensure_preview_state_attrs()
        if self._preview_active or self.preview():
            # Like every other operation's Apply: the report goes to the
            # chart's results pane once the change is kept.
            results = list(getattr(self, "_last_results", None) or [])
            if results:
                formatted = self.format_results(results) or ""
                if formatted:
                    self.results_published.emit(self.results_report_html(formatted, results))
            for table_name in list(self._preview_state_tables):
                self._repo.drop_preview_state_columns(table_name)
            self._preview_active = False
            self._preview_hide_snapshots.clear()
            self._preview_series_sql.clear()
            self._preview_state_tables.clear()
            self._preview_colour_snapshots.clear()
            self.accept()

    def cancel_operation_changes(self, *, refresh: bool = True) -> None:
        """Restore Hide/ClusterId from _Hide/_ClusterId and restore source SQL."""
        self._ensure_preview_state_attrs()
        if not self._preview_active and not self._preview_state_tables and not self._preview_colour_snapshots:
            return
        for table_name in list(self._preview_state_tables):
            self._repo.restore_preview_state_columns(table_name)
            self._repo.drop_preview_state_columns(table_name)
        self._restore_colour_state()
        for series_id, sql_query in self._preview_series_sql.items():
            self._repo.update_series_sql_query(int(series_id), str(sql_query))
        self._preview_active = False
        self._preview_hide_snapshots.clear()
        self._preview_series_sql.clear()
        self._preview_state_tables.clear()
        if refresh:
            self.applied.emit()

    def apply(self) -> bool:
        """Compatibility: Preview is the non-closing apply operation for outliers."""
        return self.preview()
