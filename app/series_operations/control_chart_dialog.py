"""Shewhart control charts for a chart series - variables and attributes.

A control chart asks one question: is this process varying the way a stable
process varies, or has something changed?  It answers it by drawing limits
around the process's *own* variation and flagging the points that fall
outside.

Eight charts, in two families, because that is one decision a user should
not have to make before they can find the tool:

**Variables** - the measurement charts.  What makes these control charts
rather than a scatter plot with error bars is where sigma comes from.  It is
**never** the standard deviation of all the data: a process that has drifted
has a large overall standard deviation precisely *because* it drifted, so
limits built from it are wide enough to contain the drift and the chart
declares the process fine.  Sigma is estimated instead from variation
*within* subgroups - the average moving range for individual measurements,
the average range or standard deviation within subgroups otherwise - which
is unaffected by shifts between them.  That is the whole idea, and it is the
one thing easy to get wrong.

**Attributes** - the count charts, the half the variables charts do not
cover.  These plot a proportion or a count rather than a measurement, so the
limits come from the distribution the count follows rather than from a
within-subgroup spread:

* **p** - fraction defective, ``d / n``.  Binomial.  Centre ``p̄ = Σd / Σn``,
  limits ``p̄ ± L·√(p̄(1-p̄)/nᵢ)``: when the sample size varies, so does the
  limit, and the chart draws a different band at every point.
* **np** - number defective, ``d``, at a fixed ``n``.  Centre ``n·p̄``,
  limits ``n·p̄ ± L·√(n·p̄(1-p̄))``.
* **c** - defects in a unit of constant size.  Poisson.  ``c̄ ± L·√c̄``.
* **u** - defects per unit, ``c / n``.  Poisson.  ``ū ± L·√(ū/nᵢ)`` - again
  per point when ``n`` varies.

The lower limit is clipped at zero on the count charts: a count cannot be
negative, and an unclipped LCL below zero never signals.

The variables estimators need the unbiasing constants d2, d3, c4, A2, D3,
D4, B3 and B4.  They are tabulated below rather than computed: c4 has a
closed form in gamma functions, but d2 and d3 are integrals over the range
distribution with no elementary form, and every SPC text ships the same
table.  Using anything else would put this chart's limits at odds with every
other tool's.

Violations are reported by the Nelson rules, which catch the patterns that
stay inside the limits - a run on one side, a trend, a hug of the centre
line - and are what a chart is for beyond spotting the obvious outlier.
Which rules are *legal* depends on the chart, and that is the one thing the
two families have to disagree about: see
:func:`app.analysis.control_charts.find_violations`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox, QFormLayout, QWidget

from app.analysis.control_charts import (  # noqa: F401 - re-exported for the tests
    CHART_C,
    CHART_INDIVIDUALS,
    CHART_MOVING_RANGE,
    CHART_NP,
    CHART_P,
    CHART_U,
    CHART_XBAR_R,
    CHART_XBAR_S,
    SPC_CONSTANTS,
    ControlChart,
    Violation,
    attribute_chart,
    attribute_limits,
    variables_chart,
)
from app.data.data_source import parse_roles, row_value, resolve_role_column
from app.data.repo.tables import QueryColumns, coerce_numeric_array
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import BoolParam, FloatParam, IntParam
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.utils.config import get_constant
from app.utils.i18n import _
from app.utils import report_html

@dataclass(frozen=True, slots=True, kw_only=True)
class ControlChartModel(OperationModel):
    #: An attribute chart - a count per point - rather than a variables
    #: chart, a measurement per point. The combo lists measurements above a
    #: line and counts below it: picking a chart is one decision, and a user
    #: who knows they have counts should not have to say so twice.
    attribute: bool = False
    #: Averages several readings into each plotted point.
    subgrouped: bool = False
    #: Reads a sample-size column. ``c`` is the exception among the
    #: attribute charts - it assumes a constant area of opportunity, so there
    #: is nothing to divide by.
    needs_size_column: bool = False
    #: On the binomial: a proportion, bounded above by its sample size.
    binomial: bool = False


#: The charts offered, in combo order.
CONTROL_CHARTS: dict[str, ControlChartModel] = {
    CHART_INDIVIDUALS: ControlChartModel(
        doc_title="Individuals control chart",
        doc_url="https://en.wikipedia.org/wiki/Shewhart_individuals_control_chart",
    ),
    CHART_MOVING_RANGE: ControlChartModel(
        doc_title="Moving range",
        doc_url="https://en.wikipedia.org/wiki/Shewhart_individuals_control_chart",
    ),
    CHART_XBAR_R: ControlChartModel(
        doc_title="X-bar and R chart",
        doc_url="https://en.wikipedia.org/wiki/X%CC%84_and_R_chart",
        subgrouped=True,
    ),
    CHART_XBAR_S: ControlChartModel(
        doc_title="X-bar and s chart",
        doc_url="https://en.wikipedia.org/wiki/X%CC%84_and_s_chart",
        subgrouped=True,
    ),
    CHART_P: ControlChartModel(
        doc_title="p-chart",
        doc_url="https://en.wikipedia.org/wiki/P-chart",
        group="attributes",
        attribute=True,
        needs_size_column=True,
        binomial=True,
    ),
    CHART_NP: ControlChartModel(
        doc_title="np-chart",
        doc_url="https://en.wikipedia.org/wiki/Np-chart",
        group="attributes",
        attribute=True,
        needs_size_column=True,
        binomial=True,
    ),
    CHART_C: ControlChartModel(
        doc_title="c-chart",
        doc_url="https://en.wikipedia.org/wiki/C-chart",
        group="attributes",
        attribute=True,
    ),
    CHART_U: ControlChartModel(
        doc_title="u-chart",
        doc_url="https://en.wikipedia.org/wiki/U-chart",
        group="attributes",
        attribute=True,
        needs_size_column=True,
    ),
}
SUBGROUPED = tuple(name for name, chart in CONTROL_CHARTS.items() if chart.subgrouped)


#: Below this an attribute chart's limits are too soft to trust; the report
#: says so rather than refusing to draw them.
RECOMMENDED_SUBGROUPS = get_constant("control_chart_recommended_subgroups", 20)


@dataclass(slots=True)
class ControlChartResult(TableResult):
    """One control chart - of either family - for one source series.

    ``upper``/``lower``/``sigma`` are scalars: the constant value on every
    variables chart and on np/c, and the *mean* band on a p or u chart whose
    limits genuinely move with the sample size - kept scalar rather than
    dropped so the many existing checks against a variables chart's own
    limits (``result.upper == ...``) still read the single number they
    always meant. ``*_band`` are the same thing per point, always populated
    (a repeated value where the chart has no reason to vary): ``to_frame``
    and the violation pass need the real band, not its average, and only a
    p/u chart's is not just that scalar broadcast.
    """

    source_name: str
    result_name: str
    chart: str
    x: np.ndarray
    y: np.ndarray
    center: float
    upper: float
    lower: float
    sigma: float
    upper_band: np.ndarray
    lower_band: np.ndarray
    sigma_band: np.ndarray
    subgroup_size: int
    violations: list[Violation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return self.chart

    @property
    def limits_vary(self) -> bool:
        """True when the band moves point to point (a p or u chart on an
        uneven sample size), which is what makes the zone rules illegal."""
        return bool(self.sigma_band.size and np.ptp(self.sigma_band) > 0.0)

    def to_df(self) -> pd.DataFrame:
        """Every line the chart may draw, as columns of one table.

        Written whether or not the corresponding checkbox is ticked: the
        columns cost almost nothing, and holding them means turning a line on
        later is a descriptor change rather than a recomputation.

        The zone columns are the one- and two-sigma lines, which is what
        divides a Shewhart chart into the A/B/C zones the run rules are
        phrased in - "two of three beyond two sigma" is a statement about a
        line the reader should be able to see.
        """
        flagged = {violation.index for violation in self.violations}
        size = self.x.size
        return pd.DataFrame(
            {
                "x": self.x,
                "y": self.y,
                "center": np.full(size, self.center),
                "ucl": self.upper_band,
                "lcl": self.lower_band,
                "zone_2_upper": self.center + 2.0 * self.sigma_band,
                "zone_2_lower": self.center - 2.0 * self.sigma_band,
                "zone_1_upper": self.center + self.sigma_band,
                "zone_1_lower": self.center - self.sigma_band,
                "violation": [int(i in flagged) for i in range(size)],
                # The flagged points as their own column, NULL elsewhere, so a
                # series can plot them alone. A WHERE clause would work for
                # apply, but the preview writes the same table and a second
                # query over it is a second scan for no gain.
                "violation_y": [
                    float(value) if index in flagged else None
                    for index, value in enumerate(self.y)
                ],
            }
        )


class SeriesControlChartDialog(SeriesOperationDialogBase):
    """Draw a Shewhart control chart - variables or attributes - for a series."""

    MODELS = CONTROL_CHARTS
    MODEL_LABEL = "Chart:"
    MODEL_TOOLTIP = "Measurements above the line, counts below it."

    Name: str = "Control Chart"
    Description = "Monitor process stability (I-MR, X-bar, p, np, c, u)"

    #: Computed on a worker thread: see SeriesOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    # The points are a time order, so they must be in x order: every estimator
    # here reads consecutive differences, and a shuffled series produces a
    # moving range that describes the sort order rather than the process.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # A moving range needs a predecessor, and two points give one range - too
    # few to estimate anything, but enough not to crash. Nelson's run rules
    # want far more; the report says so when the series is short.
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        IntParam(
            "subgroup",
            "Subgroup size:",
            tooltip=(
                "Consecutive points averaged into each plotted subgroup. "
                "Rational subgrouping: choose it so variation within a "
                "subgroup is only common-cause noise."
            ),
            default_value=5,
            minimum=2,
            maximum=25,
            visible_for={"model": SUBGROUPED},
        ),
        FloatParam(
            "sigma_limit",
            "Limits at sigma:",
            tooltip=(
                "3 is the Shewhart convention: on a stable normal process it "
                "gives about one false alarm per 370 points, which balances "
                "missed signals against chasing noise."
            ),
            default_value=3.0,
            minimum=1.0,
            maximum=6.0,
            decimals=2,
            step=0.5,
        ),
        BoolParam(
            "nelson",
            "Apply Nelson rules:",
            tooltip=(
                "Flags runs, trends and other patterns that stay inside the "
                "limits - the signals a limits-only chart misses. The zone "
                "rules switch themselves off on a chart whose limits move."
            ),
            default_value=True,
        ),
        BoolParam(
            "draw_center",
            "Draw the centre line:",
            tooltip="The process mean the limits are built around.",
            default_value=True,
        ),
        BoolParam(
            "draw_limits",
            "Draw the control limits:",
            tooltip=(
                "The upper and lower limits. Without them the chart is a run "
                "chart - the points, but nothing to judge them against."
            ),
            default_value=True,
        ),
        BoolParam(
            "draw_zones",
            "Draw the one and two sigma lines:",
            tooltip=(
                "The A/B/C zones the run rules are phrased in: \"two of three "
                "beyond two sigma\" is a statement about a line worth seeing."
            ),
            default_value=False,
        ),
        BoolParam(
            "draw_violations",
            "Highlight the flagged points:",
            tooltip="Draws the signalling points again as separate markers.",
            default_value=True,
        ),
        BoolParam(
            "exclude_violations",
            "Exclude flagged points from the limits:",
            tooltip=(
                "Recomputes the limits without the points they flagged - the "
                "trial-limits-then-revised-limits pass every SPC text runs. "
                "Use only when the flagged points have an assigned cause you "
                "have removed; otherwise it hides the problem."
            ),
            default_value=False,
        ),
    )

    Icon = """
    <path d="M3 8h18"/>
    <path d="M3 16h18"/>
    <path d="M3 12h18" stroke-dasharray="2 2"/>
    <path d="M5 13l3-2 3 3 3-5 3 4 3-2"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesControlChartDialog requires a repository instance.")

        self._last_results: list[ControlChartResult] = []
        self._parameter_form: QFormLayout | None = None

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Control Chart",
            parent=parent,
            width=800,
            height=680,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_size_columns()
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        self._size_column_combo = QComboBox(self)
        self._parameter_form = None

    def build_parameter_selector(self) -> QWidget:
        """The declared parameters, plus the one that cannot be declared.

        Everything in ``PARAMS`` is built by the base class and shown or
        hidden per chart by ``visible_for``.  The sample-size column is the
        exception: its choices are the selected series' own columns, which
        are not known until a series is picked, so it is an ordinary combo
        appended to the same form and gated by hand in ``_refresh_visibility``.
        """
        widget = super().build_parameter_selector()

        self._size_column_combo.setToolTip(
            _("The column holding the sample size (units inspected) at each point.")
        )
        if self._parameter_form is not None:
            self._parameter_form.addRow(
                _("Sample size column:"), self._size_column_combo
            )
        return widget

    def connect_common_signals(self) -> None:
        # Not super(): the base connects selection_changed straight to
        # mark_results_stale, but the size-column combo has to be
        # repopulated from the new selection first.
        changed = getattr(self.series_selector, "selection_changed", None)
        if changed is not None:
            changed.connect(self._on_selection_changed)

    def _on_selection_changed(self, *_args: Any) -> None:
        self._refresh_size_columns()
        self.mark_results_stale()

    def connect_operation_signals(self) -> None:
        super().connect_operation_signals()
        self._size_column_combo.currentIndexChanged.connect(self.mark_results_stale)

    def _refresh_visibility(self, *_ignored: Any) -> None:
        super()._refresh_visibility()
        self.set_row_visible(
            self._size_column_combo, CONTROL_CHARTS[self._chart()].needs_size_column
        )

    def _refresh_size_columns(self) -> None:
        """Fill the sample-size combo with the selected series' columns."""
        wanted = self._size_column_combo.currentText()
        columns: list[str] = []
        for row in self.selected_series():
            result = self._safe_columns(row)
            if result is None:
                continue
            for column in result.columns:
                if column not in columns:
                    columns.append(column)

        self._size_column_combo.blockSignals(True)
        self._size_column_combo.clear()
        self._size_column_combo.addItems(columns)
        if wanted in columns:
            self._size_column_combo.setCurrentText(wanted)
        elif "n" in columns:
            self._size_column_combo.setCurrentText("n")
        self._size_column_combo.blockSignals(False)

    def _chart(self) -> str:
        return self.model_combo.currentText() or CHART_INDIVIDUALS

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def _safe_columns(self, row: Any) -> QueryColumns | None:
        query = str(row_value(row, "sql_query", "query", default="")).strip()
        if not query:
            return None
        try:
            return self._repo.query_arrays(query)
        except Exception:  # noqa: BLE001 - a bad query is reported at compute time
            return None

    @staticmethod
    def _looks_numeric(values: np.ndarray) -> bool:
        """True when coercing *values* produces at least one real number.

        The fallback-column heuristic below needs to tell "this column is
        usable as counts" from "this column is text" without pandas' own
        dtype introspection (there is no DataFrame here to introspect) - a
        column that coerces to all-NaN is exactly the text case.
        """
        coerced = coerce_numeric_array(values)
        return bool(coerced.size) and not bool(np.all(np.isnan(coerced)))

    def _series_counts(
        self, row: Any, name: str, chart: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(x, counts, sizes)`` for one series, aligned and finite.

        Only the attribute charts come through here; the variables charts
        read x/y through the shared ``series_xy``, which they can because
        they need no third column. Read through query_arrays rather than
        query_df: no DataFrame is built for a dialog that only ever wanted
        two or three plain numeric columns out of it. The one column that
        might be a timestamp - x - is still handed to numeric_x exactly as
        it always was, wrapped in a single-column Series rather than a
        whole frame, because that is real date-parsing logic this is not
        trying to reimplement.
        """
        columns_result = self._safe_columns(row)
        if columns_result is None or columns_result.empty:
            raise ValueError("the series query returned no rows")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = list(columns_result.columns)

        y_col = resolve_role_column(columns, roles, "y") or "y"
        if y_col not in columns:
            numeric = [c for c in columns if self._looks_numeric(columns_result[c])]
            y_col = numeric[-1] if numeric else columns[-1]
        counts = coerce_numeric_array(columns_result[y_col])

        x_col = resolve_role_column(columns, roles, "x") or "x"
        x_values = (
            self.numeric_x(pd.Series(columns_result[x_col]), name)
            if x_col in columns
            else np.arange(counts.size, dtype=float)
        )

        if CONTROL_CHARTS[chart].needs_size_column:
            size_col = self._size_column_combo.currentText().strip()
            size_col = size_col or str(roles.get("n") or roles.get("sample_size") or "")
            if size_col not in columns:
                raise ValueError(
                    "pick the column holding the sample size in the parameters pane"
                )
            sizes = coerce_numeric_array(columns_result[size_col])
        else:
            sizes = np.ones(counts.size, dtype=float)

        finite = np.isfinite(x_values) & np.isfinite(counts) & np.isfinite(sizes)
        finite &= sizes > 0
        x_values, counts, sizes = x_values[finite], counts[finite], sizes[finite]
        if x_values.size < self.INPUT_MINIMUM_POINTS:
            raise ValueError(
                f"only {x_values.size} usable point(s); at least "
                f"{self.INPUT_MINIMUM_POINTS} are needed"
            )
        if np.any(counts < 0):
            raise ValueError("the count column has negative values")
        if CONTROL_CHARTS[chart].binomial and np.any(counts > sizes):
            raise ValueError("a subgroup has more defectives than its sample size")

        order = np.argsort(x_values, kind="stable")
        return x_values[order], counts[order], sizes[order]

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def series_settings(self) -> tuple[str, dict[str, Any]]:
        return self._chart(), self.parameter_values()

    def read_series(self, row: Any, name: str) -> Any:
        chart = self._chart()
        if CONTROL_CHARTS[chart].attribute:
            return self._series_counts(row, name, chart)
        return self.series_xy(row, name)

    def compute_series(self, name: str, data: Any, settings: tuple[str, dict[str, Any]]) -> ControlChart:
        chart, params = settings
        common = {
            "sigma_limit": float(params.get("sigma_limit", 3.0)),
            "nelson": bool(params.get("nelson", True)),
            "exclude_violations": bool(params.get("exclude_violations", False)),
        }
        if CONTROL_CHARTS[chart].attribute:
            x_values, counts, sizes = data
            return attribute_chart(
                chart, x_values, counts, sizes, recommended_subgroups=RECOMMENDED_SUBGROUPS, **common
            )
        x_values, y_values = data
        return variables_chart(
            chart, x_values, y_values, subgroup=int(params.get("subgroup", 5)), **common
        )

    def finish_series(
        self, name: str, outcome: ControlChart, settings: tuple[str, dict[str, Any]]
    ) -> ControlChartResult:
        chart, _params = settings
        for note in outcome.notes:
            applogger.warning(f"{name}: {note}", show_dialog=False, raise_error=False)
        tag = chart.split(" ")[0] if CONTROL_CHARTS[chart].attribute else chart
        return ControlChartResult(
            source_name=name,
            result_name=f"{name} - {tag}",
            chart=chart,
            x=outcome.x,
            y=outcome.y,
            center=outcome.center,
            upper=outcome.upper,
            lower=outcome.lower,
            sigma=outcome.sigma,
            upper_band=outcome.upper_band,
            lower_band=outcome.lower_band,
            sigma_band=outcome.sigma_band,
            subgroup_size=outcome.subgroup_size,
            violations=outcome.violations,
            metadata=outcome.details,
        )

    # -- One chart, without the window (the tests' entry points) -------

    def _build_attribute_chart(
        self,
        name: str,
        x_values: np.ndarray,
        counts: np.ndarray,
        sizes: np.ndarray,
        chart: str,
        params: Mapping[str, Any],
    ) -> ControlChartResult:
        settings = (chart, dict(params))
        return self.finish_series(name, self.compute_series(name, (x_values, counts, sizes), settings), settings)

    def _build_chart(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        chart: str,
        params: Mapping[str, Any],
    ) -> ControlChartResult:
        settings = (chart, dict(params))
        return self.finish_series(name, self.compute_series(name, (x_values, y_values), settings), settings)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    #: Shared by every series this operation draws, so that
    #: remove_previous_generated_series takes the whole chart away rather than
    #: leaving orphaned limit lines behind when it is re-applied.
    def _series_style(self, result: ControlChartResult, **overrides: Any) -> dict[str, Any]:
        style: dict[str, Any] = {
            "generated_control_chart": True,
            "control_chart_dialog": "series_control_chart",
            "source_name": result.source_name,
            "chart": result.chart,
        }
        style.update(overrides)
        return style

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: ControlChartResult,
    ) -> ResultSeriesSpec:
        """The plotted values themselves - the series the chart is about."""
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style=self._series_style(
                result,
                linestyle="-",
                linewidth=1.2,
                marker="o",
                markersize=4.0,
            ),
        )

    def result_series_specs(
        self,
        axis_id: int,
        table_name: str,
        result: ControlChartResult,
    ) -> Sequence[ResultSeriesSpec]:
        """The points, plus whichever reference lines are switched on.

        A control chart is the points *and* the lines: without limits it is a
        run chart, which shows the same numbers and answers a different
        question. Hence the lines default to on and this is not a one-series
        operation.

        Order matters to the drawing: the reference lines go first so the data
        is drawn over them, and the flagged points go last so they sit on top
        of everything. On a p or u chart the limit columns step from point to
        point rather than being flat, which is the honest picture of a band
        that really does move with the sample size.
        """
        params = self.parameter_values()
        specs: list[ResultSeriesSpec] = []

        def line(column: str, name: str, **style: Any) -> ResultSeriesSpec:
            return ResultSeriesSpec(
                name=f"{result.result_name} - {name}",
                # Aliased to y so the renderer's x/y roles need no special
                # case: every one of these is an ordinary two-column series.
                sql_query=f'SELECT x, {column} AS y FROM "{table_name}" ORDER BY x',
                roles={"x": "x", "y": "y"},
                style=self._series_style(result, marker="", **style),
            )

        if bool(params.get("draw_zones", False)):
            for column, label in (
                ("zone_2_upper", "+2s"),
                ("zone_2_lower", "-2s"),
                ("zone_1_upper", "+1s"),
                ("zone_1_lower", "-1s"),
            ):
                specs.append(
                    line(column, label, linestyle=":", linewidth=0.7, color="#9e9e9e")
                )

        if bool(params.get("draw_limits", True)):
            specs.append(
                line("ucl", "UCL", linestyle="--", linewidth=1.1, color="#c62828")
            )
            specs.append(
                line("lcl", "LCL", linestyle="--", linewidth=1.1, color="#c62828")
            )

        if bool(params.get("draw_center", True)):
            specs.append(
                line("center", "CL", linestyle="-", linewidth=1.0, color="#2e7d32")
            )

        specs.append(self.result_series_spec(axis_id, table_name, result))

        if bool(params.get("draw_violations", True)) and result.violations:
            specs.append(
                ResultSeriesSpec(
                    name=f"{result.result_name} - signals",
                    # violation_y is NULL for every point that did not signal,
                    # so this draws only the flagged ones without a second
                    # table or a WHERE clause the preview would have to repeat.
                    sql_query=(
                        f'SELECT x, violation_y AS y FROM "{table_name}" '
                        f"WHERE violation_y IS NOT NULL ORDER BY x"
                    ),
                    roles={"x": "x", "y": "y"},
                    style=self._series_style(
                        result,
                        linestyle="",
                        marker="o",
                        markersize=9.0,
                        color="#c62828",
                    ),
                )
            )

        return specs

    RESULT_TABLE_PREFIX = 'ControlChart'
    RESULT_TABLE_VARIANT = 'chart'

    @property
    def operation_label(self) -> str:
        return "Control Chart"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[ControlChartResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            varies = result.limits_vary
            summary_rows: list[tuple[str, Any]] = [
                (_("Chart"), result.chart),
                (_("Points plotted"), result.y.size),
            ]
            if not CONTROL_CHARTS[result.chart].attribute:
                summary_rows.append((_("Subgroup size"), result.subgroup_size))
            summary_rows.append(
                (_("Centre line"), report_html.format_number(result.center))
            )
            summary_rows += [
                (
                    _("Mean upper limit") if varies else _("Upper control limit"),
                    report_html.format_number(result.upper),
                ),
                (
                    _("Mean lower limit") if varies else _("Lower control limit"),
                    report_html.format_number(result.lower),
                ),
                (_("Sigma estimate"), report_html.format_number(result.sigma)),
                (_("Estimator"), result.metadata.get("estimator", "")),
            ]
            summary_rows += [
                (str(key), report_html.format_number(value)
                 if isinstance(value, float) else value)
                for key, value in result.metadata.items()
                if key != "estimator"
            ]
            summary = report_html.summary_table(summary_rows)

            if varies:
                zones = report_html.note(
                    _(
                        "The limits move with the sample size, so the zone "
                        "rules (Nelson 5-8) do not apply and were not run."
                    )
                )
            else:
                zones = ""

            if result.violations:
                table = report_html.table(
                    (_("Point"), "x", "y", _("Rule"), _("Signal")),
                    [
                        (
                            str(violation.index + 1),
                            report_html.format_number(violation.x),
                            report_html.format_number(violation.y),
                            ", ".join(str(rule) for rule in violation.rules),
                            "; ".join(_(text) for text in violation.descriptions),
                        )
                        for violation in result.violations
                    ],
                    align=("right", "right", "right", "right", "left"),
                )
                verdict = report_html.note(
                    _(
                        "{count} of {total} points signal. The process is not "
                        "in statistical control."
                    ).format(count=len(result.violations), total=result.y.size)
                )
            else:
                table = ""
                verdict = report_html.note(
                    _(
                        "No points signal. The process is in statistical "
                        "control - which says it is stable, not that it meets "
                        "any specification."
                    )
                )

            sections.append(
                report_html.section(result.source_name, summary, zones, verdict, table)
            )

        return report_html.document(_("Control Chart"), self._chart(), *sections)
