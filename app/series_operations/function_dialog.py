"""Plot a function over a range, without fitting anything to it.

The fit dialog answers "what parameters make this function match my data".
This one answers the question that comes before it and the one that comes
after: what does this function actually look like, and what would it look like
with these parameters?  Same function library, same parameter table, no data
and no optimiser.

Uses are ordinary and constant: seeing the shape of a model before choosing it,
drawing a reference or theoretical curve beside measurements, generating a
synthetic series to test a chart or an operation against, and checking a
starting guess before handing it to the fit dialog - which is worth doing,
because a least-squares fit from a guess in the wrong basin converges
confidently to the wrong answer.

Functions come from ``FunctionScanner``, so ``app/functions/functions.py`` and
``app/functions/user_functions.py`` are both available and a class dropped into
either appears here with no registration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.scanners.functions_scanner import FunctionScanner, SurfaceFunctionScanner
from app.series_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, mark_editor_panel, set_doc_link
from app.utils.i18n import _
from app.utils import report_html

SPACING_LINEAR = "linear"
SPACING_LOG = "log"


@dataclass(slots=True)
class FunctionResult:
    """One evaluated function.

    ``z`` is None for an ordinary y = f(x) curve. For a surface function
    (z = f(x, y)) it holds the evaluated grid, flattened in step with ``x``
    and ``y`` - three parallel 1D arrays over the same points, which is what
    the SQLite table and the Surface renderer both want.
    """

    function_name: str
    result_name: str
    x: np.ndarray
    y: np.ndarray
    params: dict[str, float]
    expression: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    z: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        if self.z is not None:
            return pd.DataFrame({"x": self.x, "y": self.y, "z": self.z})
        return pd.DataFrame({"x": self.x, "y": self.y})


class SeriesFunctionDialog(SeriesOperationDialogBase):
    """Evaluate a scanned function over a range and add it to the chart."""

    Name: str = "Function"
    Description = "Plot a function"

    # This operation reads no source series - it generates one - so the input
    # requirements the base applies to selected series never come into play,
    # and the series picker has nothing to pick. The figure and axis rows stay:
    # a generated curve still has to be drawn somewhere.
    # The whole Axis / Series page goes: there is no series to pick, and where
    # the curve is drawn is asked for under Parameters instead, where the rest
    # of the decisions are. The selector is still built and still answers
    # selected_axis_id(), which is what "same axis" resolves to.
    INPUT_MINIMUM_POINTS = 0
    SHOWS_SERIES_SELECTOR = False
    SHOWS_AXIS_SERIES_PAGE = False

    PARAMS = (
        FloatParam(
            "start",
            "From x:",
            tooltip="Start of the range to evaluate over.",
            default_value=0.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=1.0,
        ),
        FloatParam(
            "stop",
            "To x:",
            tooltip="End of the range to evaluate over.",
            default_value=10.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=1.0,
        ),
        IntParam(
            "points",
            "Points:",
            tooltip=(
                "How densely to sample. A curve with sharp features needs "
                "more; a straight line needs two."
            ),
            default_value=200,
            minimum=2,
            maximum=1_000_000,
            step=50,
        ),
        SeriesOperationDialogBase.destination_param(
            tooltip=(
                "A drawn function is usually meant to be compared with the "
                "data, so the same axis is the default here. A new axis or "
                "figure keeps it separate when the scales do not match."
            ),
            default=SeriesOperationDialogBase.DEST_SAME_AXIS,
        ),
        ChoiceParam(
            "spacing",
            "Spacing:",
            tooltip=(
                "Logarithmic puts equal numbers of points in each decade, "
                "which is what a curve plotted on a log axis needs. It "
                "requires a range strictly above zero."
            ),
            choices=(("Linear", SPACING_LINEAR), ("Logarithmic", SPACING_LOG)),
        ),
    )

    Icon = """
    <path d="M4 20c4 0 4-16 8-16"/>
    <path d="M12 4c4 0 4 16 8 16"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesFunctionDialog requires a repository instance.")

        self._last_results: list[FunctionResult] = []
        self._parameter_form: QFormLayout | None = None
        self._scanner = FunctionScanner()
        # Surface (z = f(x, y)) functions, same split as the fit dialog: see
        # SurfaceFunctionScanner's docstring for why this is a second scanner
        # instance rather than a second base class check sprinkled around.
        self._surface_scanner = SurfaceFunctionScanner()
        self._selected_function: dict[str, Any] = {}
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Plot Function",
            parent=parent,
            width=820,
            height=680,
        )
        self.axis_series_panel.setVisible(False)
        self._reload_function_tree()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._function_tree = QTreeWidget(self)
        self._params_table = QTableWidget(0, 2, self)
        self._expression_label = QLabel("", self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        """The function library, as a tree grouped by category.

        A tree rather than the shared model combo: the library runs to dozens
        of functions across several categories, and a flat combo of that
        length is unreadable. The combo stays hidden and unused.
        """
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.model_combo.setVisible(False)

        self._function_tree.setHeaderLabels([_("Function"), _("Description")])
        self._function_tree.setMinimumHeight(200)
        self._function_tree.currentItemChanged.connect(self._on_function_changed)
        mark_editor_panel(self._function_tree)
        layout.addWidget(self._function_tree, 1)

        self._expression_label.setWordWrap(True)
        self._expression_label.setTextFormat(Qt.TextFormat.RichText)
        self._expression_label.setProperty("muted", True)
        layout.addWidget(self._expression_label, 0)

        container = QWidget(panel)
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(_("Docs:"), self._doc_link)
        layout.addWidget(container, 0)

        return panel

    def build_parameter_selector(self) -> QWidget:
        """The declared range controls, plus the function's own parameters.

        The range is the same four fields whatever the function, so it is
        declared in PARAMS and built by the base. The function's parameters
        are not: their number and names change with the selection, which is
        what a table is for and what a declaration cannot express.
        """
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(super().build_parameter_selector(), 0)

        self._params_table.setHorizontalHeaderLabels([_("Parameter"), _("Value")])
        self._params_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._params_table.verticalHeader().setVisible(False)
        self._params_table.setMinimumHeight(140)
        self._params_table.setToolTip(
            _("Values used to evaluate the function. Nothing is fitted here.")
        )
        self._params_table.itemChanged.connect(lambda *_a: self.mark_results_stale())
        mark_editor_panel(self._params_table)
        layout.addWidget(self._params_table, 1)

        return widget

    def connect_operation_signals(self) -> None:
        # The range controls are wired by ParameterForm; the tree and the
        # parameter table are wired where they are built.
        return None

    def _catalog_data(self) -> dict[str, list[dict[str, Any]]]:
        """Return every plottable function, 1D and surface, grouped by category.

        Merged the same way the fit dialog merges its two scanners: a surface
        function's own categories ("Surfaces", "User surfaces") simply show
        up as ordinary groups in the same tree.
        """
        merged: dict[str, list[dict[str, Any]]] = {
            category: list(models) for category, models in self._scanner.catalog().items()
        }
        for category, models in self._surface_scanner.catalog().items():
            merged.setdefault(category, []).extend(models)
            merged[category].sort(key=lambda item: str(item.get("name", "")).lower())
        return dict(sorted(merged.items(), key=lambda item: item[0].lower()))

    def _is_2d_function(self, payload: Mapping[str, Any] | None = None) -> bool:
        """True when the selected (or given) function is z = f(x, y)."""
        data = payload if payload is not None else self._selected_function
        return int((data or {}).get("ndim", 1)) == 2

    def _reload_function_tree(self) -> None:
        """Fill the tree from the scanner, grouped by category."""
        self._function_tree.blockSignals(True)
        try:
            self._function_tree.clear()
            catalog = self._catalog_data()
            for category, functions in catalog.items():
                parent = QTreeWidgetItem([category, ""])
                # Categories are grouping only; making them selectable invites
                # a click that silently does nothing.
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                for payload in functions:
                    child = QTreeWidgetItem(
                        [str(payload.get("name", "")), str(payload.get("description", ""))]
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, payload)
                    parent.addChild(child)
                self._function_tree.addTopLevelItem(parent)
            self._function_tree.expandAll()
            self._function_tree.resizeColumnToContents(0)
        finally:
            self._function_tree.blockSignals(False)

        first = self._first_function_item()
        if first is not None:
            self._function_tree.setCurrentItem(first)

    def _first_function_item(self) -> QTreeWidgetItem | None:
        """Return the first function under the first non-empty category.

        ``topLevelItem`` returns None for an index the tree does not have, and
        the count can change under a signal while this walks it, so the result
        is checked rather than assumed - which is also what the type says.
        """
        for index in range(self._function_tree.topLevelItemCount()):
            parent = self._function_tree.topLevelItem(index)
            if parent is not None and parent.childCount():
                return parent.child(0)
        return None

    def _on_function_changed(self, current: Any, _previous: Any = None) -> None:
        payload = (
            current.data(0, Qt.ItemDataRole.UserRole)
            if current is not None
            else None
        )
        if not isinstance(payload, dict):
            return

        self._selected_function = payload
        self._rebuild_params_table(payload)

        expression = str(payload.get("expression", "") or "")
        self._expression_label.setText(expression)
        set_doc_link(
            self._doc_link,
            str(payload.get("name", "")),
            str(payload.get("doc_url", "") or ""),
        )
        self.mark_results_stale()

    def _rebuild_params_table(self, payload: Mapping[str, Any]) -> None:
        """Show one editable row per function parameter, seeded from p0."""
        names = [str(name) for name in (payload.get("params") or [])]
        defaults = [float(value) for value in (payload.get("p0") or [])]
        if len(defaults) < len(names):
            defaults += [1.0] * (len(names) - len(defaults))

        self._params_table.blockSignals(True)
        try:
            self._params_table.setRowCount(len(names))
            for row, name in enumerate(names):
                label = QTableWidgetItem(name)
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._params_table.setItem(row, 0, label)
                self._params_table.setItem(
                    row, 1, QTableWidgetItem(f"{defaults[row]:g}")
                )
        finally:
            self._params_table.blockSignals(False)

    def _function_params(self) -> tuple[list[str], np.ndarray]:
        """Return the parameter names and the values currently in the table."""
        names: list[str] = []
        values: list[float] = []
        for row in range(self._params_table.rowCount()):
            label = self._params_table.item(row, 0)
            cell = self._params_table.item(row, 1)
            names.append(label.text() if label else f"p{row}")
            try:
                values.append(float(cell.text()) if cell else 0.0)
            except ValueError:
                # A half-typed number is the normal state of a table being
                # edited, so it falls back rather than raising a dialog at
                # every keystroke.
                values.append(0.0)
        return names, np.asarray(values, dtype=float)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[FunctionResult]:
        payload = self._selected_function
        if not payload:
            return []

        if self._is_2d_function(payload):
            return self._compute_surface_result(payload)

        params = self.parameter_values()
        x_values = self._build_range(params)
        names, values = self._function_params()

        model = self._scanner.make_model(dict(payload))
        y_values = np.asarray(model(x_values, values), dtype=float)

        if y_values.shape != x_values.shape:
            raise ValueError(
                f"the function returned {y_values.size} value(s) for "
                f"{x_values.size} input(s)"
            )

        finite = int(np.count_nonzero(np.isfinite(y_values)))
        if finite == 0:
            raise ValueError(
                "the function is undefined everywhere in this range - check "
                "the range and the parameter values"
            )

        metadata: dict[str, Any] = {
            "spacing": str(params.get("spacing", SPACING_LINEAR)),
            "range": f"{x_values[0]:g} .. {x_values[-1]:g}",
        }
        if finite < y_values.size:
            # Not an error: many functions are legitimately undefined over
            # part of a range - a log below zero, a pole in a rational - and
            # the useful behaviour is to draw the part that exists and say how
            # much was dropped.
            metadata["undefined"] = y_values.size - finite

        name = str(payload.get("name", "function"))
        return [
            FunctionResult(
                function_name=name,
                result_name=name,
                x=x_values,
                y=y_values,
                params=dict(zip(names, (float(value) for value in values))),
                expression=str(payload.get("expression", "") or ""),
                metadata=metadata,
            )
        ]

    def _compute_surface_result(self, payload: Mapping[str, Any]) -> list[FunctionResult]:
        """Evaluate a z = f(x, y) function over a grid and flatten it.

        Judgment call: the same From/To/Points range declared for x is reused
        for y rather than adding a second set of range controls - a
        reasonable default for a first version (a square domain), and one
        that keeps this dialog's parameter panel unchanged for the 1D case.
        A future version could add independent Y range controls without
        touching anything else here.
        """
        params = self.parameter_values()
        x_lin = self._build_range(params)
        y_lin = x_lin.copy()
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)

        names, values = self._function_params()
        model = self._surface_scanner.make_model(dict(payload))
        xy = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        z_values = np.asarray(model(xy, values), dtype=float)

        finite = int(np.count_nonzero(np.isfinite(z_values)))
        if finite == 0:
            raise ValueError(
                "the function is undefined everywhere in this range - check "
                "the range and the parameter values"
            )

        metadata: dict[str, Any] = {
            "spacing": str(params.get("spacing", SPACING_LINEAR)),
            "range": f"{x_lin[0]:g} .. {x_lin[-1]:g} (both axes)",
        }
        if finite < z_values.size:
            metadata["undefined"] = z_values.size - finite

        name = str(payload.get("name", "function"))
        return [
            FunctionResult(
                function_name=name,
                result_name=name,
                x=x_grid.ravel(),
                y=y_grid.ravel(),
                z=z_values,
                params=dict(zip(names, (float(value) for value in values))),
                expression=str(payload.get("expression", "") or ""),
                metadata=metadata,
            )
        ]

    @staticmethod
    def _build_range(params: Mapping[str, Any]) -> np.ndarray:
        """Return the x values to evaluate at."""
        start = float(params.get("start", 0.0))
        stop = float(params.get("stop", 10.0))
        count = max(2, int(params.get("points", 200)))

        if start == stop:
            raise ValueError("the range is empty - From and To are the same")

        if str(params.get("spacing", SPACING_LINEAR)) == SPACING_LOG:
            if start <= 0.0 or stop <= 0.0:
                raise ValueError(
                    "logarithmic spacing needs a range strictly above zero"
                )
            return np.logspace(np.log10(start), np.log10(stop), count)

        return np.linspace(start, stop, count)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def resolve_target_axis_id(
        self,
        selected_axis_id: int,
        results: Sequence[Any],
    ) -> int:
        """Draw where the destination setting says.

        Defaults to the selected axis, unlike Calculus: a plotted function is
        usually a reference curve meant to sit on top of the measurements, and
        putting it somewhere else by default would defeat the point of drawing
        it.
        """
        name = str(results[0].function_name) if results else "Function"
        is_surface = bool(results and results[0].z is not None)
        if is_surface:
            # An exact rectangular meshgrid, not scattered points - built
            # from np.meshgrid over the declared range - so the plain
            # (non-scattered) Surface renderer is the honest choice here,
            # unlike the fit dialog's TriSurface for arbitrary data points.
            # "projection": "3d" is what create_chart_dialog itself sets for
            # every CHART_TYPES_NEEDING_3D_AXES entry when a Surface axis is
            # created through the UI - an axis made programmatically (as
            # "New axis"/"New figure" here do) has to set it explicitly too.
            options = {"grid": True, "projection": "3d"}
            chart_type = "Surface Plot"
        else:
            options = {"grid": True, "linestyle": "-", "marker": ""}
            chart_type = "Scatter Plot"
        return self.resolve_destination_axis(
            selected_axis_id,
            chart_type=chart_type,
            title=name,
            figure_name=name,
            options=options,
        )

    def discard_operation_artifacts(self) -> None:
        """Remove the axis or figure this dialog made, when Apply never ran."""
        self.discard_result_target()

    def selected_series(self) -> list[Any]:
        """No source series: this operation generates one.

        The base uses the selection to decide which axis to write to, and an
        empty selection is the normal state here rather than an error.
        """
        return []

    def apply(self) -> bool:
        """Apply, and keep any axis or figure this dialog created."""
        applied = super().apply()
        self._applied = self._applied or applied
        return applied

    def result_to_frame(self, result: FunctionResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: FunctionResult,
    ) -> ResultSeriesSpec:
        del axis_id
        if result.z is not None:
            return ResultSeriesSpec(
                name=result.result_name,
                sql_query=f'SELECT x, y, z FROM "{table_name}"',
                roles={"x": "x", "y": "y", "z": "z"},
                style={
                    "generated_function": True,
                    "function_dialog": "series_function",
                    "function": result.function_name,
                    "fit_mode": "2D",
                },
            )
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_function": True,
                "function_dialog": "series_function",
                "function": result.function_name,
                # A drawn function is a smooth curve, not a set of
                # measurements, so it gets a line and no markers.
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_function": True, "function_dialog": "series_function"}

    def result_table_name(self, axis_id: int, result: FunctionResult) -> str:
        return generated_table_name(
            f"Function_axis{axis_id}_{result.function_name}",
            fallback="Function_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Function"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[FunctionResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            is_surface = result.z is not None
            values = result.z if is_surface else result.y
            finite = np.isfinite(values)
            summary_rows: list[tuple[str, Any]] = [
                (_("Function"), result.function_name),
                (_("Points"), values.size),
                (_("Range"), result.metadata.get("range", "")),
                (_("Spacing"), result.metadata.get("spacing", "")),
            ]
            if finite.any():
                summary_rows.extend(
                    [
                        (
                            _("z range") if is_surface else _("y range"),
                            f"{report_html.format_number(float(np.min(values[finite])))}"
                            f" .. "
                            f"{report_html.format_number(float(np.max(values[finite])))}",
                        ),
                    ]
                )
            if "undefined" in result.metadata:
                summary_rows.append((_("Undefined points"), result.metadata["undefined"]))

            blocks = [report_html.summary_table(summary_rows)]

            if result.expression:
                # raw_note, not note: expression is authored as HTML by the
                # function class - "<b>Constant</b><br>y = C" - so escaping it
                # shows the reader the tags instead of the formula.
                blocks.append(report_html.raw_note(result.expression))

            if result.params:
                blocks.append(
                    report_html.table(
                        (_("Parameter"), _("Value")),
                        [
                            (name, report_html.format_number(value))
                            for name, value in result.params.items()
                        ],
                        align=("left", "right"),
                        empty_message=_("This function takes no parameters."),
                    )
                )

            sections.append(report_html.section(result.function_name, *blocks))

        return report_html.document(_("Function"), "", *sections)
