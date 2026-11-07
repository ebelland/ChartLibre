"""Dialog to fit SQLite tables without a plot preview.

The dialog is intentionally table-oriented:
- 1D fitting: target = f(x)
- Scanned function catalog, parameter table, bounds, fixed parameters
- Explicit error handling when selected source columns are missing/stale
- Save fitted values/residuals back to a normal SQLite table

The fitting itself - optimisation, errors, t/p values, intervals - is in
app.analysis.fit; this dialog builds the model from the catalogue, reads the
series and the parameter table, and lays the result out.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from html import escape as html_escape
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.analysis import fit as fit_engine
from app.data.data_source import parse_roles, resolve_role_column, row_value
from app.data_operations.fit_dialog.common import (
    SeriesFitResult,
    _evaluated_expression_html,
    _FitJob,
    _SurfaceInitialGuessAdapter,
)
from app.data_operations.sympy_store import catalog as stored_sympy_catalog
from app.data_operations.sympy_store import functions_by_name
from app.data_operations.sympy_store import make_model as make_stored_sympy_model
from app.logs.logger import applogger
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import (
    DataOperationDialogBase,
    OperationJob,
    ResultSeriesSpec,
    generated_table_name,
)
from app.plugin_api.data_operations.parameter_spec import (
    BoolParam,
    FloatParam,
    IntParam,
)
from app.plugin_api.functions.optimizers import (
    BY_KEY,
    DEFAULT_OPTIMIZER,
    LOSSES,
    OPTIMIZERS,
)
from app.plugin_api.functions.starting_point import (
    FROM_DECLARED,
    FROM_FUNCTION,
    FROM_SEARCH,
    ask_the_function,
    choose_starting_point,
    clip_into_bounds,
)
from app.scanners.functions_scanner import FunctionScanner, SurfaceFunctionScanner
from app.styles.style import (
    CardFrame,
    TitledCard,
    create_action_button,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils import report_html
from app.utils.i18n import _
from app.utils.messages import show_message


def _bound_text(value: float) -> str:
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{float(value):.12g}"


def _merge_points_with_errors(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop points without a usable error, sort by x, merge repeated x.

    What prepare_input_xy does, done first so each point keeps its error:
    repeated x become their weighted mean, with the error of that mean.
    """
    x, y, sigma = (np.asarray(values, dtype=float) for values in (x, y, sigma))
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0.0)
    dropped = int(usable.size - np.count_nonzero(usable))
    if dropped:
        applogger.warning(f"{label}: {dropped} point(s) without a positive, finite error left out of the fit.")
    x, y, sigma = x[usable], y[usable], sigma[usable]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size == x.size:
        order = np.argsort(x, kind="stable")
        return x[order], y[order], sigma[order]
    weight = 1.0 / np.square(sigma)
    total = np.bincount(inverse, weights=weight)
    merged_y = np.bincount(inverse, weights=weight * y) / total
    applogger.warning(f"{label}: {x.size - unique_x.size} repeated x merged into their weighted mean.")
    return unique_x, merged_y, 1.0 / np.sqrt(total)


class _StoredFunctionGuess:
    """A stored SymPy function's initial_guess, as a fit class has one: app.analysis.initial_guess's estimator."""

    def __init__(self, payload: dict, functions: dict | None = None) -> None:
        self._payload, self._functions = payload, functions

    def initial_guess(self, x: Any, y: Any) -> list[float] | None:
        from app.analysis.initial_guess import generic_guess

        payload = self._payload
        if str(payload.get("source_type", "")).lower() != "expression":
            return None  # code: nothing to read the parameters' roles from
        return generic_guess(str(payload["code_string"]), list(payload["params"]), list(payload["p0"]), x, y,
                             functions=self._functions)


class SeriesFitDialog(DataOperationDialogBase):
    """Fit a model to a series, or draw the parameters as they stand.

    Two verbs, deliberately separate:

    ``Fit``
        Optimises, writes the optimum into the parameter table, reports it,
        and previews the result.
    ``Preview`` / ``Apply``
        Draw the parameters currently in the table, whatever their origin - a
        fit, a hand-typed guess, or a value copied from a paper.

    Preview used to re-fit, which made a starting guess impossible to see: the
    optimiser replaced it before anything reached the chart.
    """

    #: The curve on its data's axis by default; a new axis or figure as the Charts tab says.
    DESTINATIONS = (DataOperationDialogBase.DEST_SAME_AXIS, DataOperationDialogBase.DEST_NEW_AXIS,
                    DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_SAME_AXIS
    Name: str = "Fit Curve"
    #: The fitted values and residuals are written to a curve's table: its name is on the Graphics page.
    SHOW_OUTPUT: ClassVar[bool] = True
    Description = "Fit data models"

    # least_squares needs residuals that actually vary: a flat y makes the
    # Jacobian singular, and the optimiser returns the starting guess with a
    # success flag rather than reporting that there was nothing to fit.
    #
    # Sorting and duplicate merging are declared because this dialog has always
    # done both - it sorted by x and averaged repeated x before fitting. A fit
    # does not strictly need either, but the declaration has to describe what
    # the code actually does to the data, or the report is a lie.
    INPUT_REQUIRES_VARYING_Y = True
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 2

    #: The Charts tab: the curve, its bands, the extra charts. The accessory charts cost a new axis
    #: each, so neither is drawn unless asked: judging a fit needs more than the curve on the data.
    PARAMS = (
        IntParam("curve_points", "Points:", default_value=200, minimum=0, maximum=100_000, step=100,
                 special_text="At the data", tab="charts", section="Fitted curve",
                 tooltip="How many points the fitted curve is drawn with, evenly spaced; "
                         "0 draws it only at the data's x."),
        FloatParam("curve_extend", "Extend:", default_value=0.0, minimum=0.0, maximum=1000.0, decimals=0,
                   step=10.0, suffix=" %", tab="charts", section="Fitted curve",
                   tooltip="Draw the curve past the data, on each side, by this share of "
                           "the data's x range: 10 % extrapolates a tenth either way."),
        BoolParam("confidence_band", "95% confidence band", tab="charts", section="Bands on the fit",
                  tooltip="Shade where the true curve lies with 95% confidence, given the parameters' "
                          "uncertainty. Narrow where the data pin the curve down, wide where they do not."),
        BoolParam("prediction_band", "95% prediction band", tab="charts", section="Bands on the fit",
                  tooltip="Shade where a new measurement would fall with 95% probability: the confidence "
                          "band plus the scatter of the points around the curve."),
        BoolParam("residual_chart", "Residuals vs. x", tab="charts", section="Extra charts",
                  tooltip="Add a chart of the fit residuals (measured minus fit) on a new axis in this figure. "
                          "The classic way to see a fit that looks good but has structure in what it left behind."),
        BoolParam("fit_vs_measured_chart", "Measured vs. fit", tab="charts", section="Extra charts",
                  tooltip="Add a chart of the measured values against the fitted values on a new axis in this "
                          "figure. A perfect fit falls on the diagonal; how far points stray from it is the fit "
                          "quality, independent of what x means."),
    )

    Icon = """
    <path d="M4 18.5h16"/>
    <path d="M4.5 18V5"/>
    <path d="M6.5 15.5c2.2-5.6 5.2-7.8 11-7.6"/>
    <circle cx="7" cy="15" r="1.2"/>
    <circle cx="11" cy="10.8" r="1.2"/>
    <circle cx="16.5" cy="8" r="1.2"/>
    """
    Keywords: str = (
        "fit fit curve adatta curva adattamento curve fitting curva regression regressione polynomial polinomio exponential "
        "esponenziale gaussian gaussiana sine model modello nonlinear non lineare trend tendenza"
    )
    Goals: ClassVar[list[str]] = ["relate", "predict"]
    Category: str = "Predictive and Specialized Modeling"
    Order: int = 34
    saved = Signal(str)

    WINDOW_SIZE = (900, 560)

    def how_to_read(self, results: Sequence[Any]) -> list[str]:
        result = results[0]
        metrics = getattr(result, "metrics", None) or {}
        y = str(getattr(result, "source_table", "") or "y")
        return [reading_guide.r_squared(metrics.get("r2"), y=y), reading_guide.rmse(metrics.get("rmse"), y=y)]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._refresh_default_output_name(force=True)

    def init_operation_widgets(self) -> None:
        """Create fit controls before base builder hooks run.

        The base dialog invokes build_model_selector() and
        build_parameter_selector() during super().__init__().  Therefore every
        widget used by those builders must be created here.
        """
        self._selected_model: dict[str, Any] = {}
        self._function_scanner = FunctionScanner()
        # Surface (z = f(x, y)) functions live in a second scanner rather
        # than a second base class check sprinkled through this file: see
        # SurfaceFunctionScanner's own docstring for why it is a subclass
        # rather than a duplicate of FunctionScanner.
        self._surface_scanner = SurfaceFunctionScanner()
        self._param_defaults: list[float] = [0.0, 1.0]
        self._last_result: SeriesFitResult | None = None
        self._source_name = "Selected series"
        self._source_x_col = "x"
        self._source_x2_col: str | None = None
        self._source_y_col = "y"

        self._model_search = QLineEdit(self)
        self._models_tree = QTreeWidget(self)
        self._function_expression_html = QLabel("", self)
        self._function_expression_html.setTextFormat(Qt.TextFormat.RichText)
        self._function_expression_html.setWordWrap(True)
        self._function_expression_html.setOpenExternalLinks(False)
        self._function_expression_html.setProperty("muted", True)
        # What each p[i] means for the selected model, shown under the formula.
        self._param_names: list[str] = []
        self._param_legend = QLabel("", self)
        self._param_legend.setWordWrap(True)
        self._param_legend.setProperty("muted", True)
        self._param_legend.hide()

        self._multi_family_combo = QComboBox(self)
        self._multi_family_combo.addItems(["Gaussian", "Lorentzian", "Pseudo-Voigt"])
        self._multi_count_spin = QSpinBox(self)
        self._multi_count_spin.setRange(1, 12)
        self._multi_count_spin.setValue(2)
        self._multi_tie_width = QCheckBox(_("Tie width"), self)
        self._multi_tie_eta = QCheckBox(_("Tie eta"), self)
        self._multi_apply_btn = QPushButton(_("Apply multi-peak"), self)

        self._spin_params = QSpinBox(self)
        self._params_table = QTableWidget(0, 5, self)
        self._params_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._params_table.customContextMenuRequested.connect(self._show_params_context_menu)
        self._results_table = QTableWidget(0, 3, self)
        # Legacy/internal table used only as a data cache for compatibility.
        # It is no longer part of the visible layout; hide it immediately so it
        # does not appear as an orphan widget in the dialog's top-left corner.
        self._results_table.hide()
        self._results_table.setVisible(False)
        self._max_nfev_edit = QLineEdit("800", self)
        self._combo_weights = QComboBox(self)
        self._combo_optimizer = QComboBox(self)
        self._combo_loss = QComboBox(self)
        self._btn_estimate = QPushButton(_("Estimate"), self)
        self._btn_estimate.clicked.connect(self.on_estimate_initial_values)
        self._btn_use_fit_params = create_action_button(
                                       parent=self,
                                       action_id="copy",
                                       action=self.on_use_fit_results_as_initial,
                                   )
        self._btn_use_fit_params.setEnabled(False)
        self._btn_use_fit_params.hide()
        self._lbl_degree = QLabel(_("Degree:"), self)
        self._spin_degree = QSpinBox(self)
        self._lbl_knots = QLabel(_("Knots:"), self)
        self._spin_knots = QSpinBox(self)
        self._lbl_spacing = QLabel(_("Spacing:"), self)
        self._combo_knot_spacing = QComboBox(self)

        self._residual_axis_id: int | None = None
        self._fit_vs_measured_axis_id: int | None = None
        self._accessory_table_names: set[str] = set()
        self._applied = False

    def build_model_selector(self) -> QWidget:
        panel = CardFrame(self, "fitModelCard")
        layout = panel.layout()
        self._model_search.setPlaceholderText(_("Search models..."))
        self._model_search.setToolTip(_("Filter the model catalog by name."))
        layout.addWidget(self._model_search)
        self._models_tree.setHeaderHidden(True)
        self._models_tree.setMinimumHeight(170)
        self._models_tree.setToolTip(_("Pick a scanned function model to populate the formula and parameters."))
        self._models_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mark_editor_panel(self._models_tree)
        layout.addWidget(self._models_tree)

        multi_box = QWidget(panel)
        multi_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        multi_layout = QVBoxLayout(multi_box)
        stdSizeAndlayout(multi_layout)
        multi_title = QLabel(_("Multi-peak builder"), multi_box)
        multi_title.setProperty("muted", True)
        multi_layout.addWidget(multi_title)

        family_row = QHBoxLayout()
        stdSizeAndlayout(family_row)
        family_row.addWidget(QLabel(_("Family:"), multi_box))
        family_row.addWidget(self._multi_family_combo)
        family_row.addWidget(QLabel(_("Peaks:"), multi_box))
        family_row.addWidget(self._multi_count_spin)
        family_row.addStretch(1)
        multi_layout.addLayout(family_row)

        tie_row = QHBoxLayout()
        stdSizeAndlayout(tie_row)
        tie_row.addWidget(self._multi_tie_width)
        tie_row.addWidget(self._multi_tie_eta)
        tie_row.addStretch(1)
        multi_layout.addLayout(tie_row)

        apply_row = QHBoxLayout()
        stdSizeAndlayout(apply_row)
        apply_row.addWidget(self._multi_apply_btn)
        apply_row.addStretch(1)
        multi_layout.addLayout(apply_row)

        layout.addWidget(multi_box)
        layout.setStretchFactor(self._models_tree, 1)
        layout.setStretchFactor(multi_box, 0)
        return panel

    def build_parameter_selector(self) -> QWidget:
        return self._build_parameters_panel()

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(
            lambda *_args: self._refresh_default_output_name(force=True)
        )
        self.series_selector.selection_changed.connect(lambda *_args: self._relayout_params_table())
        self._model_search.textChanged.connect(self._filter_model_catalog)
        self._multi_apply_btn.clicked.connect(self.on_multi_peak)
        self._models_tree.itemSelectionChanged.connect(self._on_model_tree_selection)
        self._spin_degree.valueChanged.connect(self._on_degree_changed)
        self._spin_knots.valueChanged.connect(self._on_knots_changed)
        self._build_model_catalog()
        self._select_first_model()

    def _build_parameters_panel(self) -> QWidget:
        """Build the main Parameters panel without nested collapsible panels.

        The shared Axis / Series collapsible panel is the data source selector.
        This panel only contains expression/model options, parameter editor,
        and output/fit options.
        """
        outer = CardFrame(self, "fitParamsCard")
        layout = outer.layout()

        # Expression / model options.
        option_row = QWidget(outer)
        option_layout = QHBoxLayout(option_row)
        stdSizeAndlayout(option_layout)

        self._spin_degree.setRange(1, 48)
        self._spin_degree.setValue(4)
        self._spin_degree.setToolTip(_("Polynomial degree (for polynomial-family models)."))
        self._spin_knots.setRange(2, 128)
        self._spin_knots.setValue(5)
        self._spin_knots.setToolTip(_("Number of knots (for knot-based spline models)."))
        self._combo_knot_spacing.addItems(["Quantiles", "Linear"])
        self._combo_knot_spacing.setToolTip(_("How knots are distributed along X."))

        for widget in (
            self._lbl_degree,
            self._spin_degree,
            self._lbl_knots,
            self._spin_knots,
            self._lbl_spacing,
            self._combo_knot_spacing,
        ):
            option_layout.addWidget(widget)
        option_layout.addStretch(1)
        layout.addWidget(option_row)

        layout.addWidget(QLabel(_("Expression:"), outer))
        layout.addWidget(self._function_expression_html)
        layout.addWidget(self._param_legend)

        # Parameter editor. Actions are available from the table context menu.
        self._spin_params.setRange(1, 256)
        self._spin_params.hide()

        mark_editor_panel(self._params_table)
        self._params_table.setHorizontalHeaderLabels(
            [_("Parameter"), _("Initial"), _("Lower"), _("Upper"), _("Fix")])
        # The columns share the panel's width - no horizontal scroll bar for
        # five short columns; Fix only as wide as its box.
        header = self._params_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(1, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._params_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._params_table.setMinimumHeight(160)
        self._params_table.setToolTip(
            _("Initial value, bounds and fixed flag for each function parameter. Apply runs the fit with these settings.")
        )
        layout.addWidget(self._params_table, 1)
        layout.addWidget(self._build_shared_row())

        # Output / fit options, one control per row: a translated algorithm
        # label ("Regione di fiducia (predefinito)") and a translated loss
        # label ("Arcotangente (molto robusta)") side by side do not fit this
        # panel's width at all, let alone alongside their own labels - the
        # single row this used to be overflowed it horizontally, and two
        # combos paired per row still did. A QFormLayout gives each combo the
        # whole row to grow into, which is what a translated label this long
        # actually needs.
        # Closed (todo H-10): the default algorithm, loss and weights fit nearly
        # every curve; the arrow opens them.
        fit_options = TitledCard(outer, _("Advanced"), collapsible=True, collapsed=True)
        fit_options_layout = QFormLayout()
        stdSizeAndlayout(fit_options_layout)
        fit_options.card.layout().addLayout(fit_options_layout)

        for optimizer in OPTIMIZERS:
            self._combo_optimizer.addItem(_(optimizer.label), optimizer.key)
        self._combo_optimizer.setCurrentIndex(
            max(0, self._combo_optimizer.findData(DEFAULT_OPTIMIZER))
        )
        self._combo_optimizer.currentIndexChanged.connect(self._refresh_optimizer_help)
        fit_options_layout.addRow(_("Algorithm:"), self._combo_optimizer)

        for key, label in LOSSES:
            self._combo_loss.addItem(_(label), key)
        self._combo_loss.setToolTip(
            _(
                "How a large residual is weighted. Least squares trusts every "
                "point; the robust losses cap what one bad point can do to the "
                "fit. Only the least-squares algorithms use it."
            )
        )
        fit_options_layout.addRow(_("Loss:"), self._combo_loss)

        run_row = QWidget(fit_options)
        run_row_layout = QHBoxLayout(run_row)
        stdSizeAndlayout(run_row_layout)
        self._max_nfev_edit.setMaximumWidth(90)
        self._max_nfev_edit.setToolTip(_("Maximum number of function evaluations for the optimizer."))
        run_row_layout.addWidget(self._max_nfev_edit)
        run_row_layout.addStretch(1)
        fit_options_layout.addRow(_("Max evals:"), run_row)

        for key, label in (("none", "Equal"), ("relative", "Relative (1/|y|)"), ("errors", "Error bars (yerr)")):
            self._combo_weights.addItem(_(label), key)
        self._combo_weights.setToolTip(
            _(
                "How much each point counts. Relative: each residual divided by "
                "its value, for data spanning orders of magnitude. Error bars: "
                "each residual divided by the series' yerr column - its points' "
                "own errors - so a precise point counts more."
            )
        )
        fit_options_layout.addRow(_("Weights:"), self._combo_weights)

        self._btn_estimate.setToolTip(
            _(
                "Fill the Initial column from the data: the function's own "
                "estimator when it has one, a Monte Carlo search of the "
                "parameter space when it does not."
            )
        )
        estimate_row = QHBoxLayout()
        estimate_row.addWidget(self._btn_estimate)
        estimate_row.addStretch(1)
        layout.addLayout(estimate_row)
        layout.addWidget(fit_options)
        self._refresh_optimizer_help()
        return outer

    # ==================================================================
    # AccessoryCharts: Accessory charts (residuals, measured vs. fit) and the parameter table's menu.
    # ==================================================================

    def _residual_axis_query(
        self, result: SeriesFitResult, table_name: str
    ) -> tuple[str, dict[str, str], str, str]:
        return (
            # Measured rows only: the curve's own rows (Fitted curve > Points) have no residual.
            f'SELECT x, residual AS y FROM "{table_name}" WHERE residual IS NOT NULL ORDER BY x',
            {"x": "x", "y": "y"},
            result.x_col or "x",
            _("residual"),
        )

    def _fit_vs_measured_axis_query(
        self, result: SeriesFitResult, table_name: str
    ) -> tuple[str, dict[str, str], str, str]:
        fit_col, measured_col = ("z_fit", "z") if self._is_2d_fit() else ("y_fit", "y")
        return (
            f'SELECT {fit_col} AS x, {measured_col} AS y FROM "{table_name}" WHERE {measured_col} IS NOT NULL',
            {"x": "x", "y": "y"},
            _("fit"),
            result.target_col or _("measured"),
        )

    def _sync_accessory_axis(
        self,
        *,
        wanted: bool,
        axis_id_attr: str,
        title: str,
        query_builder: Callable[[SeriesFitResult, str], tuple[str, dict[str, str], str, str]],
        results: Sequence[SeriesFitResult],
    ) -> None:
        if not wanted:
            self._remove_accessory_axis(axis_id_attr)
            return
        first = results[0]
        first_table = self.result_table_name(0, first)
        _query, _roles, x_label, y_label = query_builder(first, first_table)
        axis_id = getattr(self, axis_id_attr)
        if axis_id is not None and not self._axis_exists(int(axis_id)):
            # Gone with the figure or the Preview it was made in: a series added to it was "FOREIGN KEY
            # constraint failed", and nothing drawn.
            axis_id = None
        if axis_id is None:
            # Beside the curve: on the new figure when it went to one.
            axis_id = self.create_result_axis(
                chart_type="Scatter Plot", title=title, x_label=x_label,
                y_label=y_label, options={"grid": True}, figure_id=getattr(self, "_result_figure_id", None),
            )
            setattr(self, axis_id_attr, axis_id)
        for row in self._repo.get_series(axis_id) or []:
            self._repo.delete_series(int(row["id"]))
        for result in results:
            table_name = self.result_table_name(0, result)
            sql_query, roles, _x_label, _y_label = query_builder(result, table_name)
            self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=f"{title}: {result.source_table}",
                sql_query=sql_query, roles=roles,
                style={"marker": "o", "linestyle": "",
                       "source_series": result.source_table,
                       "generated_fit": True, "fit_dialog": "series_fit"},
            )

    def _remove_accessory_axis(self, axis_id_attr: str) -> None:
        axis_id = getattr(self, axis_id_attr, None)
        if axis_id is None:
            return
        setattr(self, axis_id_attr, None)
        try:
            self._repo.delete_axis(int(axis_id))
        except Exception:
            applogger.exception("Failed to remove accessory axis (%s)", axis_id_attr)

    def _sync_accessory_charts(self, results: Sequence[SeriesFitResult]) -> None:
        if not results:
            return
        if not (self.chart_enabled("residual_chart") or self.chart_enabled("fit_vs_measured_chart")):
            self._remove_accessory_axis("_residual_axis_id")
            self._remove_accessory_axis("_fit_vs_measured_axis_id")
            return
        for result in results:
            table_name = self.result_table_name(0, result)
            self.write_result_table(table_name, result)
            self._accessory_table_names.add(table_name)
        self._sync_accessory_axis(
            wanted=self.chart_enabled("residual_chart"), axis_id_attr="_residual_axis_id",
            title=_("Residuals"), query_builder=self._residual_axis_query, results=results,
        )
        self._sync_accessory_axis(
            wanted=self.chart_enabled("fit_vs_measured_chart"), axis_id_attr="_fit_vs_measured_axis_id",
            title=_("Measured vs. fit"), query_builder=self._fit_vs_measured_axis_query, results=results,
        )

    def resolve_target_axis_id(
        self, selected_axis_id: int, results: Sequence[SeriesFitResult]
    ) -> int:
        """Keep the fit on its source axis, and sync the accessory axes here.

        The fit overlay itself belongs on the axis its data came from, so the
        selected axis is returned unchanged. The accessory axes are built in
        this hook - and *only* here - because ``_run_operation`` calls it
        before ``begin_preview_transaction()``, which is the one point where
        an axis can be created safely.

        Creating them inside the savepoint instead destroyed data: writing a
        result table goes through pandas' to_sql, which commits and so
        invalidates the savepoint, after which every later repository write
        has its own commit suppressed (see SqliteRepo._commit) and sits in an
        implicit transaction that the Close/Cancel cleanup then discards -
        taking the user's own series on the other axes with it. The spectral
        and statistics dialogs create their result axis in this same hook for
        exactly this reason.
        """
        # "Draw on": the source axis (the default), a new axis, or a new figure - the base's choice.
        target = super().resolve_target_axis_id(selected_axis_id, results)  # type: ignore[misc]
        if results:
            self._sync_accessory_charts(results)
        return target

    def apply_results_to_axis(self, axis_id: int, results: Sequence[SeriesFitResult]) -> None:
        super().apply_results_to_axis(axis_id, results)
        self._applied = True

    def discard_operation_artifacts(self) -> None:
        """Remove any accessory axis this dialog added, when Apply never ran.

        Creating an axis commits, so it is not covered by the preview
        savepoint and has to be undone by hand - see resolve_target_axis_id.
        """
        if self._applied:
            return
        self._remove_accessory_axis("_residual_axis_id")
        self._remove_accessory_axis("_fit_vs_measured_axis_id")

        table_names = set(self._accessory_table_names)
        self._accessory_table_names.clear()
        # And any axis that reads only those tables, its id lost or not: left behind, it was an empty
        # Residuals and an empty Measured vs. fit after Cancel.
        self._remove_axes_reading(table_names)
        super().discard_operation_artifacts()  # type: ignore[misc] - the new axis or figure "Draw on" made
        for table_name in table_names:
            try:
                self._repo.delete_table(table_name)
            except Exception:
                applogger.exception(
                    "Failed to remove the accessory result table %s", table_name
                )

    def _axis_exists(self, axis_id: int) -> bool:
        figures = {self._figure_id, getattr(self, "_result_figure_id", None)} - {None}
        for figure_id in figures:
            descriptor = self._repo.load_figure_descriptor(int(figure_id))
            if descriptor is not None and any(int(axis.id) == axis_id for axis in descriptor.axes):
                return True
        return False

    def _remove_axes_reading(self, table_names: set[str]) -> None:
        """Delete the figure's axes whose every series reads one of *table_names*, then fit its grid to the rest."""
        if not table_names:
            return
        try:
            descriptor = self._repo.load_figure_descriptor(self._figure_id)
            doomed = [axis for axis in (descriptor.axes if descriptor else [])
                      if axis.series and all(any(f'"{name}"' in str(series.sql_query or "") for name in table_names)
                                             for series in axis.series)]
            for axis in doomed:
                self._repo.delete_axis(int(axis.id))
            if descriptor is not None and doomed:
                left = len(descriptor.axes) - len(doomed)
                columns = max(1, min(int(descriptor.ncols or 1), left))
                self._repo.set_figure_grid(self._figure_id, nrows=max(1, -(-left // columns)), ncols=columns)
        except Exception:
            applogger.exception("Failed to remove the accessory axes left on the figure")

    def on_use_fit_results_as_initial(self) -> None:
        """Copy latest optimized fit parameters back to the Initial column."""
        if self._last_result is None:
            show_message(self, "series.no_fit_results")
            return
        # Every row of the table: in a global fit, more than one series' worth.
        params = np.asarray(getattr(self, "_last_table_params", self._last_result.params), dtype=float)
        if params.size == 0:
            show_message(self, "series.no_parameters")
            return
        self._spin_params.blockSignals(True)
        try:
            self._spin_params.setValue(int(params.size))
            self._ensure_params_rows(int(params.size))
        finally:
            self._spin_params.blockSignals(False)
        for row, value in enumerate(params):
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))
            if self._params_table.item(row, 2) is None:
                self._params_table.setItem(row, 2, QTableWidgetItem("-inf"))
            if self._params_table.item(row, 3) is None:
                self._params_table.setItem(row, 3, QTableWidgetItem("inf"))
            self._get_fix_checkbox(row).setChecked(False)
        applogger.info("Initial parameters updated from latest fit results.")

    def _cell_text(self, row: int, col: int, default: str = "") -> str:
        item = self._params_table.item(row, col)
        if item is None:
            return default
        return item.text()

    def _params_as_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row in range(self._params_table.rowCount()):
            fix_widget = self._params_table.cellWidget(row, 4)
            fixed = bool(fix_widget.isChecked()) if isinstance(fix_widget, QCheckBox) else False
            rows.append(
                {
                    "parameter": self._cell_text(row, 0, f"p[{row}]"),
                    "initial": self._cell_text(row, 1),
                    "lower": self._cell_text(row, 2),
                    "upper": self._cell_text(row, 3),
                    "fixed": fixed,
                }
            )
        return pd.DataFrame(rows)

    def _copy_params_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self._params_as_frame().to_csv(index=False, sep="\t"))
        applogger.info("Parameter table copied to clipboard.")

    def _save_params_as_csv(self) -> None:
        default_name = f"{self._model_name().replace(' ', '_').lower()}_parameters.csv"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            _("Save parameters as CSV"),
            default_name,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self._params_as_frame().to_csv(path, index=False)
        applogger.info("Parameter table saved to %s", path)

    def _show_params_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self._params_table)
        copy_action = QAction(_("Copy parameters"), menu)
        copy_action.triggered.connect(self._copy_params_to_clipboard)
        menu.addAction(copy_action)

        use_fit_action = QAction(_("Use latest fitted parameters"), menu)
        use_fit_action.setEnabled(self._last_result is not None)
        use_fit_action.triggered.connect(self.on_use_fit_results_as_initial)
        menu.addAction(use_fit_action)

        reset_action = QAction(_("Reset parameters"), menu)
        reset_action.triggered.connect(self.on_reset_params)
        menu.addAction(reset_action)

        menu.addSeparator()
        save_csv_action = QAction(_("Save as CSV"), menu)
        save_csv_action.triggered.connect(self._save_params_as_csv)
        menu.addAction(save_csv_action)

        menu.exec(self._params_table.viewport().mapToGlobal(pos))

    def on_reset_params(self) -> None:
        self._populate_params_defaults()
        applogger.info("Parameter table reset to model defaults.")

    # ==================================================================
    # FitHooks: What the fit gives DataOperationDialogBase: results, result tables and series.
    # ==================================================================

    def compute_results(self) -> Sequence[SeriesFitResult]:
        """Evaluate the model at the parameters currently in the table.

        Preview and Apply both come through here, and neither optimises: that
        is what the Fit button is for.  Preview used to re-fit, which meant a
        hand-edited starting guess could never be seen - the optimiser
        overwrote it before anything was drawn.
        """
        self._last_result = None
        return super().compute_results()

    def result_table_name(self, axis_id: int, result: SeriesFitResult) -> str:
        del axis_id
        return result.output_table

    def result_series_spec(self, axis_id: int, table_name: str, result: SeriesFitResult) -> ResultSeriesSpec:
        del axis_id
        name = f"Fit: {result.source_table} [{result.model_name}]"

        if result.fit_mode == "2D":
            # Drawn with plot_trisurf (see TriSurfaceAxisRenderer), not
            # pivoted onto a rectangular grid: the source points a surface
            # fit runs on are rarely a complete x/y grid (that is exactly
            # what series_grid_xyz's own fallback to interpolation is for
            # elsewhere in this task), and triangulating the fitted z values
            # directly draws a continuous skin without inventing a grid the
            # data never supported.
            sql_query = f'SELECT x, y, z_fit AS z FROM "{table_name}"'
            roles = {"x": "x", "y": "y", "z": "z"}
            style = {
                "source_series": result.source_table,
                "source_x_col": result.x_col,
                "source_x2_col": result.x2_col or "",
                "source_z_col": result.target_col,
                "fit_model": result.model_name,
                "fit_mode": "2D",
                **self.generated_style_filter,
            }
        else:
            sql_query = f'SELECT x, y_fit AS y FROM "{table_name}" ORDER BY x'
            roles = {"x": "x", "y": "y"}
            style = {
                "linestyle": "--",
                "linewidth": 2.0,
                "marker": "",
                "source_series": result.source_table,
                "source_x_col": result.x_col,
                "source_y_col": result.target_col,
                "fit_model": result.model_name,
                "fit_mode": "1D",
                **self.generated_style_filter,
            }

        return ResultSeriesSpec(
            name=name,
            sql_query=sql_query,
            roles=roles,
            style=style,
        )

    def result_series_specs(
        self, axis_id: int, table_name: str, result: SeriesFitResult
    ) -> Sequence[ResultSeriesSpec]:
        """The fitted curve, plus the bands whose boxes are checked (1D only)."""
        specs = [self.result_series_spec(axis_id, table_name, result)]
        if result.fit_mode == "2D" or "ci_low" not in result.frame:
            return specs
        for chart, kind, label, alpha in (
            ("prediction_band", "pi", "95% prediction band", 0.12),
            ("confidence_band", "ci", "95% confidence band", 0.25),
        ):
            if not self.chart_enabled(chart):
                continue
            specs.append(
                ResultSeriesSpec(
                    name=f"{label}: {result.source_table} [{result.model_name}]",
                    # Aliased to x/y/y2 rather than mapped through roles:
                    # the axis's own chart type (a scatter) keeps only the
                    # roles it knows, and y2 is not one of them.
                    sql_query=(
                        f'SELECT x, {kind}_low AS y, {kind}_high AS y2 '
                        f'FROM "{table_name}" ORDER BY x'
                    ),
                    roles={"x": "x", "y": "y", "y2": "y2"},
                    style={
                        "draw_as": "band",
                        "alpha": alpha,
                        "label": label,
                        **self.generated_style_filter,
                        "fit_model": result.model_name,
                    },
                )
            )
        return specs

    def format_results(self, results: Sequence[SeriesFitResult]) -> str:
        """Return the fit report as an HTML table."""
        if not results:
            return "<p>Run Fit, or press Preview to draw the current parameters.</p>"
        if len(results) == 1:
            return self._results_html(results[0])
        # One document for all the series, each under its own heading: a
        # report per series pasted one after the other nested one <html> and
        # <body> in the next.
        sections: list[str] = []
        for result in results:
            sections.append(report_html.section(f"{_('Series')}: {result.source_table}"))
            sections.extend(self._report_sections(result))
        return self.results_document(results[0].model_name, *sections)

    # ==================================================================
    # FitRun: Running the fit: the job, its result, the results it reports.
    # ==================================================================

    def on_fit(self) -> None:
        """Optimise the parameters in the background, then show and preview them.

        Fit and Preview are deliberately different acts.  Preview draws the
        parameters currently in the table - which is how you try a starting
        guess, or hand-tune one, and see it immediately.  Fit is what changes
        those parameters: it optimises, writes the optimum back into the table
        so the table always says what is drawn, and then previews that.

        The optimisation goes through the base class's evaluate() on a worker
        thread, with its Stop button and its stop on close.
        """
        self.evaluate(self._after_fit, background=True, optimise=True)

    def _after_fit(self, results: Sequence[SeriesFitResult]) -> None:
        """Back on the GUI thread with the optimum: table, report, preview."""
        if not results:
            return
        result = results[0]
        # The table is the single source of truth for what gets drawn, so the
        # optimum has to land in it rather than only in the report - every
        # row of it, which in a global fit is more than one series' worth.
        self._set_initial_params(np.asarray(self._last_table_params, dtype=float))
        self._fill_results_table(result.params, result.param_std)
        self._btn_use_fit_params.setEnabled(True)
        self.publish_results(self.results_report(results))

        # A fit nobody can see is only half an answer.
        self.preview()

    def busy_widgets(self) -> list[QWidget]:
        return [*super().busy_widgets(), self.fit_button, self._btn_estimate]

    def prepare_job(self, *, optimise: bool = False, **options: Any) -> _FitJob | None:
        """Read the data, the model and the parameter table into a job."""
        x_data, target_data, clean = self._load_fit_data()
        if x_data is None or target_data is None:
            return None
        built = self._build_model(x_data, target_data)
        if built is None:
            return None
        model, _model_p0 = built
        p0, lb, ub, fixed = self._collect_params_from_table()
        global_parts = None
        layout = getattr(self, "_table_layout", None)
        if layout is not None and x_data.ndim == 2:
            global_parts = (model, layout, list(self._global_names))
            model = fit_engine.global_model(model, layout)
        x_col, x2_col, target_col = self._selected_column_names()
        return _FitJob(
            model=model,
            x=x_data,
            target=target_data,
            clean=clean,
            p0=p0,
            lower=lb,
            upper=ub,
            fixed=fixed,
            optimizer=self._optimizer_key(),
            loss=self._loss_key(),
            max_nfev=int(float(self._max_nfev_edit.text().strip() or "800")),
            weighted=self._weights_key() == "relative",
            source_table=self._current_table(),
            x_col=x_col,
            x2_col=x2_col,
            target_col=target_col,
            is_2d=self._is_2d_fit(),
            model_name=self._model_name(),
            param_names=[self._param_label(row) for row in range(len(p0))],
            expression=str(self._selected_model.get("expression", "")).strip(),
            output_table=self._output_table_name(),
            optimise=optimise,
            sigma=clean["sigma"].to_numpy(float) if "sigma" in clean.columns else None,
            global_parts=global_parts,
            separate=global_parts is not None and not self._fit_is_global(),
        )

    def _weights_key(self) -> str:
        return str(self._combo_weights.currentData() or "none")

    def finish_job(self, job: OperationJob, outcome: fit_engine.CurveFit) -> list[SeriesFitResult]:
        """The engine's outcome as this dialog's result; kept as _last_result."""
        assert isinstance(job, _FitJob)  # the job prepare_job made
        if job.optimise and outcome.success:
            # Remembered so that Preview/OK, which only evaluate the table,
            # can still say the parameters they draw are a converged fit.
            self._last_fit = (job.model_name, np.asarray(outcome.params, dtype=float), outcome.message)
        elif not job.optimise:
            outcome.message = self._evaluation_message(job, outcome)
        self._last_table_params = np.asarray(outcome.params, dtype=float)
        parts = self._split_global(job, outcome) if job.global_parts is not None else [(job, outcome)]
        try:
            results = [self._result_from(part_job, part, optimise=job.optimise) for part_job, part in parts]
        except Exception:
            self._last_result = None
            raise
        self._last_result = results[0]
        return results

    def _evaluation_message(self, job: _FitJob, outcome: fit_engine.CurveFit) -> str:
        """What an evaluation reports: the fit's own status while the table
        still holds that fit's optimum, "evaluated" once anything differs.

        Preview and OK draw the table, never re-fit; after Fit the table is
        the optimum, and a saved report that said only "evaluated at the
        current parameters" read as though no fit had been run at all.
        """
        last = getattr(self, "_last_fit", None)
        params = np.asarray(outcome.params, dtype=float)
        if (
            last is not None
            and last[0] == job.model_name
            and last[1].shape == params.shape
            and np.allclose(last[1], params, rtol=1e-9, atol=1e-12)
        ):
            return last[2]
        return outcome.message

    def _result_from(
        self, job: _FitJob, outcome: fit_engine.CurveFit, *, optimise: bool
    ) -> SeriesFitResult:
        """Turn the engine's outcome into the dialog's result: frame, bands, report."""
        frame = self._build_output_frame(job, outcome.fit_values, outcome.residual)
        if not job.is_2d and len(frame):
            frame = self._with_curve_rows(job, frame, outcome.params)
        if not job.is_2d and outcome.dof > 0 and len(frame):
            # The band columns ride in the result table like fit and
            # residual do, so a saved result can be redrawn without the
            # dialog; they are drawn only when their box is checked.
            x_band = frame["x"].to_numpy(float)
            for kind, prediction in (("ci", False), ("pi", True)):
                low, high = fit_engine.confidence_band(
                    job.model,
                    x_band,
                    outcome.params,
                    outcome.cov,
                    dof=outcome.dof,
                    residual_variance=outcome.metrics.get("reduced_chi2", 0.0),
                    prediction=prediction,
                )
                frame[f"{kind}_low"] = low
                frame[f"{kind}_high"] = high
        applogger.info(
            "%s %s: %s",
            "Fit" if optimise else "Evaluation",
            "success" if outcome.success else "warning",
            outcome.message,
        )
        applogger.info(self._format_metrics(outcome.metrics))
        return SeriesFitResult(
            source_table=job.source_table,
            x_col=job.x_col,
            x2_col=job.x2_col,
            target_col=job.target_col,
            fit_mode="2D" if job.is_2d else "1D",
            model=job.model_name,
            params=outcome.params,
            param_std=outcome.std,
            param_corr=outcome.corr,
            param_names=list(job.param_names),
            expression=job.expression,
            evaluated_expression=_evaluated_expression_html(
                job.expression, job.model_name, job.param_names, outcome.params
            ),
            metrics=outcome.metrics,
            output_table=job.output_table,
            frame=frame,
            message=outcome.message,
            param_t=outcome.tvalues,
            param_p=outcome.pvalues,
            param_ci_low=outcome.ci_low,
            param_ci_high=outcome.ci_high,
            dof=outcome.dof,
        )

    def _with_curve_rows(self, job: _FitJob, frame: pd.DataFrame, params: np.ndarray) -> pd.DataFrame:
        """*frame* plus rows of the curve alone - evenly spaced x, past the data if asked; no measurement there.

        The curve series reads x and y_fit over every row, so it is drawn smooth and as far as asked; the
        residuals and the fit-vs-measured chart read y and residual, empty on these rows, and skip them.
        """
        count = int(self.chart_value("curve_points", 200))
        extend = float(self.chart_value("curve_extend", 0.0)) / 100.0
        if count <= 0 and extend <= 0:
            return frame
        x = frame["x"].to_numpy(float)
        low, high = float(np.nanmin(x)), float(np.nanmax(x))
        span = high - low
        low, high = low - extend * span, high + extend * span
        dense = np.linspace(low, high, max(count, 2) if count > 0 else 50)
        try:
            fitted = np.asarray(job.model(dense, np.asarray(params, dtype=float)), dtype=float)
        except Exception:
            applogger.warning("The fitted curve could not be drawn between the data: drawn at the data only.")
            return frame
        extra = pd.DataFrame({"x": dense, "fit": fitted, "y_fit": fitted})
        for column in frame.columns:
            if column not in extra and frame[column].dtype == object:
                extra[column] = frame[column].iloc[0]
        return pd.concat([frame, extra], ignore_index=True)

    def build_extra_action_buttons(self, layout) -> None:
        """Add the Fit button next to Preview."""
        self.fit_button = create_action_button(
                              parent=self,
                              action_id="run_fit",
                              action=self.on_fit,
                              layout=layout,
                          )

    # ==================================================================
    # GlobalFit: Global fit with shared parameters.
    # ==================================================================

    def _global_rows(self) -> list[Any]:
        """The checked series when the fit is global (two or more, 1D), else []."""
        if self._is_2d_fit():
            return []
        rows = list(self.selected_series())
        return rows if len(rows) >= 2 else []

    def _fit_is_global(self) -> bool:
        """True when the checked series are fitted together (the default).

        False fits every series on its own: the layout then has no shared
        parameter, so each series gets a row per parameter and its own result.
        """
        return bool(getattr(self, "_global_fit_wanted", True))

    def _set_fit_mode(self, index: int) -> None:
        self._global_fit_wanted = index == 0
        self._relayout_params_table()
        self.mark_results_stale()

    @staticmethod
    def _series_label(row: Any) -> str:
        return str(parse_roles(row_value(row, "roles", default={})).get("name")
                   or row_value(row, "name", "series_name", default="Series"))

    def _wanted_layout(self) -> fit_engine.GlobalLayout | None:
        rows = self._global_rows()
        if not rows:
            return None
        n_params = len(self._param_defaults)
        shared = list(getattr(self, "_shared", []))[:n_params]
        shared += [False] * (n_params - len(shared))
        # What was picked is kept while the series are fitted one by one, so
        # switching back to a global fit finds the same shared parameters.
        self._shared = shared
        if not self._fit_is_global():
            shared = [False] * n_params
        self._global_names = [self._series_label(row) for row in rows]
        return fit_engine.GlobalLayout(tuple(shared), len(rows))

    def _global_label(self, row: int) -> str | None:
        """The table label of *row* in a global fit, None otherwise."""
        layout = getattr(self, "_table_layout", None)
        if layout is None:
            return None
        entries = layout.entries()
        if row >= len(entries):
            return None
        column, series = entries[row]
        name = self._param_names[column] if column < len(self._param_names) else f"p[{column}]"
        if series is None:
            return name
        names = getattr(self, "_global_names", [])
        return f"{name} · {names[series] if series < len(names) else series + 1}"

    def _build_shared_row(self) -> QWidget:
        """'Shared: A, offset' with a menu of the parameters; global fits only."""
        self._shared_row = QWidget(self)
        row_layout = QHBoxLayout(self._shared_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._fit_mode_combo = QComboBox(self._shared_row)
        self._fit_mode_combo.addItem(_("Global fit"))
        self._fit_mode_combo.addItem(_("Each curve separately"))
        self._fit_mode_combo.setToolTip(
            _("Global: one fit of all the checked series, with the parameters picked in Shared "
              "having one value for all. Separately: every series gets its own parameters and result.")
        )
        self._fit_mode_combo.currentIndexChanged.connect(self._set_fit_mode)
        row_layout.addWidget(self._fit_mode_combo)
        row_layout.addWidget(QLabel(_("Shared:"), self._shared_row))
        self._shared_button = QPushButton(self._shared_row)
        self._shared_button.setToolTip(_("One value for every series, fitted to all of them together."))
        self._shared_menu = QMenu(self._shared_button)
        self._shared_button.setMenu(self._shared_menu)
        row_layout.addWidget(self._shared_button, 1)
        self._shared_row.hide()
        return self._shared_row

    def _refresh_shared_row(self, layout: fit_engine.GlobalLayout | None) -> None:
        self._shared_row.setVisible(layout is not None)
        if layout is None:
            return
        self._shared_button.setEnabled(self._fit_is_global())
        # The layout has no shared parameter when fitting separately, so the
        # menu shows what is kept for the global fit instead of the layout.
        shared_now = list(self._shared)
        shared_now += [False] * (len(layout.shared) - len(shared_now))
        names = [
            self._param_names[column] if column < len(self._param_names) else f"p[{column}]"
            for column in range(len(layout.shared))
        ]
        actions = self._shared_menu.actions()
        if [action.text() for action in actions] != names:
            # Rebuilt only for another model: never from inside one of its
            # own actions' toggled signal, which would delete the sender.
            self._shared_menu.clear()
            for column, name in enumerate(names):
                action = self._shared_menu.addAction(name)
                action.setCheckable(True)
                action.toggled.connect(lambda checked, column=column: self._set_shared(column, checked))
            actions = self._shared_menu.actions()
        for action, shared in zip(actions, shared_now):
            blocked = action.blockSignals(True)
            action.setChecked(shared)
            action.blockSignals(blocked)
        names = [action.text() for action in actions if action.isChecked()]
        self._shared_button.setText(", ".join(names) if names else _("none"))

    def _set_shared(self, column: int, shared: bool) -> None:
        self._shared[column] = bool(shared)
        self._relayout_params_table()

    def _relayout_params_table(self) -> None:
        """Lay the table out for the series checked now, keeping its values.

        Values move by (parameter, series): a parameter made shared keeps the
        first series' row, one made unshared copies its value to every series,
        and a series checked anew starts from the first series' values.
        """
        if not hasattr(self, "_param_defaults"):
            return
        new = self._wanted_layout()
        old = getattr(self, "_table_layout", None)
        self._refresh_shared_row(new)
        if new == old:
            return
        columns = self._collect_params_from_table()
        per_series = [values[None, :] if old is None else values[old.index] for values in columns]
        n_new = new.n_series if new is not None else 1
        moved = [
            np.vstack([values[min(series, values.shape[0] - 1)] for series in range(n_new)])
            for values in per_series
        ]
        rows = [new.expand(values) if new is not None else values[0] for values in moved]
        self._table_layout = new
        self._ensure_params_rows(len(rows[0]))
        p0, lower, upper, fixed = rows
        for row in range(len(p0)):
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(p0[row]):.12g}"))
            self._params_table.setItem(row, 2, QTableWidgetItem(_bound_text(lower[row])))
            self._params_table.setItem(row, 3, QTableWidgetItem(_bound_text(upper[row])))
            self._get_fix_checkbox(row).setChecked(bool(fixed[row]))
        self.mark_results_stale()

    def _expand_to_table(self, values: np.ndarray) -> np.ndarray:
        """One series' parameter values as the table's rows (the same for all)."""
        layout = getattr(self, "_table_layout", None)
        values = np.asarray(values, dtype=float)
        if layout is None or values.size != len(layout.shared):
            return values
        return layout.expand(np.tile(values, (layout.n_series, 1)))

    def _combine_estimates(self, estimates: list[np.ndarray | None]) -> np.ndarray | None:
        """Each series' own estimate as the table's rows; shared ones averaged."""
        layout = getattr(self, "_table_layout", None)
        if layout is None:
            return None
        current = self._collect_params_from_table()[0]
        per_series = current[layout.index].copy()
        found = [index for index, values in enumerate(estimates) if values is not None]
        if not found:
            return None
        for series in found:
            per_series[series] = estimates[series]
        mean = np.mean([cast(np.ndarray, estimates[series]) for series in found], axis=0)
        theta = layout.expand(per_series)
        for row, (column, series) in enumerate(layout.entries()):
            if series is None:
                theta[row] = mean[column]
        return theta

    def _load_global_fit_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Every checked series, stacked: x as (value, series), clean with a series column."""
        rows = self._global_rows()
        frames = []
        #: Each series' own (x, target) columns: loading a series overwrites
        #: _source_x_col/_source_y_col, so after the loop they name the last one.
        self._global_columns = []
        for number, row in enumerate(rows):
            _x, _y, clean = self._load_fit_data_1d(row)
            self._global_columns.append((self._source_x_col, self._source_y_col))
            clean = clean.copy()
            clean["series"] = number
            clean["series_name"] = self._series_label(row)
            frames.append(clean)
        clean = pd.concat(frames, ignore_index=True)
        x_data = clean[["x", "series"]].to_numpy(dtype=float)
        return x_data, clean["target"].to_numpy(dtype=float), clean

    def _per_series_xy(self) -> list[tuple[np.ndarray, np.ndarray]]:
        _x, _y, clean = self._load_global_fit_data()
        return [
            (part["x"].to_numpy(float), part["target"].to_numpy(float))
            for _number, part in clean.groupby("series", sort=True)
        ]

    def _split_global(self, job: Any, outcome: fit_engine.CurveFit) -> list[tuple[Any, fit_engine.CurveFit]]:
        base_model, layout, names = job.global_parts
        shared_names = [self._param_names[column] for column, shared in enumerate(layout.shared) if shared]
        note = (
            _("Global fit of {count} series") if self._fit_is_global()
            else _("Each of {count} series fitted separately")
        ).format(count=layout.n_series)
        note += (" - " + _("shared: {names}").format(names=", ".join(shared_names))) if shared_names else ""
        series_of_point = np.asarray(job.x, dtype=float)[:, 1].astype(int)
        parts = []
        for number in range(layout.n_series):
            index = layout.index[number]
            mask = series_of_point == number
            target = job.target[mask]
            fit_values = outcome.fit_values[mask]
            if job.parts is not None:
                # Fitted on its own: its own metrics, errors and degrees of freedom.
                own = job.parts[number]
                metrics = dict(own.metrics)
            else:
                metrics = fit_engine.fit_metrics(target, fit_values, int(np.count_nonzero(~job.fixed[index])))
                # Bands use the fit's pooled residual variance, the one its errors come from.
                if "reduced_chi2" in outcome.metrics:
                    metrics["reduced_chi2"] = outcome.metrics["reduced_chi2"]
                for key in ("chi2", "chi2_reduced"):
                    if key in outcome.metrics:
                        metrics[f"global_{key}"] = outcome.metrics[key]
            square = np.ix_(index, index)
            columns = getattr(self, "_global_columns", [])
            x_col, target_col = columns[number] if number < len(columns) else (job.x_col, job.target_col)
            part_job = replace(
                job,
                x_col=x_col or job.x_col,
                target_col=target_col or job.target_col,
                model=base_model,
                x=np.asarray(job.x, dtype=float)[mask, 0],
                target=target,
                clean=job.clean[mask].reset_index(drop=True),
                p0=job.p0[index],
                lower=job.lower[index],
                upper=job.upper[index],
                fixed=job.fixed[index],
                source_table=names[number],
                param_names=list(self._param_names),
                output_table=generated_table_name(f"{job.output_table}_{names[number]}", fallback="Fit_Result"),
                sigma=None if job.sigma is None else job.sigma[mask],
                global_parts=None,
            )
            if job.parts is not None:
                part = replace(job.parts[number], message=f"{note}. {job.parts[number].message}")
                parts.append((part_job, part))
                continue
            part = fit_engine.CurveFit(
                params=outcome.params[index],
                std=outcome.std[index],
                cov=outcome.cov[square],
                corr=outcome.corr[square],
                fit_values=fit_values,
                residual=outcome.residual[mask],
                metrics=metrics,
                success=outcome.success,
                message=f"{note}. {outcome.message}",
                dof=outcome.dof,
                tvalues=outcome.tvalues[index],
                pvalues=outcome.pvalues[index],
                ci_low=outcome.ci_low[index],
                ci_high=outcome.ci_high[index],
            )
            parts.append((part_job, part))
        return parts

    # ==================================================================
    # ModelData: The chosen model, its parameters table, multi-peak models, and the data the fit reads.
    # ==================================================================
    def _catalog_data(self) -> dict[str, list[dict[str, Any]]]:
        """Return scanned fit functions grouped by category.

        The previous JSON catalog is intentionally removed.  All fit models come
        from the SymPy library (app.functions.library) and the example classes in
        app/functions/user_functions.py through FunctionScanner, plus every
        z = f(x, y) surface function of the library and of
        app/functions/user_surface_functions.py through
        SurfaceFunctionScanner. The two catalogs are merged by category - a
        surface function's own categories ("Surfaces", "User surfaces") show
        up as ordinary top-level groups in the same tree, no separate UI
        needed.
        """
        merged: dict[str, list[dict[str, Any]]] = {
            category: list(models) for category, models in self._function_scanner.catalog().items()
        }
        for category, models in self._surface_scanner.catalog().items():
            merged.setdefault(category, []).extend(models)
            merged[category].sort(key=lambda item: str(item.get("name", "")).lower())
        try:
            stored = stored_sympy_catalog(self._repo)
        except Exception:
            applogger.exception("Could not load stored SymPy functions")
            stored = []
        if stored:
            merged["Stored functions"] = sorted(stored, key=lambda item: str(item.get("name", "")).lower())
        return dict(sorted(merged.items(), key=lambda item: item[0].lower()))

    def _build_model_catalog(self) -> None:
        self._models_tree.clear()
        for cat_name, models in self._catalog_data().items():
            # Translated for display only. The payload keeps the English name,
            # which is what a saved fit refers to and what the search below
            # falls back on - a catalogue whose identities changed with the
            # interface language would lose every fit saved in another one.
            cat = QTreeWidgetItem([_(str(cat_name))])
            self._models_tree.addTopLevelItem(cat)
            for payload in models:
                item = QTreeWidgetItem([_(str(payload["name"]))])
                item.setData(0, Qt.ItemDataRole.UserRole, payload)
                cat.addChild(item)
            cat.setExpanded(False)

    def _filter_model_catalog(self, text: str) -> None:
        needle = text.lower().strip()
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            any_visible = False
            if cat is None:
                continue
            for j in range(cat.childCount()):
                child = cat.child(j)
                if child is None:
                    continue
                payload = child.data(0, Qt.ItemDataRole.UserRole)
                words = str(payload.get("keywords", "")) if isinstance(payload, dict) else ""
                # The name, or the words the function declares (its keywords).
                match = needle in child.text(0).lower() or needle in words.lower()
                child.setHidden(not match)
                any_visible = any_visible or match
            cat.setHidden(not any_visible)
            cat.setExpanded(bool(needle) and any_visible)

    def _select_first_model(self) -> None:
        """Select Linear by default, or the first model when there is none."""
        if self.select_model("Linear"):
            return
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            first = cat.child(0) if cat is not None else None
            if cat is not None and first is not None:
                cat.setExpanded(True)
                self._models_tree.setCurrentItem(first)
                return

    def select_model(self, name: str) -> bool:
        """Select the catalogue model called *name*; False when there is none."""
        wanted = name.strip().lower()
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            if cat is None:
                continue
            for j in range(cat.childCount()):
                child = cat.child(j)
                if child is None:
                    continue
                payload = child.data(0, Qt.ItemDataRole.UserRole)
                label = payload.get("name", child.text(0)) if isinstance(payload, dict) else child.text(0)
                if str(label).strip().lower() == wanted:
                    cat.setExpanded(True)
                    self._models_tree.setCurrentItem(child)
                    return True
        return False

    def _on_model_tree_selection(self) -> None:
        item = self._models_tree.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            self._apply_model_choice_from_payload(payload)

    def _hide_model_options(self) -> None:
        for w in (self._lbl_degree, self._spin_degree, self._lbl_knots, self._spin_knots, self._lbl_spacing, self._combo_knot_spacing):
            w.hide()

    def _current_table(self) -> str:
        """Return a stable source table label for results/reporting."""
        return self._source_name or "selected_series"

    def _is_2d_fit(self) -> bool:
        """True when the selected model is a surface function z = f(x, y).

        Driven entirely by the selected model's declared ``ndim`` (see
        ``FitFunctionSpec``/``SurfaceFunctionScanner``), not by anything
        about the selected series: a 2-variable model always needs a second
        independent variable, and a 1-variable one never uses one, whatever
        roles the series happens to have.
        """
        return int(self._selected_model.get("ndim", 1)) == 2

    def _selected_column_names(self) -> tuple[str, str | None, str]:
        """Return source X, optional X2, and target column names."""
        if self._is_2d_fit():
            return (
                self._source_x_col or "x",
                self._source_x2_col or "y",
                self._source_y_col or "z",
            )
        return self._source_x_col or "x", None, self._source_y_col or "target"

    def _apply_model_choice_from_payload(self, payload: dict[str, Any]) -> None:
        """Apply one scanned function class to the dialog.

        All selectable models now come from ``FunctionScanner``.  There are no
        expression, rational, orthopoly, spline, or JSON catalogue branches here.
        Function metadata is the single source of truth.
        """
        self._hide_model_options()
        self._selected_model = dict(payload)
        self._param_names = [str(name) for name in payload.get("params", [])]
        self._refresh_param_legend()

        p0 = [float(v) for v in payload.get("p0", [1.0, 1.0])]
        self._param_defaults = p0
        self._spin_params.setValue(len(p0))
        self._ensure_params_rows(len(p0))
        self._populate_params_defaults()
        # Then improve on them wherever the function can: see
        # _fill_initial_from_estimator.
        self._fill_initial_from_estimator()

        expression_html = str(payload.get("expression", "")).strip()
        description = _(str(payload.get("description", "")).strip())
        if description:
            description_html = f'<div style="color:#666; margin-top:6px;">{html_escape(description)}</div>'
            expression_html = f"{expression_html}{description_html}" if expression_html else description_html

        if expression_html:
            self._function_expression_html.setText(expression_html)
            self._function_expression_html.show()
        else:
            self._function_expression_html.clear()
            self._function_expression_html.hide()

        self._refresh_default_output_name()

    def _on_degree_changed(self, val: int) -> None:
        del val

    def _on_knots_changed(self, val: int) -> None:
        del val

    def _param_label(self, row: int) -> str:
        """Return the human-readable parameter name for the table/report."""
        global_label = self._global_label(row)
        if global_label is not None:
            return global_label
        name = self._param_names[row] if row < len(self._param_names) else ""
        return name if name else f"p[{row}]"

    def _refresh_param_legend(self) -> None:
        """Show the parameter meanings under the expression."""
        if not self._param_names:
            self._param_legend.clear()
            self._param_legend.hide()
            return

        self._param_legend.setText("   ".join(self._param_names))
        self._param_legend.show()

    def _ensure_params_rows(self, count: int) -> None:
        """Grow or shrink the parameter table, keeping the values already typed."""
        self._params_table.setRowCount(int(count))
        for row in range(int(count)):
            # Column 0 is rewritten every time rather than only when missing:
            # it carries the parameter's meaning, which changes with the model.
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            for col, text in ((1, "0.0"), (2, "-inf"), (3, "inf")):
                if self._params_table.item(row, col) is None:
                    self._params_table.setItem(row, col, QTableWidgetItem(text))
            if self._params_table.cellWidget(row, 4) is None:
                self._params_table.setCellWidget(row, 4, QCheckBox(self._params_table))

    def _populate_params_defaults(self) -> None:
        # The model's own rows first; a global fit then lays them out per series.
        self._table_layout = None
        self._shared = []
        self._ensure_params_rows(len(self._param_defaults))
        for row, value in enumerate(self._param_defaults):
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._params_table.setItem(row, 1, QTableWidgetItem(str(float(value))))
            self._params_table.setItem(row, 2, QTableWidgetItem("-inf"))
            self._params_table.setItem(row, 3, QTableWidgetItem("inf"))
            cb = self._get_fix_checkbox(row)
            cb.setChecked(False)
        self._relayout_params_table()

    def _get_fix_checkbox(self, row: int) -> QCheckBox:
        widget = self._params_table.cellWidget(row, 4)
        if not isinstance(widget, QCheckBox):
            widget = QCheckBox(self._params_table)
            self._params_table.setCellWidget(row, 4, widget)
        return widget

    def on_multi_peak(self) -> None:
        """Create a configurable multi-peak model from the inline Model frame."""
        family = self._multi_family_combo.currentText()
        count = int(self._multi_count_spin.value())
        tie_width = bool(self._multi_tie_width.isChecked())
        tie_eta = bool(self._multi_tie_eta.isChecked())

        params: list[str] = []
        p0: list[float] = []
        for i in range(count):
            params.extend([f"amp {i + 1}", f"center {i + 1}"])
            p0.extend([1.0, float(i)])
            if not tie_width:
                params.append(f"width {i + 1}")
                p0.append(1.0)
            if family == "Pseudo-Voigt" and not tie_eta:
                params.append(f"eta {i + 1}")
                p0.append(0.5)
        if tie_width:
            params.append("shared width")
            p0.append(1.0)
        if family == "Pseudo-Voigt" and tie_eta:
            params.append("shared eta")
            p0.append(0.5)
        params.append("offset")
        p0.append(0.0)

        formula = {
            "Gaussian": "Σ Aᵢ exp(-(x-cᵢ)²/(2wᵢ²)) + C",
            "Lorentzian": "Σ Aᵢ (0.5wᵢ)² / ((x-cᵢ)² + (0.5wᵢ)²) + C",
            "Pseudo-Voigt": "Σ Aᵢ [ηᵢ Lᵢ(x) + (1-ηᵢ) Gᵢ(x)] + C",
        }.get(family, "multi-peak")

        self._selected_model = {
            "name": f"Multi {family} ({count})",
            "_multi_peak": True,
            "family": family,
            "count": count,
            "tie_width": tie_width,
            "tie_eta": tie_eta,
            "params": params,
            "p0": p0,
            "expression": f"<b>Multi {family}</b><br>{formula}",
            "description": "Configurable multi-peak model built from the inline controls.",
        }
        self._param_names = params
        self._refresh_param_legend()
        self._param_defaults = p0
        self._spin_params.setValue(len(p0))
        self._ensure_params_rows(len(p0))
        self._populate_params_defaults()
        self._function_expression_html.setText(str(self._selected_model["expression"]))
        self._function_expression_html.show()
        self._refresh_default_output_name(force=True)

    def _load_fit_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if self._is_2d_fit():
            return self._load_fit_data_2d()
        if self._global_rows():
            return self._load_global_fit_data()
        return self._load_fit_data_1d()

    def _load_fit_data_2d(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Read a series' x, y and z roles for a surface (2D) fit.

        A 2D fit's independent variables are not chosen through a combo: the
        series already names them through its own roles - x and y are the
        two independent variables, z is the target - which is what
        ``DataOperationDialogBase.series_xyz`` reads. No extra "X2" picker
        is needed for the common case this dialog supports (a series that
        already carries x/y/z), which is why none was added.
        """
        row = self._selected_series_row()
        roles = parse_roles(row_value(row, "roles", default={}))
        # A "name" role names the source; otherwise the series' own name.
        source_name = str(roles.get("name") or row_value(row, "name", "series_name", default="Series"))
        if not roles.get("z"):
            applogger.error(
                "A surface model needs a series with x, y and z roles; "
                "the selected series has no z role."
            )

        x_values, y_values, z_values = self.series_xyz(row, source_name)

        finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
        dropped = int(finite.size - int(np.count_nonzero(finite)))
        if dropped:
            applogger.warning(
                "%s: dropped %d row(s) with non-finite x/y/z before fitting.",
                source_name,
                dropped,
            )
        x_clean = x_values[finite]
        y_clean = y_values[finite]
        target_clean = z_values[finite]
        if target_clean.size < self.INPUT_MINIMUM_POINTS:
            applogger.error(
                f"{source_name}: not enough finite (x, y, z) points to fit a surface."
            )

        clean_frame = cast(
            pd.DataFrame,
            pd.DataFrame({"x": x_clean, "y": y_clean, "target": target_clean}),
        )

        self._source_name = source_name
        self._source_x_col = str(roles.get("x") or "x")
        self._source_x2_col = str(roles.get("y") or "y")
        self._source_y_col = str(roles.get("z") or "z")
        self._refresh_default_output_name(force=True)

        x_data = np.column_stack([x_clean, y_clean])
        target_data = target_clean
        return x_data, target_data, clean_frame

    def _load_fit_data_1d(self, row: Any = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        row = self._selected_series_row() if row is None else row
        source_name = str(
            parse_roles(row["roles"]).get("name") or row_value(row, "name", "series_name", default="Series")
        )
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            applogger.error("Selected series has no SQL query.")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            applogger.error("Selected series query returned no rows.")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = resolve_role_column(columns, roles, "x") or ""
        y_col = resolve_role_column(columns, roles, "y") or ""
        if x_col not in columns:
            x_col = numeric[0] if numeric else ""
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else ""

        if not x_col or not y_col:
            applogger.error("Selected series query must expose at least two numeric columns.")
        if x_col == y_col:
            applogger.error("Selected series X and Y columns must be different.")

        x_raw = self.numeric_x(frame[x_col], source_name)
        y_raw = self.numeric_y(frame[y_col])
        sigma = None
        if self._weights_key() == "errors":
            sigma = self._error_column(frame, columns, roles, source_name)
            x_raw, y_raw, sigma = _merge_points_with_errors(x_raw, y_raw, sigma, source_name)

        # Same three repairs as before - drop non-finite, sort by x, average
        # repeated x - but reported rather than silent. With errors they were
        # made above, carrying each point's error along.
        x_prepared, target_prepared = self.prepare_input_xy(x_raw, y_raw, label=source_name)

        clean_frame = cast(
            pd.DataFrame,
            pd.DataFrame({"x": x_prepared, "target": target_prepared}),
        )
        if sigma is not None:
            clean_frame["sigma"] = sigma

        self._source_name = source_name
        self._source_x_col = x_col
        self._source_y_col = y_col
        self._refresh_default_output_name(force=True)

        x_data = clean_frame["x"].to_numpy(dtype=float)
        target_data = clean_frame["target"].to_numpy(dtype=float)
        return x_data, target_data, clean_frame

    def _error_column(self, frame: pd.DataFrame, columns: list[str], roles: dict, source_name: str) -> np.ndarray:
        """Each point's y error: the yerr role, or the mean of yerr_low/yerr_high."""
        found = [resolve_role_column(columns, roles, role) for role in ("yerr", "yerr_low", "yerr_high")]
        if found[0] in columns:
            return self.numeric_y(frame[found[0]])
        halves = [self.numeric_y(frame[name]) for name in found[1:] if name in columns]
        if not halves:
            applogger.error(
                f"{source_name}: weighting by error bars needs a yerr column in the series' roles.",
                raise_error=True,
            )
        return np.mean(halves, axis=0)

    def _model_name(self) -> str:
        item = self._models_tree.currentItem()
        if item is not None and item.parent() is not None:
            return item.text(0)
        return str(self._selected_model.get("name", "Custom"))

    @staticmethod
    def _make_multi_peak_model(payload: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a callable for the inline multi-peak model."""
        return fit_engine.multi_peak_model(
            str(payload.get("family", "Gaussian")),
            int(payload.get("count", 1)),
            tie_width=bool(payload.get("tie_width", False)),
            tie_eta=bool(payload.get("tie_eta", False)),
        )

    def _build_model(self, x_data: np.ndarray, target_data: np.ndarray) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], np.ndarray] | None:
        """Build the selected scanned function model."""
        del x_data, target_data  # not needed for class-backed function models
        p0, _unused, _unused, _unused = self._collect_params_from_table()
        if self._selected_model.get("_multi_peak"):
            return self._make_multi_peak_model(self._selected_model), p0
        if self._selected_model.get("_sympy_stored"):
            from app.dialogs.function_trust_dialog import allowed_to_run

            # The project's code: run only once trusted on this computer (todo ID47).
            if not allowed_to_run(self, self._repo, self._selected_model):
                return None
            try:
                return make_stored_sympy_model(self._selected_model, functions_by_name(self._repo)), p0
            except Exception:
                applogger.exception(
                    "Failed to build stored SymPy function: %s",
                    self._selected_model.get("name", ""),
                )
                return None
        scanner = self._surface_scanner if self._is_2d_fit() else self._function_scanner
        try:
            return scanner.make_model(self._selected_model), p0
        except Exception:
            applogger.exception(
                "Failed to build scanned fit function: %s",
                self._selected_model.get("name", ""),
            )
            return None

    def _float_item(self, row: int, col: int, default: float) -> float:
        item = self._params_table.item(row, col)
        text = item.text().strip() if item is not None else ""
        if not text:
            return default
        low = text.lower()
        if low in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if low in ("-inf", "-infinity"):
            return -math.inf
        return float(text)

    def _collect_params_from_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self._params_table.rowCount()
        p0 = np.zeros(n, dtype=float)
        lb = np.full(n, -np.inf, dtype=float)
        ub = np.full(n, np.inf, dtype=float)
        fixed = np.zeros(n, dtype=bool)
        for row in range(n):
            p0[row] = self._float_item(row, 1, 0.0)
            lb[row] = self._float_item(row, 2, -np.inf)
            ub[row] = self._float_item(row, 3, np.inf)
            fixed[row] = self._get_fix_checkbox(row).isChecked()
            if lb[row] > ub[row]:
                applogger.error(f"Lower bound is greater than upper bound at p[{row}].")
            p0[row] = min(max(p0[row], lb[row]), ub[row])
        return p0, lb, ub, fixed

    def _set_initial_params(self, values: np.ndarray) -> None:
        for row, value in enumerate(self._expand_to_table(values)):
            if row < self._params_table.rowCount():
                self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))

    # ==================================================================
    # StartingPoint: Algorithm and starting point: optimiser, loss, initial values; the results pane and output table.
    # ==================================================================
    # ------------------------------------------------------------------
    # Algorithm and starting point
    # ------------------------------------------------------------------
    def _optimizer_key(self) -> str:
        """Return the chosen algorithm's key."""
        return str(self._combo_optimizer.currentData() or DEFAULT_OPTIMIZER)

    def _loss_key(self) -> str:
        """Return the chosen robust loss, or plain least squares."""
        return str(self._combo_loss.currentData() or "linear")

    def _refresh_optimizer_help(self) -> None:
        """Explain the chosen algorithm, and disable the loss where it does nothing."""
        optimizer = BY_KEY.get(self._optimizer_key())
        if optimizer is None:
            return
        self._combo_optimizer.setToolTip(_(optimizer.description))
        # Only the least_squares family takes a loss. Leaving the control
        # enabled where it is ignored is a setting that silently does nothing.
        self._combo_loss.setEnabled(optimizer.supports_loss)

    def _selected_function_class(self) -> Any:
        """Return the class behind the selected model, or None.

        None for the multi-peak builder, which is assembled here rather than
        discovered and so has no class to ask.
        """
        if self._selected_model and self._selected_model.get("_sympy_stored"):
            # A project's own function: the generic estimator reads its starting values off the data.
            return (_StoredFunctionGuess(self._selected_model, functions_by_name(self._repo))
                    if not self._is_2d_fit() else None)
        if not self._selected_model or self._selected_model.get("_multi_peak"):
            return None
        scanner = self._surface_scanner if self._is_2d_fit() else self._function_scanner
        try:
            cls = scanner.load_class(self._selected_model)
        except Exception:
            applogger.exception(
                "Could not load the class for %s", self._selected_model.get("name", "")
            )
            return None
        # ask_the_function/choose_starting_point call initial_guess(x, y) -
        # the 1D contract every fit function shares. A surface function's own
        # initial_guess(x, y, z) needs one more argument, so it is wrapped
        # rather than called directly; see _SurfaceInitialGuessAdapter.
        return _SurfaceInitialGuessAdapter(cls) if self._is_2d_fit() else cls

    def _fill_initial_from_estimator(self) -> bool:
        """Put the function's own estimate in the Initial column, if it has one.

        Run when a model is selected, because it costs one pass over the data
        and a starting point read off the data beats a declared default every
        time: a Gaussian's centre is where the mass is, not 0.0.

        Only the function's own estimator, never the search - selecting a model
        is a click, and a click must not cost a twenty-thousand-sample scan.
        That is what Estimate is for.
        """
        function_class = self._selected_function_class()
        if function_class is None:
            return False
        # Checked before asking for the data, not caught after: with nothing
        # selected _selected_series_row shows an error box, and a model click
        # - or simply opening the dialog on an empty figure - must not put a
        # box in front of someone who has not asked for anything yet.
        if not self.selected_series():
            return False
        if getattr(self, "_table_layout", None) is not None:
            return self._fill_global_initial_from_estimator(function_class)
        try:
            x_data, target_data, _clean = self._load_fit_data()
        except Exception:
            # A query that does not run, or a series with no numeric columns.
            # The declared defaults stay, and nothing is reported.
            return False
        if x_data is None or target_data is None:
            return False

        p0, lb, ub, _fixed = self._collect_params_from_table()
        estimate = ask_the_function(function_class, x_data, target_data, expected=p0.size)
        if estimate is None:
            return False

        self._set_initial_params(clip_into_bounds(estimate, lb, ub))
        return True

    def on_estimate_initial_values(self) -> None:
        """Fill the Initial column from the data, searching if it has to.

        The whole policy, unlike the automatic pass above: the function's own
        estimator when it has one, and a Monte Carlo search of the parameter
        space when it does not - which is most of the library, and every user
        function this application has never seen.
        """
        if getattr(self, "_table_layout", None) is not None:
            self._estimate_global_initial_values()
            return
        try:
            x_data, target_data, _clean = self._load_fit_data()
            if x_data is None or target_data is None:
                return
            built = self._build_model(x_data, target_data)
            if built is None:
                return
            model, _model_p0 = built

            p0, lb, ub, fixed = self._collect_params_from_table()
            start = choose_starting_point(
                model,
                x_data,
                target_data,
                declared=p0,
                lower=lb,
                upper=ub,
                function_class=self._selected_function_class(),
                fixed=fixed,
            )
        except Exception as exc:
            applogger.exception("Could not estimate initial values")
            show_message(
                self, "series.estimate_failed", series=self._current_table(), error=exc
            )
            return

        self._set_initial_params(start.values)
        applogger.info("Initial values from %s (cost %.6g)", start.source, start.cost)
        self.show_results(self._estimate_report(start.source, start.cost))

    def _fill_global_initial_from_estimator(self, function_class: Any) -> bool:
        """The function's own estimate, series by series (a global fit)."""
        try:
            series = self._per_series_xy()
        except Exception:
            return False
        n_params = len(self._param_defaults)
        estimates = [ask_the_function(function_class, x, y, expected=n_params) for x, y in series]
        theta = self._combine_estimates([None if value is None else np.asarray(value, float) for value in estimates])
        if theta is None:
            return False
        _p0, lb, ub, _fixed = self._collect_params_from_table()
        self._set_initial_params(clip_into_bounds(theta, lb, ub))
        return True

    def _estimate_global_initial_values(self) -> None:
        """Estimate, series by series, from each series' own rows; shared ones averaged."""
        layout = self._table_layout
        if layout is None:
            return
        try:
            series = self._per_series_xy()
            model, _model_p0 = self._build_model(series[0][0], series[0][1]) or (None, None)
            if model is None:
                return
            p0, lb, ub, fixed = self._collect_params_from_table()
            starts = [
                choose_starting_point(
                    model, x, y,
                    declared=p0[layout.index[number]],
                    lower=lb[layout.index[number]],
                    upper=ub[layout.index[number]],
                    function_class=self._selected_function_class(),
                    fixed=fixed[layout.index[number]],
                )
                for number, (x, y) in enumerate(series)
            ]
        except Exception as exc:
            applogger.exception("Could not estimate initial values")
            show_message(self, "series.estimate_failed", series=self._current_table(), error=exc)
            return
        theta = self._combine_estimates([np.asarray(start.values, float) for start in starts])
        if theta is None:
            return
        self._set_initial_params(clip_into_bounds(theta, lb, ub))
        cost = float(sum(start.cost for start in starts))
        applogger.info("Initial values, series by series (total cost %.6g)", cost)
        self.show_results(self._estimate_report(starts[0].source, cost))

    @staticmethod
    def _estimate_report(source: str, cost: float) -> str:
        """One line saying where the starting values came from, and how good.

        Reported because "the fit is wrong" and "the starting values were
        invented" are different problems, and the user should not have to
        guess which one they have.
        """
        wording = {
            FROM_FUNCTION: _(
                "Initial values read off the data by the function's own estimator."
            ),
            FROM_SEARCH: _(
                "Initial values found by searching the parameter space at random: "
                "this function has no estimator of its own."
            ),
            FROM_DECLARED: _(
                "The search found nothing better than the declared values, which "
                "are unchanged."
            ),
        }.get(source, _("Initial values updated."))
        return wording + "\n" + _("Residual sum of squares: {cost}").format(
            cost=f"{cost:.6g}"
        )

    @staticmethod
    def _build_output_frame(job: _FitJob, fit_values: np.ndarray, residual: np.ndarray) -> pd.DataFrame:
        clean = job.clean
        data: dict[str, Any] = {
            "x": clean["x"].to_numpy(float),
            "target": clean["target"].to_numpy(float),
            "fit": np.asarray(fit_values, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "source_table": job.source_table,
            "source_x_col": job.x_col,
            "source_target_col": job.target_col,
            "fit_mode": "2D" if job.is_2d else "1D",
            "model": job.model_name,
        }
        if job.is_2d:
            data["y"] = clean["y"].to_numpy(float)
            data["z"] = data["target"]
            data["z_fit"] = data["fit"]
            data["source_y_col"] = job.x2_col or ""
        else:
            data["y"] = data["target"]
            data["y_fit"] = data["fit"]
            data["source_y_col"] = job.target_col
        return pd.DataFrame(data)

    def _fill_results_table(self, params: np.ndarray, std: np.ndarray) -> None:
        """Fill the parameter grid in the left panel."""
        self._results_table.setRowCount(len(params))
        for row, value in enumerate(params):
            self._results_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._results_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))
            self._results_table.setItem(
                row,
                2,
                QTableWidgetItem("" if not np.isfinite(std[row]) else f"{float(std[row]):.6g}"),
            )
        self._results_table.resizeColumnsToContents()

    def _results_html(self, result: SeriesFitResult) -> str:
        """Return the fit report of one result in the shared house style."""
        return self.results_document(result.model_name, *self._report_sections(result))

    def _report_sections(self, result: SeriesFitResult) -> list[str]:
        """The sections of one result's report, without the document around them.

        Several results share one document: a page holds one ``<html>``
        and one ``<body>``, and a report per result pasted into the next
        nested them.
        """
        params = np.asarray(result.params, dtype=float)
        std = np.asarray(result.param_std, dtype=float)
        names = result.param_names or [self._param_label(row) for row in range(params.size)]

        def column(values: np.ndarray | None, row: int) -> float:
            if values is None or row >= np.asarray(values).size:
                return math.nan
            return float(np.asarray(values, dtype=float)[row])

        def number(value: float) -> str:
            return report_html.format_number(value) if np.isfinite(value) else ""

        parameter_rows = [
            (
                html_escape(names[row] if row < len(names) else f"p[{row}]"),
                report_html.format_number(value, digits=8),
                self._std_text(std, row),
                number(column(result.param_t, row)),
                report_html.format_p_value(column(result.param_p, row))
                if np.isfinite(column(result.param_p, row))
                else "",
                number(column(result.param_ci_low, row)),
                number(column(result.param_ci_high, row)),
            )
            for row, value in enumerate(params)
        ]

        metric_labels = {
            "r2": "R&sup2;",
            "rmse": "RMSE",
            "ss_res": _("SS residual"),
            "aic": "AIC",
            "bic": "BIC",
            "reduced_chi2": _("Residual variance (reduced χ²)"),
        }
        metric_rows = [
            (metric_labels[key], report_html.format_number(result.metrics[key]))
            for key in ("r2", "rmse", "ss_res", "reduced_chi2", "aic", "bic")
            if key in result.metrics and np.isfinite(result.metrics[key])
        ]

        corr = np.asarray(result.param_corr, dtype=float)
        corr_rows: list[tuple[Any, ...]] = []
        if corr.size:
            for i, name in enumerate(names):
                row_values = [html_escape(name)]
                for j in range(len(names)):
                    value = corr[i, j] if i < corr.shape[0] and j < corr.shape[1] else math.nan
                    row_values.append("" if not np.isfinite(value) else f"{float(value):.4g}")
                corr_rows.append(tuple(row_values))

        summary_rows = [
            (_("Fit mode"), result.fit_mode),
            (_("Source"), result.source_table),
        ]
        if result.fit_mode == "2D":
            summary_rows.append(("X1", result.x_col or ""))
            summary_rows.append(("X2", result.x2_col or ""))
        else:
            summary_rows.append(("X", result.x_col or ""))
        summary_rows.append((_("Target"), result.target_col or ""))
        summary_rows.append((_("Status"), result.message))

        return [
            report_html.section(
                _("Curve"),
                report_html.summary_table(summary_rows),
            ),
            report_html.section(
                _("Function expression"),
                result.expression or "",
            ),
            report_html.section(
                _("Function expression with evaluated parameters"),
                result.evaluated_expression or "",
            ),
            report_html.section(
                _("Parameter estimates"),
                report_html.table(
                    [_("Parameter"), _("Estimate"), _("Std. error"), "t", "p", _("95% CI low"), _("95% CI high")],
                    parameter_rows,
                )
                + report_html.note(_(
                    "t and p test whether each parameter is zero, with "
                    "{dof} residual degrees of freedom. Fixed "
                    "parameters have no error, so none is shown."
                ).format(dof=result.dof)),
            ),
            report_html.section(
                _("Correlation matrix"),
                report_html.table(
                    [_("Parameter"), *[html_escape(name) for name in names]],
                    corr_rows,
                    empty_message=_("No correlation matrix available."),
                ),
            ),
            report_html.section(
                _("Goodness of fit"),
                report_html.table(
                    [_("Measure"), _("Value")],
                    metric_rows,
                    empty_message=_("No metrics available."),
                ),
            ),
        ]

    @staticmethod
    def _std_text(std: np.ndarray, row: int) -> str:
        """Return the standard error of one parameter, blank when unknown.

        A fixed parameter, or one the optimiser could not resolve, has no
        meaningful error; printing ``nan`` there would look like a failure
        rather than like an absence.
        """
        if row >= std.size or not np.isfinite(std[row]):
            return ""
        return f"{float(std[row]):.6g}"

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        labels = {"r2": "R^2", "rmse": "RMSE", "ss_res": "SS_RES", "aic": "AIC", "bic": "BIC",
                  "chi2": "Chi^2", "chi2_reduced": "Chi^2/dof"}
        parts = []
        for key in labels:
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                parts.append(f"{labels[key]}={value:.6g}")
        return ", ".join(parts)

