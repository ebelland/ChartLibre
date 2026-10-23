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

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHeaderView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)
from app.plugin_api.functions.optimizers import DEFAULT_OPTIMIZER, LOSSES, OPTIMIZERS

from app.data.sqlite_repo import SqliteRepo
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase

from app.styles.style import CardFrame, TitledCard, create_action_button, mark_editor_panel, stdSizeAndlayout
from app.utils.i18n import _
from app.scanners.functions_scanner import FunctionScanner, SurfaceFunctionScanner

from app.data_operations.fit_dialog.common import SeriesFitResult
from app.data_operations.fit_dialog.model_data import ModelDataMixin
from app.data_operations.fit_dialog.global_fit import GlobalFitMixin
from app.data_operations.fit_dialog.running import FitRunMixin
from app.data_operations.fit_dialog.starting_point import StartingPointMixin
from app.data_operations.fit_dialog.hooks import FitHooksMixin
from app.data_operations.fit_dialog.accessory import AccessoryChartsMixin


class SeriesFitDialog(GlobalFitMixin, ModelDataMixin, FitRunMixin, StartingPointMixin, FitHooksMixin, AccessoryChartsMixin, DataOperationDialogBase):
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
    Name: str = "Fit Curve"
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
    Goals: list[str] = ["relate", "predict"]
    Category: str = "Predictive and Specialized Modeling"
    Order: int = 34
    saved = Signal(str)

    # format_results returns a table.
    RESULTS_ARE_HTML: bool = True

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        applied_callback: Callable[[], None] | None = None,
        table: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            parent=parent,
            width=900,
            height=560
        )
        self.setModal(True)
        self._applied_callback = applied_callback
        self._initial_table = table
        self._refresh_default_output_name(force=True)
        self.model_combo.setVisible(False)

    def init_operation_widgets(self) -> None:
        """Create fit controls before base builder hooks run.

        The base dialog invokes build_model_selector() and
        build_parameter_selector() during super().__init__().  Therefore every
        widget used by those builders must be created here.
        """
        self._applied_callback: Callable[[], None] | None = None
        self._initial_table: str | None = None
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
        self._output_table_edit = QLineEdit(self)
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

        # Optional accessory charts: judging a fit needs more than the curve
        # overlaid on the data, and neither picture is one this dialog draws
        # by default - they cost a new axis each, so only appear when asked.
        self._residual_chart_check = QCheckBox(_("Residuals vs. x"), self)
        self._residual_chart_check.setToolTip(
            _(
                "Add a chart of the fit residuals (measured minus fit) on a "
                "new axis in this figure. The classic way to see a fit that "
                "looks good but has structure in what it left behind."
            )
        )
        self._fit_vs_measured_chart_check = QCheckBox(_("Measured vs. fit"), self)
        self._fit_vs_measured_chart_check.setToolTip(
            _(
                "Add a chart of the measured values against the fitted "
                "values on a new axis in this figure. A perfect fit falls "
                "on the diagonal; how far points stray from it is the fit "
                "quality, independent of what x means."
            )
        )
        self._confidence_band_check = QCheckBox(_("95% confidence band"), self)
        self._confidence_band_check.setToolTip(
            _(
                "Shade where the true curve lies with 95% confidence, given "
                "the parameters' uncertainty. Narrow where the data pin the "
                "curve down, wide where they do not."
            )
        )
        self._prediction_band_check = QCheckBox(_("95% prediction band"), self)
        self._prediction_band_check.setToolTip(
            _(
                "Shade where a new measurement would fall with 95% "
                "probability: the confidence band plus the scatter of the "
                "points around the curve."
            )
        )
        # The fitted curve drawn smooth and, if asked, past the data (1D fits).
        self._curve_points_spin = QSpinBox(self)
        self._curve_points_spin.setRange(0, 100_000)
        self._curve_points_spin.setSingleStep(100)
        self._curve_points_spin.setValue(200)
        self._curve_points_spin.setSpecialValueText(_("At the data"))
        self._curve_points_spin.setToolTip(_("How many points the fitted curve is drawn with, evenly spaced; "
                                             "0 draws it only at the data's x."))
        self._curve_extend_spin = QDoubleSpinBox(self)
        self._curve_extend_spin.setRange(0.0, 1000.0)
        self._curve_extend_spin.setDecimals(0)
        self._curve_extend_spin.setSingleStep(10.0)
        self._curve_extend_spin.setSuffix(" %")
        self._curve_extend_spin.setToolTip(_("Draw the curve past the data, on each side, by this share of "
                                             "the data's x range: 10 % extrapolates a tenth either way."))
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

    def extra_tabs(self) -> list[tuple[QWidget, str]]:
        """What the fit writes and draws, in a tab of its own: the Parameters tab is tall enough."""
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        output = TitledCard(page, _("Output table"))
        self._output_table_edit.setToolTip(_("Name of the SQLite table where fitted values/residuals are saved."))
        output.card.layout().addWidget(self._output_table_edit)
        layout.addWidget(output)
        curve = TitledCard(page, _("Fitted curve"))
        curve_form = QFormLayout()
        curve_form.addRow(_("Points:"), self._curve_points_spin)
        curve_form.addRow(_("Extend:"), self._curve_extend_spin)
        curve.card.layout().addLayout(curve_form)
        layout.addWidget(curve)
        bands = TitledCard(page, _("Bands on the fit"))
        bands.card.layout().addWidget(self._confidence_band_check)
        bands.card.layout().addWidget(self._prediction_band_check)
        layout.addWidget(bands)
        charts = TitledCard(page, _("Extra charts"), collapsible=True, collapsed=True)
        charts.card.layout().addWidget(self._residual_chart_check)
        charts.card.layout().addWidget(self._fit_vs_measured_chart_check)
        layout.addWidget(charts)
        layout.addStretch(1)
        return [(page, _("Output"))]
