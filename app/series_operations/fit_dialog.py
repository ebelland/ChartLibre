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

from html import escape as html_escape
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast

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
    QLabel,
    QMenu,
    QLineEdit,
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
from app.functions.optimizers import (
    DEFAULT_OPTIMIZER,
    LOSSES,
    OPTIMIZERS,
    BY_KEY,
)
from app.functions.starting_point import (
    FROM_DECLARED,
    FROM_FUNCTION,
    FROM_SEARCH,
    ask_the_function,
    choose_starting_point,
    clip_into_bounds,
)

from app.data.data_source import row_value , parse_roles
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.utils import report_html

from app.styles.style import (
    CardFrame,
    create_action_button,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.scanners.functions_scanner import FunctionScanner, SurfaceFunctionScanner


def _split_xy(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        applogger.error("This model requires 2D input with x and y independent columns.")
    return arr[:, 0], arr[:, 1]


class _SurfaceInitialGuessAdapter:
    """Adapt a surface function's 3-argument ``initial_guess`` to the plain
    ``(x, y)`` shape ``ask_the_function``/``choose_starting_point`` expect.

    Those two (``app/functions/starting_point.py``) are shared with every 1D
    fit function, whose ``initial_guess(x, y)`` reads ``x`` as the sole
    independent variable and ``y`` as the target. A surface function's own
    ``initial_guess(x, y, z)`` needs x and y separately and z as the target -
    so this unpacks the ``(N, 2)`` array this dialog builds for a 2D fit
    before forwarding, rather than teaching the shared starting-point module
    about a minority of functions that take one more argument.
    """

    def __init__(self, cls: Any) -> None:
        self._cls = cls

    def initial_guess(self, xy: np.ndarray, z: np.ndarray) -> list[float] | None:
        arr = np.asarray(xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return self._cls.initial_guess(arr[:, 0], arr[:, 1], np.asarray(z, dtype=float))

@dataclass(slots=True)
class SeriesFitResult(TableResult):
    source_table: str
    x_col: str
    target_col: str
    x2_col: str | None
    fit_mode: str
    model: str
    params: np.ndarray
    param_std: np.ndarray
    param_corr: np.ndarray
    param_names: list[str]
    expression: str
    evaluated_expression: str
    metrics: dict[str, float]
    output_table: str
    frame: pd.DataFrame
    message: str
    #: t value, p value (is the parameter zero?) and 95% interval per
    #: parameter, from Student's t with the residual degrees of freedom.
    param_t: np.ndarray | None = None
    param_p: np.ndarray | None = None
    param_ci_low: np.ndarray | None = None
    param_ci_high: np.ndarray | None = None
    dof: int = 0

    @property
    def parameters(self) -> dict[str, Any]:
        return {name: float(value) for name, value in zip(self.param_names, self.params)}

    @property
    def series(self) -> tuple[str, ...]:
        return (self.source_table,)

    def to_df(self) -> pd.DataFrame:
        return self.frame


@dataclass(slots=True)
class _FitJob:
    """Everything one fit needs, read off the window before it starts.

    A fit may run in the background while the window stays usable, so what
    was selected when Fit was pressed - model, columns, names - is captured
    here rather than read again when the result comes back.
    """

    model: Callable[[np.ndarray, np.ndarray], np.ndarray]
    x: np.ndarray
    target: np.ndarray
    clean: pd.DataFrame
    p0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fixed: np.ndarray
    optimizer: str
    loss: str
    max_nfev: int
    weighted: bool
    source_table: str
    x_col: str
    x2_col: str | None
    target_col: str
    is_2d: bool
    model_name: str
    param_names: list[str]
    expression: str
    output_table: str
    #: Fit (optimise the parameters) or merely evaluate them.
    optimise: bool = False

    def run(self, should_stop: Callable[[], bool] | None = None) -> fit_engine.CurveFit:
        """The calculation alone - no Qt, safe on a worker thread."""
        return fit_engine.fit_curve(
            self.model,
            self.x,
            self.target,
            self.p0,
            self.lower,
            self.upper,
            self.fixed,
            optimise=self.optimise,
            optimizer=self.optimizer,
            loss=self.loss,
            max_nfev=self.max_nfev,
            weighted=self.weighted,
            should_stop=should_stop,
        )


class SeriesFitDialog(SeriesOperationDialogBase):
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
    Name: str  = "Fit"
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
            title="Series Fit",
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
        self._weighted_check = QCheckBox(_("Weighted RMSE/R²"), self)
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
        self._residual_axis_id: int | None = None
        self._fit_vs_measured_axis_id: int | None = None
        self._accessory_table_name: str | None = None
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
        self._params_table.setHorizontalHeaderLabels(["Parameter", "Initial", "Lower", "Upper", "Fix"])
        self._params_table.horizontalHeader().setStretchLastSection(True)
        self._params_table.setMinimumHeight(160)
        self._params_table.setToolTip(
            _("Initial value, bounds and fixed flag for each function parameter. Apply runs the fit with these settings.")
        )
        layout.addWidget(self._params_table, 1)

        # Output / fit options, one control per row: a translated algorithm
        # label ("Regione di fiducia (predefinito)") and a translated loss
        # label ("Arcotangente (molto robusta)") side by side do not fit this
        # panel's width at all, let alone alongside their own labels - the
        # single row this used to be overflowed it horizontally, and two
        # combos paired per row still did. A QFormLayout gives each combo the
        # whole row to grow into, which is what a translated label this long
        # actually needs.
        fit_options = QWidget(outer)
        fit_options_layout = QFormLayout(fit_options)
        stdSizeAndlayout(fit_options_layout)

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
        self._weighted_check.setToolTip(_("Weight residuals by magnitude when computing RMSE/R²."))
        run_row_layout.addWidget(self._weighted_check)
        run_row_layout.addStretch(1)
        fit_options_layout.addRow(_("Max evals:"), run_row)

        self._btn_estimate.setToolTip(
            _(
                "Fill the Initial column from the data: the function's own "
                "estimator when it has one, a Monte Carlo search of the "
                "parameter space when it does not."
            )
        )
        fit_options_layout.addRow("", self._btn_estimate)

        layout.addWidget(fit_options)
        self._refresh_optimizer_help()

        layout.addWidget(QLabel(_("Output table:"), outer))
        self._output_table_edit.setToolTip(_("Name of the SQLite table where fitted values/residuals are saved."))
        layout.addWidget(self._output_table_edit)

        layout.addWidget(QLabel(_("Bands on the fit:"), outer))
        layout.addWidget(self._confidence_band_check)
        layout.addWidget(self._prediction_band_check)

        layout.addWidget(QLabel(_("Extra charts:"), outer))
        layout.addWidget(self._residual_chart_check)
        layout.addWidget(self._fit_vs_measured_chart_check)

        return outer

    def _catalog_data(self) -> dict[str, list[dict[str, Any]]]:
        """Return scanned fit functions grouped by category.

        The previous JSON catalog is intentionally removed.  All fit models are
        now function classes discovered from app/functions/functions.py and
        app/functions/user_functions.py through FunctionScanner, plus every
        z = f(x, y) surface function from app/functions/surface_functions.py
        and app/functions/user_surface_functions.py through
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
                match = needle in child.text(0).lower()
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
            if cat is not None and cat.childCount():
                cat.setExpanded(True)
                self._models_tree.setCurrentItem(cat.child(0))
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
        return self._initial_table or self._source_name or "selected_series"

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
        self._params_table.resizeColumnsToContents()

    def _populate_params_defaults(self) -> None:
        self._ensure_params_rows(len(self._param_defaults))
        for row, value in enumerate(self._param_defaults):
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._params_table.setItem(row, 1, QTableWidgetItem(str(float(value))))
            self._params_table.setItem(row, 2, QTableWidgetItem("-inf"))
            self._params_table.setItem(row, 3, QTableWidgetItem("inf"))
            cb = self._get_fix_checkbox(row)
            cb.setChecked(False)

    def _get_fix_checkbox(self, row: int) -> QCheckBox:
        widget = self._params_table.cellWidget(row, 4)
        if not isinstance(widget, QCheckBox):
            widget = QCheckBox(self._params_table)
            self._params_table.setCellWidget(row, 4, widget)
        return widget

    @staticmethod
    def _unmatched_bracket_positions(expr: str) -> list[tuple[int, str, str]]:
        """Return unmatched bracket positions while ignoring strings and comments.

        The returned tuples are (absolute_position, character, reason).
        Brackets covered: (), [], {}. The UI message says parentheses because
        that is the most common user-facing case, but highlighting covers all
        expression grouping delimiters.
        """
        opens = {"(": ")", "[": "]", "{": "}"}
        closes = {")": "(", "]": "[", "}": "{"}
        stack: list[tuple[str, int]] = []
        errors: list[tuple[int, str, str]] = []
        quote: str | None = None
        triple_quote: str | None = None
        escape = False
        index = 0
        n = len(expr)

        while index < n:
            ch = expr[index]
            nxt3 = expr[index : index + 3]

            if quote is not None:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif triple_quote is not None and nxt3 == triple_quote:
                    quote = None
                    triple_quote = None
                    index += 2
                elif triple_quote is None and ch == quote:
                    quote = None
                index += 1
                continue

            if ch == "#":
                newline = expr.find("\n", index)
                if newline < 0:
                    break
                index = newline + 1
                continue

            if nxt3 == "'''" or nxt3 == '"""':
                quote = nxt3[0]
                triple_quote = nxt3
                index += 3
                continue

            if ch in ("'", '"'):
                quote = ch
                triple_quote = None
                index += 1
                continue

            if ch in opens:
                stack.append((ch, index))
            elif ch in closes:
                if not stack:
                    errors.append((index, ch, f"unmatched closing '{ch}'"))
                else:
                    open_ch, open_pos = stack.pop()
                    if open_ch != closes[ch]:
                        errors.append((open_pos, open_ch, f"expected '{opens[open_ch]}' before '{ch}'"))
                        errors.append((index, ch, f"unmatched closing '{ch}'"))
            index += 1

        for open_ch, open_pos in stack:
            errors.append((open_pos, open_ch, f"unmatched opening '{open_ch}'"))
        return sorted(errors, key=lambda item: item[0])


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
        return self._load_fit_data_1d()

    def _load_fit_data_2d(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Read a series' x, y and z roles for a surface (2D) fit.

        A 2D fit's independent variables are not chosen through a combo: the
        series already names them through its own roles - x and y are the
        two independent variables, z is the target - which is what
        ``SeriesOperationDialogBase.series_xyz`` reads. No extra "X2" picker
        is needed for the common case this dialog supports (a series that
        already carries x/y/z), which is why none was added.
        """
        row = self._selected_series_row()
        roles = parse_roles(row_value(row, "roles", default={}))
        source_name = str(roles.get("name", "Series"))
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
                show_dialog=False,
                raise_error=False,
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

    def _load_fit_data_1d(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        row = self._selected_series_row()
        source_name = str(parse_roles(row["roles"]).get("name", "Series"))
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            applogger.error("Selected series has no SQL query.")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            applogger.error("Selected series query returned no rows.")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = str(roles.get("x", ""))
        y_col = str(roles.get("y", ""))
        if x_col not in columns:
            x_col = numeric[0] if numeric else ""
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else ""

        if not x_col or not y_col:
            applogger.error("Selected series query must expose at least two numeric columns.")
        if x_col == y_col:
            applogger.error("Selected series X and Y columns must be different.")

        # Same three repairs as before - drop non-finite, sort by x, average
        # repeated x - but reported rather than silent.
        x_prepared, target_prepared = self.prepare_input_xy(
            self.numeric_x(frame[x_col], source_name),
            self.numeric_y(frame[y_col]),
            label=source_name,
        )

        clean_frame = cast(
            pd.DataFrame,
            pd.DataFrame({"x": x_prepared, "target": target_prepared}),
        )

        self._source_name = source_name
        self._source_x_col = x_col
        self._source_y_col = y_col
        self._refresh_default_output_name(force=True)

        x_data = clean_frame["x"].to_numpy(dtype=float)
        target_data = clean_frame["target"].to_numpy(dtype=float)
        return x_data, target_data, clean_frame

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
        for row, value in enumerate(np.asarray(values, dtype=float)):
            if row < self._params_table.rowCount():
                self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))

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
        # optimum has to land in it rather than only in the report.
        self._set_initial_params(np.asarray(result.params, dtype=float))
        self._fill_results_table(result.params, result.param_std)
        self._btn_use_fit_params.setEnabled(True)
        self.publish_results(self.format_results([result]))

        # A fit nobody can see is only half an answer.
        self.preview()

    def busy_widgets(self) -> list[QWidget]:
        return [*super().busy_widgets(), self.fit_button, self._btn_estimate]

    def prepare_job(self, *, optimise: bool = False) -> _FitJob | None:
        """Read the data, the model and the parameter table into a job."""
        x_data, target_data, clean = self._load_fit_data()
        if x_data is None or target_data is None:
            return None
        built = self._build_model(x_data, target_data)
        if built is None:
            return None
        model, _model_p0 = built
        p0, lb, ub, fixed = self._collect_params_from_table()
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
            weighted=self._weighted_check.isChecked(),
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
        )

    def finish_job(self, job: _FitJob, outcome: fit_engine.CurveFit) -> list[SeriesFitResult]:
        """The engine's outcome as this dialog's result; kept as _last_result."""
        if job.optimise and outcome.success:
            # Remembered so that Preview/OK, which only evaluate the table,
            # can still say the parameters they draw are a converged fit.
            self._last_fit = (job.model_name, np.asarray(outcome.params, dtype=float), outcome.message)
        elif not job.optimise:
            outcome.message = self._evaluation_message(job, outcome)
        try:
            self._last_result = self._result_from(job, outcome, optimise=job.optimise)
        except Exception:
            self._last_result = None
            raise
        return [self._last_result]

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

    def build_extra_action_buttons(self, layout) -> None:
        """Add the Fit button next to Preview."""
        self.fit_button = create_action_button(
                              parent=self,
                              action_id="run_fit",
                              action=self.on_fit,
                              layout=layout,
                          )

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
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Could not estimate initial values")
            show_message(
                self, "series.estimate_failed", series=self._current_table(), error=exc
            )
            return

        self._set_initial_params(start.values)
        applogger.info("Initial values from %s (cost %.6g)", start.source, start.cost)
        self.set_results_text(self._estimate_report(start.source, start.cost))

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
        """Return the fit report in the shared house style."""
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
            "ss_res": "SS residual",
            "aic": "AIC",
            "bic": "BIC",
            "reduced_chi2": "Residual variance (reduced &chi;&sup2;)",
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
            ("Mode", result.fit_mode),
            ("Source", result.source_table),
        ]
        if result.fit_mode == "2D":
            summary_rows.append(("X1", result.x_col or ""))
            summary_rows.append(("X2", result.x2_col or ""))
        else:
            summary_rows.append(("X", result.x_col or ""))
        summary_rows.append(("Target", result.target_col or ""))
        summary_rows.append(("Status", result.message))

        return report_html.document(
            "Fit",
            result.model_name,
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
                    ["Parameter", "Estimate", "Std. error", "t", "p", "95% CI low", "95% CI high"],
                    parameter_rows,
                )
                + report_html.note(
                    f"t and p test whether each parameter is zero, with "
                    f"{result.dof} residual degrees of freedom. Fixed "
                    "parameters have no error, so none is shown."
                ),
            ),
            report_html.section(
                _("Correlation matrix"),
                report_html.table(
                    ["Parameter", *[html_escape(name) for name in names]],
                    corr_rows,
                    empty_message="No correlation matrix available.",
                ),
            ),
            report_html.section(
                _("Goodness of fit"),
                report_html.table(
                    ["Measure", "Value"],
                    metric_rows,
                    empty_message="No metrics available.",
                ),
            ),
        )

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
        labels = {"r2": "R^2", "rmse": "RMSE", "ss_res": "SS_RES", "aic": "AIC", "bic": "BIC"}
        parts = []
        for key in ("r2", "rmse", "ss_res", "aic", "bic"):
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                parts.append(f"{labels[key]}={value:.6g}")
        return ", ".join(parts)

    def _default_output_table_name(self) -> str:
        table = self._current_table() or "table"
        target = self._source_y_col or "target"
        model = self._model_name() or "fit"
        return generated_table_name(f"Fit_{table}_{target}_{model}", fallback="Fit_Result")

    def _refresh_default_output_name(self, force: bool = False) -> None:
        if force or not self._output_table_edit.text().strip():
            self._output_table_edit.setText(self._default_output_table_name())

    def _output_table_name(self) -> str:
        """Return the output table name, prefixed however the user typed it.

        The field is editable, so the prefix is applied here rather than only
        to the default: a hand-typed name is still a generated table.
        """
        return generated_table_name(
            self._output_table_edit.text().strip() or self._default_output_table_name(),
            fallback="Fit_Result",
        )

    # ------------------------------------------------------------------
    # SeriesOperationDialogBase hooks
    # ------------------------------------------------------------------

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
        return self._output_table_name()

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
                "generated_fit": True,
                "fit_dialog": "series_fit",
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
                "generated_fit": True,
                "fit_dialog": "series_fit",
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
        for check, kind, label, alpha in (
            (self._prediction_band_check, "pi", "95% prediction band", 0.12),
            (self._confidence_band_check, "ci", "95% confidence band", 0.25),
        ):
            if not check.isChecked():
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
                        "generated_fit": True,
                        "fit_dialog": "series_fit",
                        "fit_model": result.model_name,
                    },
                )
            )
        return specs

    def format_results(self, results: Sequence[SeriesFitResult]) -> str:
        """Return the fit report as an HTML table."""
        if not results:
            return "<p>Run Fit, or press Preview to draw the current parameters.</p>"
        return self._results_html(results[0])

    # ------------------------------------------------------------------
    # Optional accessory charts: residuals, and measured vs. fit
    # ------------------------------------------------------------------
    #
    # Both read straight off the output table _build_output_frame already
    # writes - x/residual always, plus y/y_fit (1D) or z/z_fit (2D) - so no
    # extra computation happens here, only a second axis and a query onto the
    # same table the overlay series already reads from.
    #
    # Each checkbox owns one axis, created on the first Preview or Apply that
    # has it checked and reused afterwards, the same way the spectral and
    # statistics dialogs reuse their own single result axis: an axis outside
    # the preview savepoint (creating one commits) would otherwise stack up
    # one per Preview click. Unchecking removes it immediately rather than
    # waiting for Close, since a chart nobody asked for any more should not
    # linger just because the dialog is still open.

    def _residual_axis_query(
        self, result: SeriesFitResult, table_name: str
    ) -> tuple[str, dict[str, str], str, str]:
        return (
            f'SELECT x, residual AS y FROM "{table_name}" ORDER BY x',
            {"x": "x", "y": "y"},
            result.x_col or "x",
            _("residual"),
        )

    def _fit_vs_measured_axis_query(
        self, result: SeriesFitResult, table_name: str
    ) -> tuple[str, dict[str, str], str, str]:
        fit_col, measured_col = ("z_fit", "z") if self._is_2d_fit() else ("y_fit", "y")
        return (
            f'SELECT {fit_col} AS x, {measured_col} AS y FROM "{table_name}"',
            {"x": "x", "y": "y"},
            _("fit"),
            result.target_col or _("measured"),
        )

    def _sync_accessory_axis(
        self,
        *,
        checkbox: QCheckBox,
        axis_id_attr: str,
        title: str,
        query_builder: Callable[[SeriesFitResult, str], tuple[str, dict[str, str], str, str]],
        result: SeriesFitResult,
        table_name: str,
    ) -> None:
        if not checkbox.isChecked():
            self._remove_accessory_axis(axis_id_attr)
            return

        sql_query, roles, x_label, y_label = query_builder(result, table_name)
        axis_id = getattr(self, axis_id_attr)
        if axis_id is None:
            axis_id = self.create_result_axis(
                chart_type="Scatter Plot",
                title=title,
                x_label=x_label,
                y_label=y_label,
                options={"grid": True},
            )
            setattr(self, axis_id_attr, axis_id)

        # This axis exists for exactly one series; replacing rather than
        # appending is what keeps repeated Previews from stacking copies.
        for row in self._repo.get_series(axis_id) or []:
            self._repo.delete_series(int(row["id"]))
        self._repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=self._repo.next_series_index(axis_id),
            name=title,
            sql_query=sql_query,
            roles=roles,
            style={"marker": "o", "linestyle": ""},
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

    def _sync_accessory_charts(self, result: SeriesFitResult, table_name: str) -> None:
        if not (
            self._residual_chart_check.isChecked()
            or self._fit_vs_measured_chart_check.isChecked()
        ):
            self._remove_accessory_axis("_residual_axis_id")
            self._remove_accessory_axis("_fit_vs_measured_axis_id")
            return

        # The accessory axes read the *final* output table rather than the
        # preview one, so it has to exist by now - on Preview it otherwise
        # would not, and both charts would be empty until Apply. Written
        # here, outside the preview savepoint, for the same reason the axes
        # are (see resolve_target_axis_id); Apply overwrites it with the same
        # content, and Close without Apply drops it again.
        self.write_result_table(table_name, result)
        self._accessory_table_name = table_name

        self._sync_accessory_axis(
            checkbox=self._residual_chart_check,
            axis_id_attr="_residual_axis_id",
            title=_("Residuals"),
            query_builder=self._residual_axis_query,
            result=result,
            table_name=table_name,
        )
        self._sync_accessory_axis(
            checkbox=self._fit_vs_measured_chart_check,
            axis_id_attr="_fit_vs_measured_axis_id",
            title=_("Measured vs. fit"),
            query_builder=self._fit_vs_measured_axis_query,
            result=result,
            table_name=table_name,
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
        if results:
            self._sync_accessory_charts(
                results[0], self.result_table_name(selected_axis_id, results[0])
            )
        return selected_axis_id

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

        table_name = self._accessory_table_name
        self._accessory_table_name = None
        if table_name:
            try:
                self._repo.delete_table(table_name)
            except Exception:
                applogger.exception(
                    "Failed to remove the accessory result table %s", table_name
                )

    def on_use_fit_results_as_initial(self) -> None:
        """Copy latest optimized fit parameters back to the Initial column."""
        if self._last_result is None:
            show_message(self, "series.no_fit_results")
            return
        params = np.asarray(self._last_result.params, dtype=float)
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
        self._params_table.resizeColumnsToContents()
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


def _evaluated_expression_html(
    expression: str, model_name: str, names: Sequence[str], params: np.ndarray
) -> str:
    """The expression plus the value each parameter took."""
    expression = expression or html_escape(model_name)
    rows = [
        f"{html_escape(names[row] if row < len(names) else f'p[{row}]')} = {float(value):.8g}"
        for row, value in enumerate(np.asarray(params, dtype=float))
    ]
    if not rows:
        return expression
    return f"{expression}<br><br><b>Evaluated parameters</b><br>" + "<br>".join(rows)
