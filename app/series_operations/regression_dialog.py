"""Robust and machine-learning regression of a chart series (todo.txt P2-15).

A separate dialog from Fit rather than another model on its combo: Fit's
whole shape is "an algebraic expression with named parameters" - Gaussian,
polynomial, a user function - fitted through ``scipy.optimize.curve_fit``
and reported as those parameters. None of the five models here have that
shape. RANSAC and Huber are still linear regressions, but robust to
outliers rather than interpretable beyond a slope and an intercept;
Isotonic, Random Forest and Gradient Boosting are non-parametric - there is
no "amplitude, center, width" to report, only a fitted curve. Bolting them
onto Fit's model tree would mean pretending they belong to a family they
do not.

Every model here reads one series' (x, y), fits against it, and predicts
over a dense evenly-spaced grid across the series' own x range - not only
at the original x's, which is what keeps a Random Forest or Gradient
Boosting fit from reading as a jagged line through exactly the input
points.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, RANSACRegressor

from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

# --- Models -------------------------------------------------------------

REGRESSION_RANSAC = "RANSAC (robust linear)"
REGRESSION_HUBER = "Huber (robust linear)"
REGRESSION_ISOTONIC = "Isotonic (monotone)"
REGRESSION_RANDOM_FOREST = "Random Forest"
REGRESSION_GRADIENT_BOOSTING = "Gradient Boosting"

REGRESSION_MODELS = (
    REGRESSION_RANSAC,
    REGRESSION_HUBER,
    REGRESSION_ISOTONIC,
    REGRESSION_RANDOM_FOREST,
    REGRESSION_GRADIENT_BOOSTING,
)

REGRESSION_DOCS = {
    REGRESSION_RANSAC: (
        "scikit-learn RANSACRegressor",
        "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html",
    ),
    REGRESSION_HUBER: (
        "scikit-learn HuberRegressor",
        "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html",
    ),
    REGRESSION_ISOTONIC: (
        "Isotonic regression",
        "https://en.wikipedia.org/wiki/Isotonic_regression",
    ),
    REGRESSION_RANDOM_FOREST: (
        "Random forest",
        "https://en.wikipedia.org/wiki/Random_forest",
    ),
    REGRESSION_GRADIENT_BOOSTING: (
        "Gradient boosting",
        "https://en.wikipedia.org/wiki/Gradient_boosting",
    ),
}

#: Points in the dense evaluation grid every model predicts over.
GRID_POINTS = 200

# Re-exported for callers/tests, matching calculus_dialog.py's own convention.
DEST_SAME_AXIS = SeriesOperationDialogBase.DEST_SAME_AXIS
DEST_NEW_AXIS = SeriesOperationDialogBase.DEST_NEW_AXIS
DEST_NEW_FIGURE = SeriesOperationDialogBase.DEST_NEW_FIGURE


@dataclass(slots=True)
class RegressionResult:
    """One fitted curve for one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": self.x, "y": self.y})


class SeriesRegressionDialog(SeriesOperationDialogBase):
    """Fit a robust or non-parametric regression model to a chart series."""

    Name: str = "Regression"
    Description = "Robust and ML regression models"

    PARAMS = (
        IntParam(
            "max_trials",
            "Max trials:",
            tooltip="How many random subsets RANSAC tries before keeping the best one.",
            default_value=100,
            minimum=10,
            maximum=10_000,
            visible_for={"model": (REGRESSION_RANSAC,)},
        ),
        FloatParam(
            "epsilon",
            "Epsilon:",
            tooltip=(
                "Robustness/efficiency trade-off: values closer to 1 down-weight "
                "more points as outliers, larger values behave more like an "
                "ordinary least-squares fit."
            ),
            default_value=1.35,
            minimum=1.0,
            maximum=10.0,
            visible_for={"model": (REGRESSION_HUBER,)},
        ),
        FloatParam(
            "alpha",
            "Regularization:",
            tooltip="L2 regularization strength.",
            default_value=0.0001,
            minimum=0.0,
            maximum=10.0,
            decimals=6,
            step=0.0001,
            visible_for={"model": (REGRESSION_HUBER,)},
        ),
        ChoiceParam(
            "increasing",
            "Direction:",
            tooltip="Whether the fitted curve must rise, fall, or whichever the data itself suggests.",
            choices=(
                ("Auto", "auto"),
                ("Increasing", "true"),
                ("Decreasing", "false"),
            ),
            default_value="auto",
            visible_for={"model": (REGRESSION_ISOTONIC,)},
        ),
        IntParam(
            "n_estimators",
            "Trees:",
            tooltip="Number of trees in the ensemble. More is smoother and slower.",
            default_value=100,
            minimum=10,
            maximum=2000,
            visible_for={"model": (REGRESSION_RANDOM_FOREST, REGRESSION_GRADIENT_BOOSTING)},
        ),
        IntParam(
            "max_depth",
            "Max depth:",
            tooltip="Deepest a single tree may grow. 0 means unlimited.",
            default_value=0,
            minimum=0,
            maximum=100,
            visible_for={"model": (REGRESSION_RANDOM_FOREST, REGRESSION_GRADIENT_BOOSTING)},
        ),
        FloatParam(
            "learning_rate",
            "Learning rate:",
            tooltip="How much each boosting stage corrects the previous one.",
            default_value=0.1,
            minimum=0.001,
            maximum=1.0,
            decimals=3,
            step=0.01,
            visible_for={"model": (REGRESSION_GRADIENT_BOOSTING,)},
        ),
        SeriesOperationDialogBase.destination_param(
            tooltip=(
                "A fitted curve usually shares its source's scale, so the same "
                "axis overlays it directly on the data. A new axis or figure "
                "keeps the two apart instead."
            ),
            default=SeriesOperationDialogBase.DEST_SAME_AXIS,
        ),
    )

    Icon = """
    <path d="M4 18c3-1 4-8 6-8s2 6 4 6 3-9 6-9"/>
    <circle cx="6" cy="15" r="1"/>
    <circle cx="12" cy="9" r="1"/>
    <circle cx="16" cy="14" r="1"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesRegressionDialog requires a repository instance.")

        self._last_results: list[RegressionResult] = []
        self._parameter_form: QFormLayout | None = None
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Regression",
            parent=parent,
            width=760,
            height=640,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        container = QWidget(panel)
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(REGRESSION_MODELS)
        self.model_combo.setToolTip(_("Choose the regression model."))
        form.addRow(_("Model:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(container)
        return panel

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.mark_results_stale)

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        title, url = REGRESSION_DOCS[self._model()]
        set_doc_link(self._doc_link, title, url)

    def _model(self) -> str:
        return self.model_combo.currentText() or REGRESSION_RANSAC

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[RegressionResult]:
        model = self._model()
        params = self.parameter_values()

        results: list[RegressionResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                results.append(self._fit_one(name, x_values, y_values, model, params))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

    def _fit_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> RegressionResult:
        x_grid = np.linspace(float(x_values.min()), float(x_values.max()), GRID_POINTS)
        metadata: dict[str, Any] = {}

        if model == REGRESSION_ISOTONIC:
            increasing_choice = str(params.get("increasing", "auto"))
            increasing: bool | str = (
                increasing_choice
                if increasing_choice == "auto"
                else increasing_choice == "true"
            )
            estimator = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            estimator.fit(x_values, y_values)
            y_grid = estimator.predict(x_grid)
        else:
            X = x_values.reshape(-1, 1)
            X_grid = x_grid.reshape(-1, 1)

            if model == REGRESSION_RANSAC:
                estimator = RANSACRegressor(
                    max_trials=int(params.get("max_trials", 100)), random_state=0
                )
                estimator.fit(X, y_values)
                inlier_mask = getattr(estimator, "inlier_mask_", None)
                if inlier_mask is not None:
                    metadata["inliers"] = f"{int(np.count_nonzero(inlier_mask))}/{inlier_mask.size}"

            elif model == REGRESSION_HUBER:
                estimator = HuberRegressor(
                    epsilon=float(params.get("epsilon", 1.35)),
                    alpha=float(params.get("alpha", 0.0001)),
                )
                estimator.fit(X, y_values)
                n_outliers = getattr(estimator, "outliers_", None)
                if n_outliers is not None:
                    metadata["outliers"] = int(np.count_nonzero(n_outliers))

            elif model == REGRESSION_RANDOM_FOREST:
                max_depth = int(params.get("max_depth", 0)) or None
                estimator = RandomForestRegressor(
                    n_estimators=int(params.get("n_estimators", 100)),
                    max_depth=max_depth,
                    random_state=0,
                )
                estimator.fit(X, y_values)

            else:
                max_depth = int(params.get("max_depth", 0)) or None
                estimator = GradientBoostingRegressor(
                    n_estimators=int(params.get("n_estimators", 100)),
                    learning_rate=float(params.get("learning_rate", 0.1)),
                    max_depth=max_depth or 3,
                    random_state=0,
                )
                estimator.fit(X, y_values)

            y_grid = estimator.predict(X_grid)
            try:
                metadata["r2"] = float(estimator.score(X, y_values))
            except Exception:  # noqa: BLE001 - not every estimator scores the same way
                pass

        return RegressionResult(
            source_name=name,
            result_name=f"{name} - {model}",
            model=model,
            x=x_grid,
            y=np.asarray(y_grid, dtype=float),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Where the result is drawn
    # ------------------------------------------------------------------

    def apply(self) -> bool:
        applied = super().apply()
        self._applied = self._applied or applied
        return applied

    def resolve_target_axis_id(
        self,
        selected_axis_id: int,
        results: Sequence[Any],
    ) -> int:
        source = str(getattr(results[0], "source_name", "")) if results else ""
        model = self._model()
        axis_id = self.resolve_destination_axis(
            selected_axis_id,
            chart_type="Scatter Plot",
            title=model,
            figure_name=f"{source} - {model}".strip(" -") or "Regression",
            options={"grid": True, "linestyle": "-", "marker": ""},
        )
        return axis_id

    def discard_operation_artifacts(self) -> None:
        """Remove the axis or figure this dialog made, when Apply never ran."""
        self.discard_result_target()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: RegressionResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: RegressionResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_regression": True,
                "regression_dialog": "series_regression",
                "source_name": result.source_name,
                "model": result.model,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_regression": True, "regression_dialog": "series_regression"}

    def result_table_name(self, axis_id: int, result: RegressionResult) -> str:
        return generated_table_name(
            f"Regression_axis{axis_id}_{result.source_name}_{result.model}",
            fallback="Regression_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Regression"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[RegressionResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        rows = [
            (
                result.source_name,
                result.model,
                report_html.format_number(result.y.size, digits=0),
                report_html.format_number(result.metadata["r2"])
                if "r2" in result.metadata
                else "&mdash;",
                ", ".join(
                    f"{key}: {value}"
                    for key, value in result.metadata.items()
                    if key != "r2" and value not in (None, "")
                ),
            )
            for result in results
        ]

        return report_html.document(
            _("Regression"),
            self._model(),
            report_html.section(
                _("Results"),
                report_html.table(
                    (_("Series"), _("Model"), _("Points"), _("R²"), _("Detail")),
                    rows,
                    align=("left", "left", "right", "right", "left"),
                    empty_message=_("No results for this selection."),
                ),
            ),
        )
