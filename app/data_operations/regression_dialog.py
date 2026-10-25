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
from app.analysis.regression import (
    KIND_GRADIENT_BOOSTING,
    KIND_HUBER,
    KIND_ISOTONIC,
    KIND_RANDOM_FOREST,
    KIND_RANSAC,
    RegressionSettings,
    fit_regression,
)
from app.plugin_api.data_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.plugin_api.data_operations.results import XYResult
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import (
    OperationModel,
    DataOperationDialogBase,
)
from app.utils import report_html

# --- Models -------------------------------------------------------------

REGRESSION_RANSAC = "RANSAC (robust linear)"
REGRESSION_HUBER = "Huber (robust linear)"
REGRESSION_ISOTONIC = "Isotonic (monotone)"
REGRESSION_RANDOM_FOREST = "Random Forest"
REGRESSION_GRADIENT_BOOSTING = "Gradient Boosting"

#: The models offered, in combo order, with their documentation.
REGRESSION_MODELS: dict[str, OperationModel] = {
    REGRESSION_RANSAC: OperationModel(
        doc_title="scikit-learn RANSACRegressor",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html",
    ),
    REGRESSION_HUBER: OperationModel(
        doc_title="scikit-learn HuberRegressor",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html",
    ),
    REGRESSION_ISOTONIC: OperationModel(
        doc_title="Isotonic regression",
        doc_url="https://en.wikipedia.org/wiki/Isotonic_regression",
    ),
    REGRESSION_RANDOM_FOREST: OperationModel(
        doc_title="Random forest",
        doc_url="https://en.wikipedia.org/wiki/Random_forest",
    ),
    REGRESSION_GRADIENT_BOOSTING: OperationModel(
        doc_title="Gradient boosting",
        doc_url="https://en.wikipedia.org/wiki/Gradient_boosting",
    ),
}

#: The engine's name for each model.
_KIND: dict[str, str] = {
    REGRESSION_RANSAC: KIND_RANSAC,
    REGRESSION_HUBER: KIND_HUBER,
    REGRESSION_ISOTONIC: KIND_ISOTONIC,
    REGRESSION_RANDOM_FOREST: KIND_RANDOM_FOREST,
    REGRESSION_GRADIENT_BOOSTING: KIND_GRADIENT_BOOSTING,
}

# Re-exported for callers/tests, matching calculus_dialog.py's own convention.
DEST_SAME_AXIS = DataOperationDialogBase.DEST_SAME_AXIS
DEST_NEW_AXIS = DataOperationDialogBase.DEST_NEW_AXIS
DEST_NEW_FIGURE = DataOperationDialogBase.DEST_NEW_FIGURE


@dataclass(slots=True)
class RegressionResult(XYResult):
    """One fitted curve for one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class SeriesRegressionDialog(DataOperationDialogBase):
    """Fit a robust or non-parametric regression model to a chart series."""

    #: Where the result goes: the Charts tab's "Draw on".
    DESTINATIONS = (DataOperationDialogBase.DEST_SAME_AXIS, DataOperationDialogBase.DEST_NEW_AXIS, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_SAME_AXIS

    MODELS = REGRESSION_MODELS
    MODEL_TOOLTIP = 'Choose the regression model.'

    Name: str = "Regression"
    Description = "Robust and ML regression models"

    #: Computed on a worker thread: see DataOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

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
    )

    Icon = """
    <path d="M4 18c3-1 4-8 6-8s2 6 4 6 3-9 6-9"/>
    <circle cx="6" cy="15" r="1"/>
    <circle cx="12" cy="9" r="1"/>
    <circle cx="16" cy="14" r="1"/>
    """
    Keywords: str = (
        "regressione robust robusta machine learning random forest gradient boosting svr huber ransac "
        "theil-sen ml"
    )
    Goals: list[str] = ["relate", "predict"]
    Category: str = "Predictive and Specialized Modeling"
    Order: int = 33

    WINDOW_SIZE = (760, 640)
    SELECTS_ALL_SERIES = True

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_series(
        self, name: str, data: tuple[np.ndarray, np.ndarray], settings: tuple[str, dict[str, Any]]
    ) -> RegressionResult:
        model, params = settings
        x_values, y_values = data
        return self._fit_one(name, x_values, y_values, model, params)

    def _fit_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> RegressionResult:
        fitted = fit_regression(
            _KIND[model],
            x_values,
            y_values,
            RegressionSettings(
                max_trials=int(params.get("max_trials", 100)),
                epsilon=float(params.get("epsilon", 1.35)),
                alpha=float(params.get("alpha", 0.0001)),
                increasing=str(params.get("increasing", "auto")),
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=int(params.get("max_depth", 0)),
                learning_rate=float(params.get("learning_rate", 0.1)),
            ),
        )
        return RegressionResult(
            source_name=name,
            result_name=f"{name} - {model}",
            model=model,
            x=fitted.x,
            y=fitted.y,
            metadata=fitted.details,
        )

    # ------------------------------------------------------------------
    # Where the result is drawn
    # ------------------------------------------------------------------

    def resolve_target_axis_id(
        self,
        selected_axis_id: int,
        results: Sequence[Any],
    ) -> int:
        model = self.current_model()
        axis_id = self.resolve_destination_axis(
            selected_axis_id,
            chart_type="Scatter Plot",
            title=model,
            figure_name=self.result_figure_name(results, model),
            options={"grid": True, "linestyle": "-", "marker": ""},
        )
        return axis_id

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    RESULT_TABLE_PREFIX = 'Regression'
    RESULT_TABLE_VARIANT = 'model'

    RESULT_COLUMNS = (("Series", "left"), ("Model", "left"), ("Points", "right"), ("R²", "right"),
                      ("Detail", "left"))

    def how_to_read(self, results: Sequence[RegressionResult]) -> list[str]:
        result = results[0]
        return [reading_guide.r_squared((result.metadata or {}).get("r2"), y=result.source_name)]

    def result_row(self, result: RegressionResult) -> tuple[str, ...]:
        r2 = report_html.format_number(result.metadata["r2"]) if "r2" in result.metadata else "&mdash;"
        return (result.source_name, result.model, report_html.format_number(result.y.size, digits=0), r2,
                self.details(result.metadata, leave_out=("r2",)))
