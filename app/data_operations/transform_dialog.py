"""Rescale or reshape a chart series' Y distribution (todo.txt P2-15).

Unlike Calculus or Regression, a transform never changes what x means or
adds a curve beside the data - it replaces Y with a reshaped version of
itself, point for point, same length, same x. That is what makes it a
distinct operation rather than a mode of something else: a Box-Cox'd or
standardized series is meant to be worked on further (a fit, a control
chart) rather than compared against its own source on the same chart.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from app.analysis.transform import (
    KIND_POWER,
    KIND_QUANTILE,
    KIND_ROBUST,
    KIND_STANDARD,
    POWER_BOX_COX,
    POWER_YEO_JOHNSON,
    TransformSettings,
    transform_values,
)
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, IntParam
from app.plugin_api.data_operations.results import XYResult
from app.plugin_api.data_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    DataOperationDialogBase,
)
from app.utils import report_html

# --- Models -------------------------------------------------------------

TRANSFORM_POWER = "Power transform"
TRANSFORM_QUANTILE = "Quantile transform"
TRANSFORM_STANDARD = "Standard scaler"
TRANSFORM_ROBUST = "Robust scaler"

#: The models offered, in combo order, with their documentation.
TRANSFORM_MODELS: dict[str, OperationModel] = {
    TRANSFORM_POWER: OperationModel(
        doc_title="scikit-learn PowerTransformer",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html",
    ),
    TRANSFORM_QUANTILE: OperationModel(
        doc_title="scikit-learn QuantileTransformer",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html",
    ),
    TRANSFORM_STANDARD: OperationModel(
        doc_title="scikit-learn StandardScaler",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
    ),
    TRANSFORM_ROBUST: OperationModel(
        doc_title="scikit-learn RobustScaler",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html",
    ),
}

#: The engine's name for each model.
_KIND: dict[str, str] = {
    TRANSFORM_POWER: KIND_POWER,
    TRANSFORM_QUANTILE: KIND_QUANTILE,
    TRANSFORM_STANDARD: KIND_STANDARD,
    TRANSFORM_ROBUST: KIND_ROBUST,
}

# Re-exported for callers/tests, matching calculus_dialog.py's own convention.
DEST_SAME_AXIS = DataOperationDialogBase.DEST_SAME_AXIS
DEST_NEW_AXIS = DataOperationDialogBase.DEST_NEW_AXIS
DEST_NEW_FIGURE = DataOperationDialogBase.DEST_NEW_FIGURE


@dataclass(slots=True)
class TransformResult(XYResult):
    """One reshaped series for one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class SeriesTransformDialog(DataOperationDialogBase):
    """Rescale or reshape a chart series' Y values."""

    #: Where the result goes: the Charts tab's "Draw on".
    DESTINATIONS = (DataOperationDialogBase.DEST_SAME_AXIS, DataOperationDialogBase.DEST_NEW_AXIS, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_NEW_AXIS

    MODELS = TRANSFORM_MODELS
    MODEL_TOOLTIP = 'Choose the transform.'

    Name: str = "Transform"
    Description = "Rescale or reshape a series' distribution"

    #: Computed on a worker thread: see DataOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    PARAMS = (
        ChoiceParam(
            "method",
            "Method:",
            tooltip=(
                "Yeo-Johnson accepts any data; Box-Cox is often a better fit "
                "but requires every value to be strictly positive."
            ),
            choices=(
                ("Yeo-Johnson", POWER_YEO_JOHNSON),
                ("Box-Cox", POWER_BOX_COX),
            ),
            default_value=POWER_YEO_JOHNSON,
            visible_for={"model": (TRANSFORM_POWER,)},
        ),
        ChoiceParam(
            "output_distribution",
            "Target shape:",
            tooltip="The distribution the transformed values are mapped onto.",
            choices=(
                ("Uniform", "uniform"),
                ("Normal", "normal"),
            ),
            default_value="uniform",
            visible_for={"model": (TRANSFORM_QUANTILE,)},
        ),
        IntParam(
            "n_quantiles",
            "Quantiles:",
            tooltip=(
                "How finely the mapping is estimated. Clipped to the number of "
                "points in the series when the series is shorter than this."
            ),
            default_value=1000,
            minimum=2,
            maximum=100_000,
            visible_for={"model": (TRANSFORM_QUANTILE,)},
        ),
        BoolParam(
            "with_centering",
            "Centre on the median:",
            tooltip="Subtract the median before scaling.",
            default_value=True,
            visible_for={"model": (TRANSFORM_ROBUST,)},
        ),
        BoolParam(
            "with_scaling",
            "Scale by the IQR:",
            tooltip="Divide by the interquartile range.",
            default_value=True,
            visible_for={"model": (TRANSFORM_ROBUST,)},
        ),
    )

    Icon = """
    <path d="M4 15c4-8 5 8 9 0s5-8 7 0" transform="translate(0 -2)"/>
    <path d="M4 19h16" opacity="0.4"/>
    """
    Keywords: str = (
        "trasforma trasformazione log logarithm logaritmo box-cox normalize normalizza standardize "
        "standardizza scale scala rank"
    )
    Goals: list[str] = ["prepare", "describe"]
    Category: str = "Transform"
    Order: int = 90

    WINDOW_SIZE = (760, 640)
    SELECTS_ALL_SERIES = True

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_series(
        self, name: str, data: tuple[np.ndarray, np.ndarray], settings: tuple[str, dict[str, Any]]
    ) -> TransformResult:
        model, params = settings
        x_values, y_values = data
        return self._transform_one(name, x_values, y_values, model, params)

    def _transform_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> TransformResult:
        transformed = transform_values(
            _KIND[model],
            y_values,
            TransformSettings(
                method=str(params.get("method", POWER_YEO_JOHNSON)),
                output_distribution=str(params.get("output_distribution", "uniform")),
                n_quantiles=int(params.get("n_quantiles", 1000)),
                with_centering=bool(params.get("with_centering", True)),
                with_scaling=bool(params.get("with_scaling", True)),
            ),
        )
        return TransformResult(
            source_name=name,
            result_name=f"{name} - {model}",
            model=model,
            x=x_values,
            y=transformed.y,
            metadata=transformed.details,
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
        return self.resolve_destination_axis(
            selected_axis_id,
            chart_type="Scatter Plot",
            title=model,
            figure_name=self.result_figure_name(results, model),
            options={"grid": True, "linestyle": "-", "marker": ""},
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: TransformResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                **self.generated_style_filter,
                "source_name": result.source_name,
                "model": result.model,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    RESULT_TABLE_PREFIX = 'Transform'
    RESULT_TABLE_VARIANT = 'model'

    RESULT_COLUMNS = (("Series", "left"), ("Model", "left"), ("Points", "right"), ("Detail", "left"))

    def result_row(self, result: TransformResult) -> tuple[str, ...]:
        return (result.source_name, result.model, report_html.format_number(result.y.size, digits=0),
                self.details(result.metadata))
