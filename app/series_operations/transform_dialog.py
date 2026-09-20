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
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import BoolParam, ChoiceParam, IntParam
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

# --- Models -------------------------------------------------------------

TRANSFORM_POWER = "Power transform"
TRANSFORM_QUANTILE = "Quantile transform"
TRANSFORM_STANDARD = "Standard scaler"
TRANSFORM_ROBUST = "Robust scaler"

TRANSFORM_MODELS = (
    TRANSFORM_POWER,
    TRANSFORM_QUANTILE,
    TRANSFORM_STANDARD,
    TRANSFORM_ROBUST,
)

TRANSFORM_DOCS = {
    TRANSFORM_POWER: (
        "scikit-learn PowerTransformer",
        "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html",
    ),
    TRANSFORM_QUANTILE: (
        "scikit-learn QuantileTransformer",
        "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html",
    ),
    TRANSFORM_STANDARD: (
        "scikit-learn StandardScaler",
        "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
    ),
    TRANSFORM_ROBUST: (
        "scikit-learn RobustScaler",
        "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html",
    ),
}

POWER_YEO_JOHNSON = "yeo-johnson"
POWER_BOX_COX = "box-cox"

# Re-exported for callers/tests, matching calculus_dialog.py's own convention.
DEST_SAME_AXIS = SeriesOperationDialogBase.DEST_SAME_AXIS
DEST_NEW_AXIS = SeriesOperationDialogBase.DEST_NEW_AXIS
DEST_NEW_FIGURE = SeriesOperationDialogBase.DEST_NEW_FIGURE


@dataclass(slots=True)
class TransformResult:
    """One reshaped series for one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": self.x, "y": self.y})


class SeriesTransformDialog(SeriesOperationDialogBase):
    """Rescale or reshape a chart series' Y values."""

    Name: str = "Transform"
    Description = "Rescale or reshape a series' distribution"

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
        SeriesOperationDialogBase.destination_param(
            tooltip=(
                "A transformed series is usually on a different scale than its "
                "source (z-scores, quantiles, ...), so a new axis is the "
                "default. Same axis overlays them when the ranges are "
                "comparable; a new figure keeps the result out of this chart "
                "entirely."
            ),
            default=SeriesOperationDialogBase.DEST_NEW_AXIS,
        ),
    )

    Icon = """
    <path d="M4 15c4-8 5 8 9 0s5-8 7 0" transform="translate(0 -2)"/>
    <path d="M4 19h16" opacity="0.4"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesTransformDialog requires a repository instance.")

        self._last_results: list[TransformResult] = []
        self._parameter_form: QFormLayout | None = None
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Transform",
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

        self.model_combo.addItems(TRANSFORM_MODELS)
        self.model_combo.setToolTip(_("Choose the transform."))
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
        title, url = TRANSFORM_DOCS[self._model()]
        set_doc_link(self._doc_link, title, url)

    def _model(self) -> str:
        return self.model_combo.currentText() or TRANSFORM_POWER

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[TransformResult]:
        model = self._model()
        params = self.parameter_values()

        results: list[TransformResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                results.append(self._transform_one(name, x_values, y_values, model, params))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

    def _transform_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> TransformResult:
        column = y_values.reshape(-1, 1)
        metadata: dict[str, Any] = {}

        if model == TRANSFORM_POWER:
            method = str(params.get("method", POWER_YEO_JOHNSON))
            if method == POWER_BOX_COX and not np.all(y_values > 0):
                raise ValueError(
                    "Box-Cox requires every value to be strictly positive; "
                    "use Yeo-Johnson instead, or subtract a baseline first."
                )
            transformer = PowerTransformer(method=method)
            transformed = transformer.fit_transform(column)
            metadata["method"] = method

        elif model == TRANSFORM_QUANTILE:
            n_quantiles = min(int(params.get("n_quantiles", 1000)), y_values.size)
            transformer = QuantileTransformer(
                output_distribution=str(params.get("output_distribution", "uniform")),
                n_quantiles=max(2, n_quantiles),
                random_state=0,
            )
            transformed = transformer.fit_transform(column)
            metadata["n_quantiles"] = max(2, n_quantiles)

        elif model == TRANSFORM_ROBUST:
            transformer = RobustScaler(
                with_centering=bool(params.get("with_centering", True)),
                with_scaling=bool(params.get("with_scaling", True)),
            )
            transformed = transformer.fit_transform(column)

        else:
            transformer = StandardScaler()
            transformed = transformer.fit_transform(column)

        return TransformResult(
            source_name=name,
            result_name=f"{name} - {model}",
            model=model,
            x=x_values,
            y=np.asarray(transformed, dtype=float).ravel(),
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
        return self.resolve_destination_axis(
            selected_axis_id,
            chart_type="Scatter Plot",
            title=model,
            figure_name=f"{source} - {model}".strip(" -") or "Transform",
            options={"grid": True, "linestyle": "-", "marker": ""},
        )

    def discard_operation_artifacts(self) -> None:
        """Remove the axis or figure this dialog made, when Apply never ran."""
        self.discard_result_target()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: TransformResult) -> pd.DataFrame:
        return result.to_frame()

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
                "generated_transform": True,
                "transform_dialog": "series_transform",
                "source_name": result.source_name,
                "model": result.model,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_transform": True, "transform_dialog": "series_transform"}

    def result_table_name(self, axis_id: int, result: TransformResult) -> str:
        return generated_table_name(
            f"Transform_axis{axis_id}_{result.source_name}_{result.model}",
            fallback="Transform_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Transform"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[TransformResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        rows = [
            (
                result.source_name,
                result.model,
                report_html.format_number(result.y.size, digits=0),
                ", ".join(
                    f"{key}: {value}" for key, value in result.metadata.items() if value not in (None, "")
                ),
            )
            for result in results
        ]

        return report_html.document(
            _("Transform"),
            self._model(),
            report_html.section(
                _("Results"),
                report_html.table(
                    (_("Series"), _("Model"), _("Points"), _("Detail")),
                    rows,
                    align=("left", "left", "right", "left"),
                    empty_message=_("No results for this selection."),
                ),
            ),
        )
