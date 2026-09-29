"""Gaussian process regression, with its own uncertainty band (todo.txt P2-15).

Every other regression-shaped operation in this app returns one curve. This
one is worth a dialog of its own because a GP is the one smoother here that
quantifies its own uncertainty: fitting it returns not just a mean curve but
a posterior standard deviation at every point, so the natural result is
three series - the mean, and a +/-2 sigma band flanking it - rather than one.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QWidget
from app.analysis import gaussian_process as gp
from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.series_operations.parameter_spec import FloatParam
from app.utils import report_html
from app.utils.i18n import _

KERNEL_RBF = "RBF"
KERNEL_MATERN_32 = "Matern (nu=1.5)"
KERNEL_MATERN_52 = "Matern (nu=2.5)"
KERNEL_RATIONAL_QUADRATIC = "Rational Quadratic"

#: The models offered, in combo order, with their documentation.
GP_KERNELS: dict[str, OperationModel] = {
    KERNEL_RBF: OperationModel(
        doc_title="RBF kernel",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html",
    ),
    KERNEL_MATERN_32: OperationModel(
        doc_title="Matern kernel (nu=1.5)",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html",
    ),
    KERNEL_MATERN_52: OperationModel(
        doc_title="Matern kernel (nu=2.5)",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html",
    ),
    KERNEL_RATIONAL_QUADRATIC: OperationModel(
        doc_title="Rational Quadratic kernel",
        doc_url="https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html",
    ),
}

#: The engine's name for each kernel.
_KERNEL_KEY: dict[str, str] = {
    KERNEL_RBF: gp.KERNEL_RBF,
    KERNEL_MATERN_32: gp.KERNEL_MATERN_32,
    KERNEL_MATERN_52: gp.KERNEL_MATERN_52,
    KERNEL_RATIONAL_QUADRATIC: gp.KERNEL_RATIONAL_QUADRATIC,
}


@dataclass(slots=True)
class GPRegressionResult(TableResult):
    """One source series' Gaussian-process fit: a mean curve and its band."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    mean: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"x": self.x, "mean": self.mean, "upper": self.upper, "lower": self.lower}
        )


class SeriesGPRegressionDialog(SeriesOperationDialogBase):
    """Fit a Gaussian process to a series and draw its mean +/-2 sigma band."""

    MODELS = GP_KERNELS
    MODEL_TOOLTIP = "Choose the kernel."
    MODEL_LABEL = "Kernel:"

    Name: str = "GP Regression"
    Description = "Gaussian process regression with an uncertainty band"

    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        FloatParam(
            "length_scale",
            "Length scale:",
            tooltip=(
                "Starting guess for how quickly the fit can vary with x. "
                "Re-optimized automatically while fitting - a starting point, "
                "not a fixed choice."
            ),
            default_value=1.0,
            minimum=1.0e-3,
            maximum=1000.0,
        ),
        FloatParam(
            "noise_level",
            "Noise level:",
            tooltip=(
                "Starting guess for the observation noise variance, added to "
                "the kernel. Also re-optimized while fitting."
            ),
            default_value=1.0,
            minimum=1.0e-6,
            maximum=100.0,
        ),
        SeriesOperationDialogBase.destination_param(
            tooltip=(
                "A Gaussian process' mean and band rarely share a scale with "
                "noisy source data, so a new axis is the default."
            ),
            default=SeriesOperationDialogBase.DEST_NEW_AXIS,
        ),
    )

    Icon = """
    <path d="M4 16c3-8 6-8 8 0s5 8 8 0" stroke-dasharray="2 2"/>
    <path d="M4 12c3-6 6-6 8 0s5 6 8 0"/>
    <path d="M4 8c3-4 6-4 8 0s5 4 8 0" stroke-dasharray="2 2"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesGPRegressionDialog requires a repository instance.")

        self._last_results: list[GPRegressionResult] = []
        self._parameter_form: QFormLayout | None = None
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series GP Regression",
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
        self._parameter_form = None

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()

    def _kernel(self) -> str:
        return self.model_combo.currentText() or KERNEL_RBF

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[GPRegressionResult]:
        kernel_name = self._kernel()
        params = self.parameter_values()

        results: list[GPRegressionResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                results.append(self._fit_one(name, x_values, y_values, kernel_name, params))
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
        kernel_name: str,
        params: Mapping[str, Any],
    ) -> GPRegressionResult:
        fitted = gp.fit_gaussian_process(
            _KERNEL_KEY[kernel_name],
            x_values,
            y_values,
            length_scale=float(params.get("length_scale", 1.0)),
            noise_level=float(params.get("noise_level", 1.0)),
        )
        return GPRegressionResult(
            source_name=name,
            result_name=f"{name} - GP ({kernel_name})",
            model=kernel_name,
            x=fitted.x,
            mean=fitted.mean,
            upper=fitted.upper,
            lower=fitted.lower,
            metadata={
                "kernel": fitted.kernel,
                "log_marginal_likelihood": fitted.log_marginal_likelihood,
            },
        )

    # ------------------------------------------------------------------
    # Where the result is drawn - three series from one saved table
    # ------------------------------------------------------------------

    def resolve_target_axis_id(
        self, selected_axis_id: int, results: Sequence[Any]
    ) -> int:
        axis_id = self.resolve_destination_axis(
            selected_axis_id,
            chart_type="Scatter Plot",
            title=self._kernel(),
            figure_name=self._result_figure_name(results),
            options={"grid": True, "linestyle": "-", "marker": ""},
        )
        self._label_result_axis(results)
        return axis_id

    def _result_figure_name(self, results: Sequence[Any]) -> str:
        source = str(getattr(results[0], "source_name", "")) if results else ""
        return f"{source} - GP Regression".strip(" -") or "GP Regression"

    def _label_result_axis(self, results: Sequence[Any]) -> None:
        if self._result_axis_id is None or not results:
            return
        try:
            self._repo.update_axis_descriptor(
                axis_id=self._result_axis_id,
                title=str(results[0].model or ""),
                x_label="x",
                y_label="y",
            )
        except Exception:
            applogger.exception("Failed to label the GP regression result axis")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_series_specs(
        self,
        axis_id: int,
        table_name: str,
        result: GPRegressionResult,
    ) -> Sequence[ResultSeriesSpec]:
        del axis_id
        base_style = {
            "generated_gp_regression": True,
            "gp_regression_dialog": "series_gp_regression",
            "source_name": result.source_name,
            "model": result.model,
        }
        band_style = {
            **base_style,
            "linestyle": "--",
            "linewidth": 1.0,
            "marker": "",
            "alpha": 0.6,
        }
        return [
            ResultSeriesSpec(
                name=f"{result.source_name} - GP mean",
                sql_query=f'SELECT x, mean AS y FROM "{table_name}" ORDER BY x',
                roles={"x": "x", "y": "y"},
                style={**base_style, "linestyle": "-", "linewidth": 1.8, "marker": ""},
            ),
            ResultSeriesSpec(
                name=f"{result.source_name} - GP +2σ",
                sql_query=f'SELECT x, upper AS y FROM "{table_name}" ORDER BY x',
                roles={"x": "x", "y": "y"},
                style=dict(band_style),
            ),
            ResultSeriesSpec(
                name=f"{result.source_name} - GP -2σ",
                sql_query=f'SELECT x, lower AS y FROM "{table_name}" ORDER BY x',
                roles={"x": "x", "y": "y"},
                style=dict(band_style),
            ),
        ]

    RESULT_TABLE_PREFIX = 'GP'
    RESULT_TABLE_VARIANT = 'model'

    @property
    def operation_label(self) -> str:
        return "GP Regression"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[GPRegressionResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        rows = [
            (
                result.source_name,
                result.model,
                report_html.format_number(result.metadata.get("log_marginal_likelihood")),
                str(result.metadata.get("kernel", "")),
            )
            for result in results
        ]

        return report_html.document(
            _("GP Regression"),
            self._kernel(),
            report_html.section(
                _("Results"),
                report_html.table(
                    (_("Series"), _("Kernel"), _("Log marginal likelihood"), _("Fitted kernel")),
                    rows,
                    align=("left", "left", "right", "left"),
                    empty_message=_("No results for this selection."),
                ),
            ),
        )
