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
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel

from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.series_operations.parameter_spec import ChoiceParam, FloatParam
from app.styles.style import create_doc_link, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

KERNEL_RBF = "RBF"
KERNEL_MATERN_32 = "Matern (nu=1.5)"
KERNEL_MATERN_52 = "Matern (nu=2.5)"
KERNEL_RATIONAL_QUADRATIC = "Rational Quadratic"

GP_KERNELS = (KERNEL_RBF, KERNEL_MATERN_32, KERNEL_MATERN_52, KERNEL_RATIONAL_QUADRATIC)

GP_DOCS = {
    KERNEL_RBF: (
        "RBF kernel",
        "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html",
    ),
    KERNEL_MATERN_32: (
        "Matern kernel (nu=1.5)",
        "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html",
    ),
    KERNEL_MATERN_52: (
        "Matern kernel (nu=2.5)",
        "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html",
    ),
    KERNEL_RATIONAL_QUADRATIC: (
        "Rational Quadratic kernel",
        "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html",
    ),
}

#: How many points the posterior mean/band is evaluated at, evenly spaced
#: across the source series' own x range.
GRID_POINTS = 200


def _build_kernel(name: str, *, length_scale: float, noise_level: float):
    """Return a base kernel plus additive white noise, ready to fit.

    The exposed length_scale/noise_level are a starting guess, not a frozen
    choice: GaussianProcessRegressor re-optimizes both (and every kernel
    hyperparameter) from here via its own internal likelihood maximization -
    see the dialog's own tooltips.
    """
    if name == KERNEL_MATERN_32:
        base = Matern(length_scale=length_scale, nu=1.5)
    elif name == KERNEL_MATERN_52:
        base = Matern(length_scale=length_scale, nu=2.5)
    elif name == KERNEL_RATIONAL_QUADRATIC:
        base = RationalQuadratic(length_scale=length_scale)
    else:
        base = RBF(length_scale=length_scale)
    return base + WhiteKernel(noise_level=noise_level)


@dataclass(slots=True)
class GPRegressionResult:
    """One source series' Gaussian-process fit: a mean curve and its band."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    mean: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"x": self.x, "mean": self.mean, "upper": self.upper, "lower": self.lower}
        )


class SeriesGPRegressionDialog(SeriesOperationDialogBase):
    """Fit a Gaussian process to a series and draw its mean +/-2 sigma band."""

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
        self.refresh_results()

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

        self.model_combo.addItems(GP_KERNELS)
        self.model_combo.setToolTip(_("Choose the kernel."))
        form.addRow(_("Kernel:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(container)
        return panel

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.refresh_results)

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        title, url = GP_DOCS[self._kernel()]
        set_doc_link(self._doc_link, title, url)

    def _kernel(self) -> str:
        return self.model_combo.currentText() or KERNEL_RBF

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self.set_results_text(f"Error:\n{exc}")
            return

        self._last_results = list(results)
        self.set_results_text(
            self.format_results(results)
            if results
            else _("Select one or more source series.")
        )

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
        kernel = _build_kernel(
            kernel_name,
            length_scale=float(params.get("length_scale", 1.0)),
            noise_level=float(params.get("noise_level", 1.0)),
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=0,
        )
        model.fit(x_values.reshape(-1, 1), y_values)

        x_grid = np.linspace(float(x_values.min()), float(x_values.max()), GRID_POINTS)
        mean, std = model.predict(x_grid.reshape(-1, 1), return_std=True)
        upper = mean + 2.0 * std
        lower = mean - 2.0 * std

        try:
            log_likelihood = float(model.log_marginal_likelihood())
        except Exception:  # noqa: BLE001 - diagnostic only, never fatal
            log_likelihood = float("nan")

        return GPRegressionResult(
            source_name=name,
            result_name=f"{name} - GP ({kernel_name})",
            model=kernel_name,
            x=x_grid,
            mean=mean,
            upper=upper,
            lower=lower,
            metadata={
                "kernel": repr(model.kernel_),
                "log_marginal_likelihood": log_likelihood,
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

    def discard_operation_artifacts(self) -> None:
        self.discard_result_target()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: GPRegressionResult) -> pd.DataFrame:
        return result.to_frame()

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

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_gp_regression": True, "gp_regression_dialog": "series_gp_regression"}

    def result_table_name(self, axis_id: int, result: GPRegressionResult) -> str:
        return generated_table_name(
            f"GP_axis{axis_id}_{result.source_name}_{result.model}",
            fallback="GP_Result",
        )

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
