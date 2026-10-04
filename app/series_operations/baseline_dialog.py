"""Baseline correction: subtract the slow background under a spectrum.

A spectrum, a chromatogram or any similarly peaky series often sits on a
wandering background - fluorescence, detector drift, scattering - that has
nothing to do with the peaks themselves. Peak heights, and especially peak
areas, are wrong until that background is estimated and removed. This is a
declared prerequisite for meaningful peak areas (see todo.txt P2-3).

Two estimators, in order of how much they assume:

* **Asymmetric Least Squares (AsLS)**, Eilers & Boelens 2005 - a smooth
  curve fitted to sit cheaply *below* the data and expensively above it: an
  asymmetric weighted least-squares fit against a roughness penalty,
  re-weighted a few times. Two knobs, ``lambda`` (smoothness) and ``p``
  (asymmetry), and it copes with overlapping or concave baselines an
  outline-based method cannot.
* **Rubber band** - the lower convex hull of the series, linearly
  interpolated between its vertices: literally a band stretched under the
  data from below. No parameters, but it can only ever be as good as a
  convex shape, so a baseline with its own concave dips is not one this
  method can follow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QFormLayout, QSpinBox, QWidget
from app.analysis.baseline import (
    ASLS_MINIMUM_POINTS,
    METHOD_ASLS,
    METHOD_RUBBER_BAND,
    correct_baseline,
)
from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.utils.i18n import _

BASELINE_ASLS = "Asymmetric Least Squares (AsLS)"
BASELINE_RUBBER = "Rubber band"
#: The models offered, in combo order, with their documentation.
BASELINE_MODELS: dict[str, OperationModel] = {
    BASELINE_ASLS: OperationModel(
        doc_title="Eilers & Boelens (2005), Asymmetric Least Squares Smoothing",
        doc_url="https://www.researchgate.net/publication/228961729_Asymmetric_least_squares_smoothing",
    ),
    BASELINE_RUBBER: OperationModel(
        doc_title="Rubber-band / convex-hull baseline",
        doc_url="https://en.wikipedia.org/wiki/Convex_hull",
    ),
}

@dataclass(slots=True)
class BaselineResult(TableResult):
    """One source series, its estimated baseline, and the corrected series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    corrected: np.ndarray
    baseline: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"x": self.x, "corrected": self.corrected, "baseline": self.baseline}
        )


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------
class SeriesBaselineDialog(SeriesOperationDialogBase):
    """Estimate and subtract the slow background under a series."""

    MODELS = BASELINE_MODELS
    MODEL_TOOLTIP = "Choose the baseline estimator."
    MODEL_LABEL = "Method:"

    Name: str = "Baseline Correction"
    Description = "Subtract a spectrum's background (AsLS or rubber band)"

    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = ASLS_MINIMUM_POINTS

    Icon = """
    <path d="M3 17c3 0 3-10 6-10s3 10 6 10 3-6 6-6"/>
    <path d="M3 19h18" stroke-dasharray="2 2"/>
    """

    def __init__(
        self, *, repo: SqliteRepo, figure_id: int, parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesBaselineDialog requires a repository instance.")

        self._last_results: list[BaselineResult] = []
        super().__init__(
            repo=repo, figure_id=figure_id, title="Baseline Correction", parent=parent,
            width=800, height=620,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        self._lambda_spin = QDoubleSpinBox(self)
        self._p_spin = QDoubleSpinBox(self)
        self._iterations_spin = QSpinBox(self)
        self._draw_baseline_check = QCheckBox("", self)

    def build_parameter_selector(self) -> QWidget:
        widget = QWidget(self)
        self._parameter_form = QFormLayout(widget)
        self._parameter_form.setContentsMargins(0, 0, 0, 0)
        self._parameter_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._lambda_spin.setRange(10.0, 1e10)
        self._lambda_spin.setDecimals(0)
        self._lambda_spin.setSingleStep(10_000.0)
        self._lambda_spin.setValue(100_000.0)
        self._lambda_spin.setToolTip(
            _("Smoothness. Larger makes the baseline stiffer, so it bends "
              "less to follow narrow peaks.")
        )
        self._parameter_form.addRow(_("Lambda:"), self._lambda_spin)

        self._p_spin.setRange(0.0001, 0.5)
        self._p_spin.setDecimals(4)
        self._p_spin.setSingleStep(0.001)
        self._p_spin.setValue(0.01)
        self._p_spin.setToolTip(
            _("Asymmetry. Smaller pushes the baseline further below the "
              "peaks; it should stay well under 0.5.")
        )
        self._parameter_form.addRow(_("p:"), self._p_spin)

        self._iterations_spin.setRange(1, 50)
        self._iterations_spin.setValue(10)
        self._iterations_spin.setToolTip(_("Re-weighting passes."))
        self._parameter_form.addRow(_("Iterations:"), self._iterations_spin)

        self._draw_baseline_check.setChecked(True)
        self._draw_baseline_check.setToolTip(_("Draw the estimated baseline itself."))
        self._parameter_form.addRow(_("Draw the baseline:"), self._draw_baseline_check)

        return widget

    def connect_operation_signals(self) -> None:
        super().connect_operation_signals()
        self._draw_baseline_check.toggled.connect(self.mark_results_stale)
        for spin in (self._lambda_spin, self._p_spin, self._iterations_spin):
            spin.valueChanged.connect(self.mark_results_stale)


    def _refresh_visibility(self) -> None:
        is_asls = self.current_model() == BASELINE_ASLS
        self.set_row_visible(self._lambda_spin, is_asls)
        self.set_row_visible(self._p_spin, is_asls)
        self.set_row_visible(self._iterations_spin, is_asls)


    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def compute_results(self) -> list[BaselineResult]:
        model = self.current_model()
        results: list[BaselineResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                is_asls = model == BASELINE_ASLS
                correction = correct_baseline(
                    METHOD_ASLS if is_asls else METHOD_RUBBER_BAND,
                    x_values,
                    y_values,
                    lam=float(self._lambda_spin.value()),
                    p=float(self._p_spin.value()),
                    iterations=int(self._iterations_spin.value()),
                )
                meta: dict[str, Any] = (
                    {"lambda": self._lambda_spin.value(), "p": self._p_spin.value()}
                    if is_asls else {}
                )
                meta["baseline area"] = correction.area
                baseline, corrected = correction.baseline, correction.corrected
                results.append(
                    BaselineResult(
                        source_name=name,
                        result_name=f"{name} - baseline corrected",
                        model=model,
                        x=x_values,
                        corrected=corrected,
                        baseline=baseline,
                        metadata=meta,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message)
        return results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _series_style(self, result: BaselineResult, **overrides: Any) -> dict[str, Any]:
        style: dict[str, Any] = {
            "generated_baseline": True,
            "baseline_dialog": "series_baseline",
            "source_name": result.source_name,
            "model": result.model,
        }
        style.update(overrides)
        return style

    def result_series_spec(
        self, axis_id: int, table_name: str, result: BaselineResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, corrected AS y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style=self._series_style(result, linestyle="-", linewidth=1.4, marker=""),
        )

    def result_series_specs(
        self, axis_id: int, table_name: str, result: BaselineResult,
    ) -> Sequence[ResultSeriesSpec]:
        specs: list[ResultSeriesSpec] = []
        if self._draw_baseline_check.isChecked():
            specs.append(
                ResultSeriesSpec(
                    name=f"{result.result_name} - baseline",
                    sql_query=f'SELECT x, baseline AS y FROM "{table_name}" ORDER BY x',
                    roles={"x": "x", "y": "y"},
                    style=self._series_style(
                        result, linestyle="--", linewidth=1.0, marker="", color="#9e9e9e"
                    ),
                )
            )
        specs.append(self.result_series_spec(axis_id, table_name, result))
        return specs

    RESULT_TABLE_PREFIX = 'Baseline'
    RESULT_TABLE_VARIANT = None

    @property
    def operation_label(self) -> str:
        return "Baseline Correction"

    def format_results(self, results: Sequence[BaselineResult]) -> str:
        if not results:
            return _("No results.")
        lines = []
        for result in results:
            details = ", ".join(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
                                 for key, value in result.metadata.items())
            lines.append(f"{result.source_name}: {result.model} ({details})")
        return "\n".join(lines)
