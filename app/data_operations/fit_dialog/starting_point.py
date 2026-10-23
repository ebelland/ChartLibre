"""Algorithm and starting point: optimiser, loss, initial values; the results pane and output table."""
from __future__ import annotations

from html import escape as html_escape
import math
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QTableWidgetItem
from app.plugin_api.functions.optimizers import DEFAULT_OPTIMIZER, BY_KEY
from app.plugin_api.functions.starting_point import (
    FROM_DECLARED,
    FROM_FUNCTION,
    FROM_SEARCH,
    ask_the_function,
    choose_starting_point,
    clip_into_bounds,
)

from app.plugin_api.data_operations.dialog_base import generated_table_name
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.utils import report_html

from app.utils.i18n import _

from app.data_operations.fit_dialog.common import _SurfaceInitialGuessAdapter, SeriesFitResult, _FitJob


class StartingPointMixin:
    """Algorithm and starting point: optimiser, loss, initial values; the results pane and output table. A part of SeriesFitDialog (see dialog.py)."""

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
        except Exception:  # noqa: BLE001 - a plugin's own code
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
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Could not estimate initial values")
            show_message(
                self, "series.estimate_failed", series=self._current_table(), error=exc
            )
            return

        self._set_initial_params(start.values)
        applogger.info("Initial values from %s (cost %.6g)", start.source, start.cost)
        self.set_results_text(self._estimate_report(start.source, start.cost))

    def _fill_global_initial_from_estimator(self, function_class: Any) -> bool:
        """The function's own estimate, series by series (a global fit)."""
        try:
            series = self._per_series_xy()
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Could not estimate initial values")
            show_message(self, "series.estimate_failed", series=self._current_table(), error=exc)
            return
        theta = self._combine_estimates([np.asarray(start.values, float) for start in starts])
        if theta is None:
            return
        self._set_initial_params(clip_into_bounds(theta, lb, ub))
        cost = float(sum(start.cost for start in starts))
        applogger.info("Initial values, series by series (total cost %.6g)", cost)
        self.set_results_text(self._estimate_report(starts[0].source, cost))

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
