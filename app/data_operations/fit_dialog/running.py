"""Running the fit: the job, its result, the results it reports."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from PySide6.QtWidgets import QWidget
from app.analysis import fit as fit_engine

from app.logs.logger import applogger

from app.styles.style import create_action_button

from app.data_operations.fit_dialog.common import SeriesFitResult, _FitJob, _evaluated_expression_html


class FitRunMixin:
    """Running the fit: the job, its result, the results it reports. A part of SeriesFitDialog (see dialog.py)."""

    if TYPE_CHECKING:  # GlobalFitMixin's
        def _fit_is_global(self) -> bool: ...


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
        # optimum has to land in it rather than only in the report - every
        # row of it, which in a global fit is more than one series' worth.
        self._set_initial_params(np.asarray(self._last_table_params, dtype=float))
        self._fill_results_table(result.params, result.param_std)
        self._btn_use_fit_params.setEnabled(True)
        self.publish_results(self.format_results(results))

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
        global_parts = None
        layout = getattr(self, "_table_layout", None)
        if layout is not None and x_data.ndim == 2:
            global_parts = (model, layout, list(self._global_names))
            model = fit_engine.global_model(model, layout)
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
            weighted=self._weights_key() == "relative",
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
            sigma=clean["sigma"].to_numpy(float) if "sigma" in clean.columns else None,
            global_parts=global_parts,
            separate=global_parts is not None and not self._fit_is_global(),
        )

    def _weights_key(self) -> str:
        return str(self._combo_weights.currentData() or "none")

    def finish_job(self, job: _FitJob, outcome: fit_engine.CurveFit) -> list[SeriesFitResult]:
        """The engine's outcome as this dialog's result; kept as _last_result."""
        if job.optimise and outcome.success:
            # Remembered so that Preview/OK, which only evaluate the table,
            # can still say the parameters they draw are a converged fit.
            self._last_fit = (job.model_name, np.asarray(outcome.params, dtype=float), outcome.message)
        elif not job.optimise:
            outcome.message = self._evaluation_message(job, outcome)
        self._last_table_params = np.asarray(outcome.params, dtype=float)
        parts = self._split_global(job, outcome) if job.global_parts is not None else [(job, outcome)]
        try:
            results = [self._result_from(part_job, part, optimise=job.optimise) for part_job, part in parts]
        except Exception:
            self._last_result = None
            raise
        self._last_result = results[0]
        return results

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
