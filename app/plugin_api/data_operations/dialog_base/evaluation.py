"""Computing: the results text, the generated-series pipeline, evaluation here or on a worker thread."""
from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget

from app.analysis import Stopped
from app.data.data_source import parse_roles, row_value
from app.logs.logger import applogger
from app.widgets.html_results import looks_like_html
from app.utils.background import BackgroundTask, run_in_background
from app.utils.i18n import _

from app.plugin_api.data_operations.dialog_base.common import OperationJob, SeriesOutcome, SeriesJob
from app.utils.config import get_constant


class EvaluationMixin:
    """Computing: the results text, the generated-series pipeline, evaluation here or on a worker thread. A part of DataOperationDialogBase (see dialog.py)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def selected_series(self) -> list[Any]:
        if self.reads_table() and not getattr(self, "_switching_figure", False):
            # The table's columns, drawn if they are not yet.
            try:
                self._prepare_table_source()
            except ValueError:
                return []
        return self.series_selector.selected_series()

    def _selected_series_row(self) -> Any:
        rows = self.selected_series()

        if not rows:
            message = "Select one source series in the Data panel."
            applogger.error(message)
            raise ValueError(message)

        if len(rows) > 1:
            # A warning, not a message box: this runs on every Preview and
            # every parameter tweak while two or more series are checked.
            # The status bar shows it without asking to be dismissed.
            name = row_value(rows[0], "name", default="first")
            applogger.warning(
                "%d series are selected; using only '%s'. Uncheck the "
                "others, or reorder them, to use a different one.",
                len(rows),
                name,
            )

        return rows[0]
    

    #: Whether ``format_results`` returns markup rather than plain text.
    #: A dialog that builds a table has to set this, or its markup is escaped
    #: and the user reads ``<tr><td>`` instead of a table - which is what the
    #: spectral dialog did, and why the cluster dialog gave up on HTML.
    RESULTS_ARE_HTML: bool = False

    def set_results_text(self, text: str) -> None:
        """Show results in the results pane.

        Several operation dialogs return generated HTML reports while older
        callers still pass plain-text summaries through this same method.
        Route HTML-looking content to ``setHtml`` so tags render as markup,
        and route everything else through ``setText`` so plain text remains
        escaped and safe.
        """
        content = str(text or "")

        if looks_like_html(content):
            self._results_label.setHtml(content)
            return

        if hasattr(self._results_label, "setContent"):
            self._results_label.setContent(content)
        else:
            self._results_label.setText(content)

    def set_results_html(self, markup: str) -> None:
        """Show markup in the results pane."""
        self._results_label.setHtml(str(markup or ""))

    def publish_results(self, formatted: str) -> None:
        """Show ``format_results`` output, as text or as markup.

        One entry point so every code path - preview, apply, refresh - honours
        RESULTS_ARE_HTML.  Three of them used to call set_results_text
        directly, so a dialog that returned HTML had it escaped in some places
        and rendered in others.
        """
        if self.RESULTS_ARE_HTML:
            self.set_results_html(formatted)
        else:
            self.set_results_text(formatted)

    def mark_results_stale(self, *_ignored: Any) -> None:
        """Invalidate any previewed results without recomputing anything.

        Wired wherever a control used to call refresh_results()/
        _queue_refresh_results() straight away. Preview and OK are the only
        paths left that actually run compute_results().
        """
        self.store_cached_results([])
        self.publish_results(_(self.PENDING_RESULTS_MESSAGE))

    # ------------------------------------------------------------------
    # Shared generated-series apply pipeline
    # ------------------------------------------------------------------
    @property
    def operation_label(self) -> str:
        """Human readable operation name used in messages/logs: the operation's Name."""
        return self.Name or self.windowTitle() or self.__class__.__name__

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        """Style key/value pairs identifying series generated by this dialog.

        Example::

            {"generated_smoothing": True, "smoothing_dialog": "series_smoothing"}

        ``remove_previous_generated_series`` deletes only descriptors whose
        ``style_json`` contains all these key/value pairs.

        Derived from the source file's name by default - smoothing_dialog.py
        gives the example above; an operation that is a package
        (fit_dialog/dialog.py) takes its folder's name. The file, not the
        module: the scanner loads a plugin under a module name of its own,
        different every session (_dynamic_data_operation_..._<hash>), and
        keys made from that were neither stable nor the ones the series were
        stored with. Four operations predate the rule and keep their own
        values, which are stored in existing projects.
        """
        path = Path(inspect.getfile(type(self)))
        name = path.stem if path.stem.endswith("_dialog") else path.parent.name
        stem = name.removesuffix("_dialog")
        return {f"generated_{stem}": True, f"{stem}_dialog": f"series_{stem}"}

    def compute_results(self) -> Sequence[Any]:
        """Return operation-specific result objects for the selected series.

        Preview and Apply call this, on the GUI thread, unless the operation
        runs in the background (see evaluate). An operation that implements
        prepare_job/finish_job - or, more simply, compute_series - gets it for
        free; one that has not been split that way yet overrides this instead.
        """
        job = self.prepare_job()
        if job is None:
            return []
        return self.finish_job(job, job.run())

    # ------------------------------------------------------------------
    # Evaluation: one way to compute, here or on a worker thread
    # ------------------------------------------------------------------

    #: Whether Preview and Apply compute on a worker thread. Only for an
    #: operation that has a job (prepare_job or compute_series): the window
    #: stays responsive, Stop appears with a progress bar if the run takes
    #: more than a moment, and closing the window stops it.
    RUN_IN_BACKGROUND: bool = False

    #: Off switch for every operation at once - the tests turn it off so a
    #: Preview has finished by the time it returns. See dev/tests/conftest.py.
    BACKGROUND_ENABLED: ClassVar[bool] = True

    #: How long a background run may take before Stop and the progress bar
    #: appear: a run that ends sooner shows nothing, rather than a flicker.
    BUSY_DELAY_MS: ClassVar[int] = get_constant("background_busy_delay_ms", 300)

    # -- Per series: the common case ------------------------------------

    def series_settings(self) -> Any:
        """Everything compute_series needs besides the data, read on the GUI thread.

        ``(model, parameters)`` for an operation with MODELS, else the parameters.
        """
        if self.MODELS:
            return self.current_model(), self.parameter_values()
        return self.parameter_values()

    #: Whether a series with a z role is read as a surface: read_series then
    #: gives read_curve_or_surface's ``(is_surface, data)`` instead of x and y.
    READS_SURFACES: bool = False

    def read_series(self, row: Any, name: str) -> Any:
        """One selected series' data, read on the GUI thread; x and y by default."""
        if self.READS_SURFACES:
            return self.read_curve_or_surface(row, name)
        return self.series_xy(row, name)

    def read_curve_or_surface(self, row: Any, name: str) -> tuple[bool, Any]:
        """``(True, series_grid_xyz)`` for a series with a z role - a surface -
        else ``(False, series_xy)``: for the operations that handle both."""
        if parse_roles(row_value(row, "roles", default={})).get("z"):
            return True, self.series_grid_xyz(row, name)
        return False, self.series_xy(row, name)

    def compute_series(self, name: str, data: Any, settings: Any) -> Any:
        """The calculation for one series, on whatever thread runs the job.

        Must not touch a widget, the repository or the logger's dialogs:
        everything it needs arrives in *data* and *settings*. Whatever it
        returns goes to finish_series, back on the GUI thread.
        """
        raise NotImplementedError

    def finish_series(self, name: str, outcome: Any, settings: Any) -> Any:
        """Turn compute_series' outcome into a result, on the GUI thread (notes go to the log here)."""
        del name, settings
        return outcome

    def _computes_per_series(self) -> bool:
        from app.plugin_api.data_operations.dialog_base.dialog import DataOperationDialogBase  # the class this is part of

        return type(self).compute_series is not DataOperationDialogBase.compute_series

    def prepare_job(self, **options: Any) -> OperationJob | None:
        """Read the widgets into a job, on the GUI thread; None when there is nothing to do.

        The job carries everything the calculation needs - data, parameters,
        and whatever names the result will need later - so the window can be
        changed while it runs without mixing two runs. *options* are the
        operation's own (the fit's ``optimise``, for one).

        An operation with compute_series gets a SeriesJob: the selected
        series are read here, one by one, and a series that cannot be read
        is reported with the ones that fail to compute.
        """
        if not self._computes_per_series():
            raise NotImplementedError(
                f"{type(self).__name__} implements neither prepare_job nor compute_results"
            )
        del options
        settings = self.series_settings()
        inputs: list[tuple[str, Any]] = []
        errors: list[str] = []
        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                inputs.append((name, self.read_series(row, name)))
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")
        if not inputs and not errors:
            return None
        return SeriesJob(inputs, settings, self.compute_series, errors)

    def finish_job(self, job: OperationJob, outcome: Any) -> Sequence[Any]:
        """Turn a job's outcome into this dialog's results, on the GUI thread."""
        if isinstance(job, SeriesJob) and isinstance(outcome, SeriesOutcome):
            results = [self.finish_series(name, value, job.settings) for name, value in outcome.outcomes]
            # One bad series should not hide the good ones; only when none
            # worked is it an error.
            if outcome.errors and not results:
                raise ValueError("; ".join(outcome.errors))
            for message in outcome.errors:
                applogger.warning(message)
            return results
        del job
        return list(outcome) if isinstance(outcome, (list, tuple)) else [outcome]

    def busy_widgets(self) -> list[QWidget]:
        """What is switched off while a background calculation runs.

        Everything that would start a second run or act on stale results;
        subclasses add their own buttons (the fit's Fit and Estimate).
        """
        return [self.preview_button, self.apply_button]

    @property
    def evaluating(self) -> bool:
        task = getattr(self, "_evaluation_task", None)
        return task is not None and task.running

    def evaluate(
        self,
        on_results: Callable[[Sequence[Any]], None],
        *,
        background: bool | None = None,
        **options: Any,
    ) -> None:
        """Compute the results and hand them to *on_results*, on the GUI thread.

        In the background when *background* says so (RUN_IN_BACKGROUND when
        it does not): busy_widgets() are off, Stop and a progress bar appear
        once the run has taken BUSY_DELAY_MS, and closing the window stops the
        run. *on_results* is not called when the run is stopped or fails; the
        reason is shown instead.
        """
        if self.evaluating:
            return
        job = self.prepare_job(**options)
        if job is None:
            return
        if not (self.RUN_IN_BACKGROUND if background is None else background):
            on_results(self.finish_job(job, job.run()))
            return

        def finished(outcome: Any) -> None:
            stopped = task.cancel_event.is_set()
            self._set_evaluating(False)
            if stopped:
                # Asked to stop, but the job finished before it looked: the
                # window may already be closing, so nothing is written.
                self._on_evaluation_failed(Stopped())
                return
            try:
                results = self.finish_job(job, outcome)
            except Exception as exc:  # noqa: BLE001 - shown like a failed run
                self._on_evaluation_failed(exc)
                return
            on_results(results)

        self._evaluation_progress = getattr(job, "progress", None)
        self._set_evaluating(True)
        task = run_in_background(
            lambda cancel: job.run(should_stop=cancel.is_set),
            finished,
            self._on_evaluation_failed,
        )
        self._evaluation_task = task

    def stop_evaluation(self) -> None:
        """Ask a background calculation to stop at its next step."""
        task: BackgroundTask | None = getattr(self, "_evaluation_task", None)
        if task is not None and task.running:
            task.cancel()

    def _on_evaluation_failed(self, error: BaseException) -> None:
        self._set_evaluating(False)
        if isinstance(error, Stopped):
            self.set_results_text(_("Stopped. Nothing was changed."))
            return
        applogger.error(f"{self.operation_label} failed: {error}")

    def _set_evaluating(self, running: bool) -> None:
        # The buttons go off at once - a second click must not start a second
        # run - but Stop and the bar wait: most runs end before they would
        # have been read.
        for widget in self.busy_widgets():
            widget.setEnabled(not running)
        timer = getattr(self, "_busy_timer", None)
        if timer is None:
            timer = self._busy_timer = QTimer(self)
            timer.setInterval(100)
            timer.timeout.connect(self._show_progress)
        if running:
            self._busy_since = 0
            timer.start()
            return
        timer.stop()
        self.stop_button.setVisible(False)
        self.progress_bar.setVisible(False)

    def _show_progress(self) -> None:
        """Every 100 ms of a background run: Stop and the bar once it is slow, then the count."""
        self._busy_since = getattr(self, "_busy_since", 0) + 100
        if self._busy_since < self.BUSY_DELAY_MS:
            return
        if not self.stop_button.isVisible():
            self.stop_button.setVisible(True)
            self.progress_bar.setVisible(True)
            self.set_results_text(_("Working... press Stop to end it early."))
        progress = getattr(self, "_evaluation_progress", None)
        done, total = (progress[0], progress[1]) if progress else (0, 0)
        if total > 1:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        else:
            # One series, or a job that does not count: busy, not a percentage.
            self.progress_bar.setRange(0, 0)
