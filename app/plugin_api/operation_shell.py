"""What every analysis window shares, whichever kind it is (todo O-01).

The data operations (DataOperationDialogBase: a chart's series) and the table
analyses (TableOperationDialogBase: a table's columns) differ in what they
read; around that, they are the same window and behave the same way. What
they both did, each in its own words, lives here once:

* the Results pane: a report shown as markup, plain text escaped
  (show_results), and a failure said there in red, not only in the log
  (show_failure);
* the Apply in the project's history (record_in_history): what was read,
  written and reported, with the dialog and the version that did it;
* the "Draw on" choices both Charts tabs offer, worded once;
* the calculation in the background (run_job): the buttons off at once, Stop
  and a progress bar once it has taken a moment, Stop polled by the
  calculation (stop_requested), "Stopped" or the failure said at its end;
* the window's memory (remember_window, restore_window): its size, its
  choices and its parameters, kept in user.json between sessions.

A base derives from it beside QDialog and says where its results pane is
(results_widget). No Qt object of its own: a mixin.
"""
from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app import APP_VERSION
from app.analysis import Stopped
from app.logs.logger import applogger
from app.utils.background import BackgroundTask, run_in_background
from app.utils.config import get_constant
from app.utils.i18n import _
from app.widgets.html_results import looks_like_html


class OperationShell:
    """The parts of an analysis window both kinds share."""

    #: The places a result can be drawn, the same words in both Charts tabs.
    DEST_SAME_AXIS: str = "same_axis"
    DEST_NEW_AXIS: str = "new_axis"
    DEST_NEW_FIGURE: str = "new_figure"

    @classmethod
    def destination_label(cls, value: str) -> str:
        """The Charts tab's wording of a "Draw on" choice."""
        return {cls.DEST_SAME_AXIS: _("Current axis"), cls.DEST_NEW_AXIS: _("New axis in the current figure"),
                cls.DEST_NEW_FIGURE: _("New figure")}[value]

    def destination_combo_for(self, parent: Any, values: Sequence[str], default: str, on_change: Any) -> Any:
        """The Charts tab's "Draw on" list: *values* (DEST_*), *default* selected, as wide as its words."""
        from PySide6.QtWidgets import QComboBox

        from app.plugin_api.data_operations.parameter_form import FIELD_MAX_WIDTH

        combo = QComboBox(parent)
        for value in values:
            combo.addItem(self.destination_label(value), value)
        combo.setCurrentIndex(max(0, combo.findData(default)))
        combo.setToolTip(_("Where the result is drawn: on the axis of its data, on a new axis beside it, or in "
                           "a figure of its own."))
        combo.setMaximumWidth(FIELD_MAX_WIDTH)
        combo.currentIndexChanged.connect(lambda _index: on_change())
        return combo

    # ------------------------------------------------------------------
    # The Results pane
    # ------------------------------------------------------------------

    def results_widget(self) -> Any:
        """The base's results pane (an HtmlResultsView)."""
        raise NotImplementedError

    def show_results(self, content: str) -> None:
        """Show a report in the Results pane: markup rendered, plain text kept as it is written."""
        widget = self.results_widget()
        text = str(content or "")
        if looks_like_html(text):
            widget.setHtml(text)
        elif hasattr(widget, "setContent"):
            widget.setContent(text)
        else:
            widget.setText(text)

    def show_failure(self, error: BaseException | str, *, headline: str = "") -> None:
        """Say in the Results pane why nothing came of the run: the log alone left the pane unchanged."""
        message = str(error) or type(error).__name__
        self.results_widget().setHtml(
            "<p style='color:#a40000'><b>" + html.escape(headline or _("Nothing was drawn.")) + "</b></p>"
            + "<p>" + html.escape(message) + "</p>")

    # ------------------------------------------------------------------
    # The project's history
    # ------------------------------------------------------------------

    def record_in_history(
        self,
        *,
        operation: str,
        parameters: Mapping[str, Any],
        entries: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        report: str,
    ) -> None:
        """An Apply in the project's ``__operations__``. Logged, never raised: the record must not lose the result."""
        try:
            self._repo.record_operation(  # type: ignore[attr-defined]
                operation=operation,
                dialog=f"{type(self).__module__.rsplit('.', 1)[-1]}:{type(self).__name__}",
                parameters=dict(parameters),
                entries=dict(entries),
                sources=[dict(source) for source in sources],
                results=[dict(item) for item in results],
                app_version=str(APP_VERSION),
                report=report,
            )
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Could not record %s in the project's history", operation)

    # ------------------------------------------------------------------
    # The calculation in the background
    # ------------------------------------------------------------------

    #: How long a calculation runs before Stop and the progress bar appear: most end sooner.
    BUSY_DELAY_MS: int = get_constant("background_busy_delay_ms", 300)
    #: The Results pane's headline when the calculation fails.
    FAILURE_HEADLINE: str = "Nothing was drawn."

    def busy_widgets(self) -> list[Any]:
        """What is switched off while a calculation runs: whatever would start a second one."""
        return [self.preview_button, self.apply_button]  # type: ignore[attr-defined]

    @property
    def job_running(self) -> bool:
        job: BackgroundTask | None = getattr(self, "_job", None)
        return job is not None and job.running

    def run_job(self, work: Callable[[Any], Any], on_done: Callable[[Any], None], *,
                progress: Any = None) -> bool:
        """Run *work(cancel_event)* in the background and hand its outcome to *on_done*, on the GUI thread.

        False when a calculation is running already. *progress*: a [done, total] the work updates, shown
        on the bar. *on_done* is not called when the run is stopped or fails: the reason is shown instead.
        """
        if self.job_running:
            return False
        self._job_progress = progress
        self._set_job_running(True)

        def finished(outcome: Any) -> None:
            # Stopped, but the work ended before it looked: nothing is written all the same.
            stopped = task.cancel_event.is_set()
            self._set_job_running(False)
            if stopped:
                self._job_failed(Stopped())
                return
            try:
                on_done(outcome)
            except Exception as exc:  # noqa: BLE001 - shown like a failed run
                self._job_failed(exc)

        task = run_in_background(work, finished, self._job_failed)
        self._job = task
        return True

    def stop(self) -> None:
        """Ask the calculation to stop at its next step (and the window to close: nothing is written)."""
        job: BackgroundTask | None = getattr(self, "_job", None)
        if job is not None and job.running:
            job.cancel()

    def stop_requested(self) -> bool:
        """True once Stop was pressed (or the window closed) during the calculation; poll it in long loops."""
        job: BackgroundTask | None = getattr(self, "_job", None)
        return job is not None and job.cancel_event.is_set()

    def _job_failed(self, error: BaseException) -> None:
        self._set_job_running(False)
        if isinstance(error, Stopped):
            self.show_results(_("Stopped. Nothing was changed."))
            return
        applogger.error("%s failed: %s", getattr(self, "Name", type(self).__name__), error)
        self.show_failure(error, headline=_(self.FAILURE_HEADLINE))

    def _set_job_running(self, running: bool) -> None:
        # The buttons go off at once - a second click must not start a second run - but Stop and the
        # bar wait: most runs end before they would have been read.
        from PySide6.QtCore import QTimer

        for widget in self.busy_widgets():
            widget.setEnabled(not running)
        timer = getattr(self, "_busy_timer", None)
        if timer is None:
            timer = self._busy_timer = QTimer(self)  # type: ignore[arg-type]
            timer.setInterval(100)
            timer.timeout.connect(self._show_progress)
        if running:
            self._busy_since = 0
            timer.start()
            return
        timer.stop()
        self.stop_button.setVisible(False)  # type: ignore[attr-defined]
        self.progress_bar.setVisible(False)  # type: ignore[attr-defined]

    def _show_progress(self) -> None:
        """Every 100 ms of a run: Stop and the bar once it is slow, then the count."""
        self._busy_since = getattr(self, "_busy_since", 0) + 100
        if self._busy_since < self.BUSY_DELAY_MS:
            return
        bar = self.progress_bar  # type: ignore[attr-defined]
        if not self.stop_button.isVisible():  # type: ignore[attr-defined]
            self.stop_button.setVisible(True)  # type: ignore[attr-defined]
            bar.setVisible(True)
            self.show_results(_("Working... press Stop to end it early."))
        progress = getattr(self, "_job_progress", None)
        done, total = (progress[0], progress[1]) if progress else (0, 0)
        if total > 1:
            bar.setRange(0, total)
            bar.setValue(done)
        else:
            bar.setRange(0, 0)  # one series, or work that does not count: busy, not a percentage

    # ------------------------------------------------------------------
    # The window's memory
    # ------------------------------------------------------------------

    #: Beside the entries in the stored state; "@" is in no attribute's name, so restoring skips them.
    PARAMETERS_STATE: str = "@parameters"

    def state_key(self) -> str:
        """Where this window's choices are kept in user.json: its class name."""
        return type(self).__name__

    def geometry_key(self) -> str:
        """Where its size is kept: the state key, unless the base keeps sizes per layout."""
        return self.state_key()

    def state_extra(self) -> dict[str, Any]:
        """Further values a base keeps beside the entries and the parameters (a table source's columns)."""
        return {}

    def restore_state_extra(self, stored: Mapping[str, Any]) -> None:
        """Put back what state_extra kept."""

    def parameter_forms(self) -> list[Any]:
        """The ParameterForms whose values are kept: the base's."""
        return []

    def remember_window(self) -> None:
        """Keep the entries, the parameters and the size - from every exit path, when the choice is final."""
        from app.utils.dialog_state import save_dialog_state, save_window_geometry

        extra: dict[str, Any] = dict(self.state_extra())
        forms = self.parameter_forms()
        if forms:
            values: dict[str, Any] = {}
            for form in forms:
                values.update(form.values())
            extra[self.PARAMETERS_STATE] = values
        try:
            save_dialog_state(self, self.state_key(), extra)
        except TypeError:
            # A parameter that does not go into JSON: the entries all the same.
            applogger.exception("Could not remember %s's parameters", self.state_key())
            save_dialog_state(self, self.state_key())
        save_window_geometry(self, self.geometry_key())

    def restore_window(self) -> None:
        """Put back the size, the entries and the parameters remembered; a stale value is logged, not raised."""
        from app.utils.dialog_state import restore_dialog_state, restore_window_geometry, stored_dialog_state

        restore_window_geometry(self, self.geometry_key())
        restore_dialog_state(self, self.state_key())
        stored = stored_dialog_state(self.state_key())
        parameters = stored.get(self.PARAMETERS_STATE)
        if isinstance(parameters, dict):
            for form in self.parameter_forms():
                try:
                    form.set_values(parameters)
                except Exception:  # noqa: BLE001 - a stale value must not stop the window opening
                    applogger.exception("Could not restore %s's parameters", self.state_key())
        self.restore_state_extra(stored)
