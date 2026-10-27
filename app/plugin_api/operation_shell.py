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
  choices and its parameters, kept in user.json between sessions;
* the window itself (build_window): Settings and Results side by side, and
  the one row of buttons - Preview, Stop and its bar, Copy, OK, Cancel;
* the signals the main window listens to (applied, preview_drawn).

It is the QDialog both bases derive from. A base hands build_window its
Settings and its Results pane, and says where that pane is (results_widget).
"""
from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app import APP_VERSION
from app.analysis import Stopped
from app.logs.logger import applogger
from app.plugin_api.data_operations.parameter_form import FIELD_MAX_WIDTH, ParameterForm
from app.plugin_api.data_operations.parameter_spec import Param
from app.styles.style import (
    CardFrame,
    TitledCard,
    apply_dialog_shell,
    configure_combo_width,
    create_action_button,
    stdSizeAndlayout,
)
from app.utils.background import BackgroundTask, run_in_background
from app.utils.config import get_constant
from app.utils.i18n import _
from app.widgets.html_results import looks_like_html


class OperationShell(QDialog):
    """The analysis window both kinds are: its layout, its buttons and its behaviour."""

    #: Something was written into the project (a Preview, an OK, a rollback): the window refreshes.
    applied = Signal()
    #: Preview drew a new figure (its id), or took it away again (-1): the window shows or drops its tab.
    preview_drawn = Signal(int)

    #: The places a result can be drawn, the same words in both Charts tabs.
    DEST_SAME_AXIS: str = "same_axis"
    DEST_NEW_AXIS: str = "new_axis"
    DEST_NEW_FIGURE: str = "new_figure"
    #: No chart at all: for an analysis whose result is its report (Statistics).
    DEST_NONE: str = "none"

    @classmethod
    def destination_label(cls, value: str) -> str:
        """The Charts tab's wording of a "Draw on" choice."""
        return {cls.DEST_SAME_AXIS: _("Current axis"), cls.DEST_NEW_AXIS: _("New axis in the current figure"),
                cls.DEST_NEW_FIGURE: _("New figure"), cls.DEST_NONE: _("None")}[value]

    def destination_combo_for(self, parent: Any, values: Sequence[str], default: str, on_change: Any) -> Any:
        """The Charts tab's "Draw on" list: *values* (DEST_*), *default* selected, as wide as its words."""
        combo = QComboBox(parent)
        for value in values:
            combo.addItem(self.destination_label(value), value)
        combo.setCurrentIndex(max(0, combo.findData(default)))
        combo.setToolTip(_("Where the result is drawn: on the axis of its data, on a new axis beside it, or in "
                           "a figure of its own."))
        # As narrow as the panel needs (its words elided, whole in the list that drops down), never wider
        # than a field: at its words' width it pushed the Charts page past a narrow window's edge.
        configure_combo_width(combo, follow_tooltip=False)
        combo.setMaximumWidth(FIELD_MAX_WIDTH)
        combo.currentIndexChanged.connect(lambda _index: on_change())
        return combo

    # ------------------------------------------------------------------
    # Where the data comes from
    # ------------------------------------------------------------------

    def build_source_choice(self, parent: QWidget, *, from_chart: bool, on_change: Callable[[bool], Any]) -> QHBoxLayout:
        """"Data from: (o) Table ( ) Chart" - every analysis reads either; *from_chart* checked at first.

        Sets source_table_radio and source_chart_radio; *on_change(from_chart)* follows the choice.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Data from:"), parent))
        self.source_table_radio = QRadioButton(_("Table"), parent)
        self.source_table_radio.setToolTip(_("A table's columns."))
        self.source_chart_radio = QRadioButton(_("Chart"), parent)
        self.source_chart_radio.setToolTip(_("The series a chart draws."))
        group = QButtonGroup(parent)
        for radio in (self.source_table_radio, self.source_chart_radio):
            group.addButton(radio)
            row.addWidget(radio)
        row.addStretch(1)
        (self.source_chart_radio if from_chart else self.source_table_radio).setChecked(True)
        self.source_chart_radio.toggled.connect(on_change)
        return row

    def reads_chart(self) -> bool:
        """Whether the data comes from a chart (Data from: Chart)."""
        radio = getattr(self, "source_chart_radio", None)
        return radio is not None and radio.isChecked()

    # ------------------------------------------------------------------
    # The Charts tab
    # ------------------------------------------------------------------

    #: The Charts tab's form: the parameters declared ``tab="charts"`` - a check box per chart the analysis
    #: can draw, and how it is drawn. None when the analysis declares none.
    chart_form: ParameterForm | None = None

    def build_charts_page(
        self,
        chart_params: Sequence[Param],
        *,
        destinations: Sequence[str] = (),
        default_destination: str = "",
        on_change: Callable[[], Any],
        context: Callable[[], Mapping[str, Any]] | None = None,
        on_destination_change: Callable[[], Any] | None = None,
    ) -> QWidget | None:
        """The Charts tab: "Draw on" first (*destinations*), then the charts (*chart_params*); None if neither.

        A chart a choice elsewhere makes impossible is hidden by its own ``visible_for``, read against
        *context* (the response's type, the model...): the tab lists only what can be drawn.
        """
        if not (destinations or chart_params):
            return None
        panel = CardFrame(self, "operationChartsCard")
        layout = panel.layout()
        if destinations:
            # The charts a destination makes possible (none for "None") follow it.
            self.destination_combo = self.destination_combo_for(
                panel, destinations, default_destination or destinations[0],
                lambda: (self.chart_form is not None and self.chart_form.refresh_visibility(),
                         (on_destination_change or on_change)()))
        if chart_params:
            self.chart_form = ParameterForm(chart_params, self, on_change=on_change, context=context)
            layout.addWidget(self.chart_form.widget)
        if destinations:
            # In the charts' own form when there are some: their boxes under the combo, in its column.
            if self.chart_form is not None:
                self.chart_form.layout.insertRow(0, _("Draw on:"), self.destination_combo)
            else:
                row = QFormLayout()
                stdSizeAndlayout(row)
                row.addRow(_("Draw on:"), self.destination_combo)
                layout.insertLayout(0, row)
        layout.addStretch(1)
        return panel

    def chart_value(self, name: str, default: Any = None) -> Any:
        """The Charts tab's *name* (a ``tab="charts"`` parameter); *default* when there is no such parameter."""
        if self.chart_form is None or name not in self.chart_form.values():
            return default
        return self.chart_form.values()[name]

    def chart_enabled(self, name: str) -> bool:
        """Whether the chart *name* is ticked and can be drawn (not hidden); True for one not declared."""
        if self.chart_form is None or name not in self.chart_form.values():
            return True
        return bool(self.chart_form.values()[name]) and not self.chart_form.field(name).isHidden()

    # ------------------------------------------------------------------
    # The window
    # ------------------------------------------------------------------

    def build_window(
        self,
        settings: QWidget,
        results: QWidget,
        *,
        copy: Callable[[], Any],
        revert: Callable[[], Any] | None = None,
        extra_buttons: Callable[[QHBoxLayout], None] | None = None,
        size: QSize | None = None,
        results_minimum_width: int = 260,
    ) -> None:
        """Lay the window out: *settings* and *results* side by side, each titled, the buttons under them.

        Preview, Revert (when there is *revert*), the operation's own (*extra_buttons*), then Stop and its
        bar - shown only while a calculation runs - and, on the right, Copy (*copy*), OK and Cancel.
        """
        # The tabs in a titled frame like Results beside it: the tab bar starts
        # at the same height inside the white box instead of above it (B-14).
        # The pages' own cards are flattened inside it by the sheets (#operationSettingsCard).
        left = TitledCard(self, _("Settings"), "operationSettingsCard", margins=(6, 6, 6, 6))
        left.layout().setStretchFactor(left.card, 1)
        left.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.card.layout().addWidget(settings, 1)

        right = TitledCard(self, _("Results"), "operationResultsCard")
        right.layout().setStretchFactor(right.card, 1)
        right.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right.card.layout().addWidget(results, 1)
        right.setMinimumWidth(results_minimum_width)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        row = QHBoxLayout()
        stdSizeAndlayout(row)
        self.preview_button = create_action_button(parent=self, action_id="preview", action=self.preview, layout=row)
        self.revert_button = None
        if revert is not None:
            # Back to the defaults, beside Preview: it acts on the entries only.
            self.revert_button = create_action_button(parent=self, action_id="operation_revert", action=revert,
                                                      layout=row)
        if extra_buttons is not None:
            # The operation's own, on the left: they act on the dialog's state, unlike OK and Cancel.
            extra_buttons(row)
        # After every button that stays and before the stretch, which takes up the
        # difference: beside Preview they pushed the others aside each time they
        # appeared and went (the Fit dialog's flicker). Shown only while a
        # calculation runs (_set_job_running): every analysis has the same Stop.
        self.stop_button = create_action_button(parent=self, action_id="operation_stop", action=self.stop, layout=row)
        self.stop_button.setVisible(False)
        # Busy (no percentage) for one piece of work; a count for several. See _show_progress.
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setVisible(False)
        row.addWidget(self.progress_bar)
        row.addStretch(1)
        self.copy_html_button = create_action_button(parent=self, action_id="copy", action=copy, layout=row)
        self.apply_button = create_action_button(parent=self, action_id="apply", action=self.ok, layout=row)
        self.close_button = create_action_button(parent=self, action_id="close", action=self.reject, layout=row)

        root = QVBoxLayout(self)
        # One helper owns dialog padding and size for every dialog in the app.
        apply_dialog_shell(self, root, size=size)
        root.addWidget(splitter, 1)
        root.addLayout(row, 0)

    def preview(self) -> Any:
        """Compute and draw, to be looked at: the base's."""
        raise NotImplementedError

    def ok(self) -> None:
        """Compute if needed, write for good and close: the base's."""
        raise NotImplementedError

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
        return [self.preview_button, self.apply_button]

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
        """Every 100 ms of a run: Stop and the bar once it is slow, then the count."""
        self._busy_since = getattr(self, "_busy_since", 0) + 100
        if self._busy_since < self.BUSY_DELAY_MS:
            return
        bar = self.progress_bar
        if not self.stop_button.isVisible():
            self.stop_button.setVisible(True)
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

    # ------------------------------------------------------------------
    # The model, and leaving
    # ------------------------------------------------------------------

    def current_model(self, default: str | None = None) -> str:
        """The selected model's key, or *default* while none is selected.

        The item's data when it has any - a combo that shows translated names keeps the untranslated one
        there, and that is what visible_for rules and the code compare against - otherwise its text. With
        no *default*, the first of MODELS; "" for a window without a model.
        """
        combo = getattr(self, "model_combo", None)
        if default is None:
            default = next(iter(getattr(self, "MODELS", {}) or {}), "")
        if combo is None:
            return default
        data = combo.currentData()
        if data is not None and str(data):
            return str(data)
        return combo.currentText() or default

    def discard_unapplied(self) -> None:
        """Take back what a Preview wrote and OK never kept: the base's own way."""

    def _leave(self) -> None:
        self.stop()
        self.remember_window()
        self.discard_unapplied()

    def reject(self) -> None:
        """Cancel or Close: the calculation stopped, the choices kept, the Preview taken back."""
        self._leave()
        super().reject()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        """The window's close button: as Cancel."""
        self._leave()
        super().closeEvent(event)
