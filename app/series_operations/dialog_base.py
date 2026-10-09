"""Shared base dialog for chart-series operation panels.

The base class owns the common shell used by smoothing, interpolation, outlier
removal and fitting dialogs:

- left side: tabs with the Axis/Series selector, the model selector and the parameters
- right side: results pane and log output
- bottom action buttons

Subclasses provide operation-specific controls by overriding the builder hooks.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import html
import json
import re
from typing import Any, ClassVar, Protocol

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QProgressBar, QRadioButton, QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget
import numpy as np
import pandas as pd

from scipy.interpolate import griddata

from app import APP_VERSION
from app.analysis import Stopped
from app.analysis.sampling import sampling_frequency
from app.charts.grids import pivot_to_grid
from app.data.data_source import parse_roles, row_value, resolve_role_column
from app.data.repo.operations import OPERATIONS_TABLE
from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.widgets.axis_series_selector import AxisSeriesSelector
from app.widgets.table_source_selector import TableSourceSelector
from app.styles.style import (
    apply_dialog_shell,
    create_doc_link,
    icon_from_svg_source,
    set_doc_link,
    MARGIN_TOOLBOX_PAGE,
    CardFrame,
    create_action_button,
    create_section_title,
    stdSizeAndlayout,
)
from app.logs.logger import applogger
from app.utils.coercion import coerce_axis, to_numbers, to_numeric_axis
from app.utils.series_validation import (
    SeriesIssue,
    clean_xy,
    errors,
    validate_xy,
)
from app.series_operations.parameter_form import ParameterForm
from app.series_operations.results import OperationResult
from app.series_operations.parameter_spec import ChoiceParam, Param, defaults
from app.utils.messages import show_message
from app.utils.dialog_state import (
    dialog_entries,
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
    set_dialog_entries,
    stored_dialog_state,
)
from app.widgets.html_results import HtmlResultsView, looks_like_html, plain_to_html
from app.utils.background import BackgroundTask, run_in_background
from app.utils.i18n import _

_TABLE_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")

# Every table an operation writes starts with this.  One character, and the
# source list can hide the whole class of them: a project with six fits and a
# spectral analysis has more generated tables than imported ones, and the
# imported ones are what the user is looking for.
GENERATED_TABLE_PREFIX: str = "_"


def generated_table_name(raw: str, *, fallback: str = "Result") -> str:
    """Return a safe generated-table name, prefixed and never bare.

    Callers pass whatever they assembled - an axis id, a series name, a model -
    and get back something SQLite accepts and the source list can recognise.
    """
    safe = _TABLE_SAFE_RE.sub("_", str(raw).strip()).strip("_") or fallback
    return f"{GENERATED_TABLE_PREFIX}{safe}"


@dataclass(frozen=True, slots=True, kw_only=True)
class OperationModel:
    """One model an operation offers, as the Model combo lists it.

    The base needs only where its documentation is. An operation that needs
    more per model - which parameters it shows, which function it calls -
    subclasses this and adds the fields (FilterModel, smoothing's ModelSpec),
    so everything about one model sits in one entry of the MODELS dict
    rather than in a tuple of names beside a dict of links beside a set of
    the models that need an extra row.
    """

    doc_title: str = ""
    doc_url: str = ""
    #: Models of different groups are separated by a line in the combo.
    group: str = ""

    @property
    def doc(self) -> tuple[str, str]:
        """(title, url) for the Docs link."""
        return self.doc_title, self.doc_url


@dataclass(slots=True)
class ResultSeriesSpec:
    """Repository descriptor for one generated/updated chart series."""
    name: str
    sql_query: str
    roles: Mapping[str, Any]
    style: Mapping[str, Any]

class OperationJob(Protocol):
    """What prepare_job returns: an operation's inputs, read off the window.

    ``run`` is the calculation alone - no widget, no repository - so it can
    run on a worker thread. It polls *should_stop* where it can and raises
    app.analysis.Stopped when told to.
    """

    def run(self, should_stop: Callable[[], bool] | None = None) -> Any: ...


@dataclass(slots=True)
class CallJob:
    """A job that is one call: for an operation that computes all its series together.

    ``compute`` is prepared on the GUI thread with everything it needs
    already read, and must touch neither a widget nor the repository.
    """

    compute: Callable[[], Any]

    def run(self, should_stop: Callable[[], bool] | None = None) -> Any:
        if should_stop is not None and should_stop():
            raise Stopped()
        return self.compute()


@dataclass(slots=True)
class SeriesOutcome:
    """What a SeriesJob hands back: one outcome per series that worked, and why the others did not."""

    outcomes: list[tuple[str, Any]]
    errors: list[str]


@dataclass(slots=True)
class SeriesJob:
    """The usual job: every selected series read on the GUI thread, each computed on the worker.

    ``inputs`` holds (series name, data) pairs, read off the repository by
    ``read_series``; ``compute`` is the dialog's ``compute_series``, which
    must touch neither a widget nor the repository. ``progress`` is written
    here and read by the dialog's timer - ``[done, total]`` - which is how
    the bar moves without a signal crossing threads for every series.
    """

    inputs: list[tuple[str, Any]]
    settings: Any
    compute: Callable[[str, Any, Any], Any]
    errors: list[str] = field(default_factory=list)
    progress: list[int] = field(default_factory=lambda: [0, 0])

    def run(self, should_stop: Callable[[], bool] | None = None) -> SeriesOutcome:
        total = len(self.inputs)
        self.progress[:] = [0, total]
        outcomes: list[tuple[str, Any]] = []
        errors = list(self.errors)
        for done, (name, data) in enumerate(self.inputs):
            if should_stop is not None and should_stop():
                raise Stopped()
            try:
                outcomes.append((name, self.compute(name, data, self.settings)))
            except Stopped:
                raise
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")
            self.progress[:] = [done + 1, total]
        return SeriesOutcome(outcomes, errors)


class SeriesOperationDialogBase(QDialog):
    """Abstract shell for all chart-series operation dialogs.

    Subclasses should:
    - create model controls in ``build_model_selector``;
    - create parameter controls in ``build_parameter_selector``;
    - optionally override ``build_results_pane`` when a QLabel is not enough;
    - implement ``compute_results``;
    - provide generated-series metadata through the apply hooks below;
    - connect operation-specific signals in ``connect_operation_signals``.
    """
    #: How the operation presents itself.  All three live on the class rather
    #: than in config.json because an operation is a plugin: dropping one .py
    #: file into ``app/series_operations`` is meant to be the whole install,
    #: and presentation that had to be registered elsewhere made that untrue.
    #: The scanner reads them straight out of the source, so none of it costs
    #: an import.
    Name: str = ""
    Description: str = ""

    #: Shown in the results pane whenever a control changes, until Preview or
    #: OK actually computes something. Controls no longer trigger compute_results()
    #: as a side effect of being edited - only Preview and OK (through
    #: _run_operation) do, so a heavy operation never runs on an intermediate
    #: value while a spin box is being dragged or typed into.
    PENDING_RESULTS_MESSAGE: str = "Press Preview or OK to see the result."

    #: The operation's icon, as an SVG document.  Inline rather than a path to
    #: one: a file beside the module is a second thing to copy and a second
    #: thing to lose.  ``style.icon_from_svg_source`` renders it; an operation
    #: that leaves this empty gets no icon and is otherwise unaffected.

    Icon = """
    <rect x="5" y="5" width="14" height="14" rx="3"/>
    <path d="M8.5 12h7"/>
    <path d="M12 8.5v7"/>
    """

    applied = Signal()

    #: Emitted with the run's report, as HTML, once Apply has committed.
    #: The main window forwards it to the chart panel the operation ran on, so
    #: the numbers end up beside the chart instead of disappearing with the
    #: dialog that produced them.
    results_published = Signal(str)

    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        title: str="Series Operation",
        parent: QWidget | None = None,
        width: int = 900,
        height: int = 640
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(title)
        self.resize(width, height)

        # The window icon is the operation's own artwork, set once here rather
        # than by each subclass.  Two of them had got it wrong exactly because
        # it was copied: the spectral dialog asked for "fit", and the
        # statistics dialog for "statistics", which is not a file - so that
        # window simply had no icon.  An id that has to match a filename is a
        # thing to get wrong; ``Icon`` cannot be.
        if self.Icon:
            self.setWindowIcon(icon_from_svg_source(self.Icon, size=32))

        self._repo = repo
        self._figure_id = int(figure_id)
        self._original_grid: tuple[int, int] | None = self._capture_current_grid()

        self._results_label = HtmlResultsView(self)
        self._results_view = self._results_label
        self._preview_axis_ids: set[int] = set()
        self._preview_table_names: set[str] = set()
        self._preview_active = False
        #: True once Apply has written the results; discard_result_target and
        #: the dialogs' own cleanup leave an applied result alone. Kept when a
        #: subclass set it before calling this constructor.
        self._applied: bool = getattr(self, "_applied", False)
        # Source series name -> was its x a timestamp column? Filled by
        # series_xy as each series is read, and read back by
        # restore_temporal_x when the result is written, so a result lands
        # on the same axis its source is drawn on. Keyed by name because
        # that is the one thing a result carries back from the series it
        # came from.
        self._temporal_x_sources: dict[str, bool] = {}
        self.series_selector = AxisSeriesSelector(self._repo, self._figure_id, self)
        # Every operation is opened on one figure and keeps it: create_result_
        # axis adds to self._figure_id, so a combo that could change figures
        # would show one figure's axes while writing to another's.
        self.series_selector.set_figure_locked(self.LOCK_FIGURE_SELECTION)
        self.series_selector.set_series_visible(self.SHOWS_SERIES_SELECTOR)
        # The Axis / Series page is a tab of the left panel. It must be
        # vertically expanding; otherwise its internal series_list can expand
        # only inside the selector's fixed size and the empty space remains in
        # the page below it. Model and Parameters already behave this way
        # because they are not forced to QSizePolicy.Fixed here.
        self.series_selector.setMinimumHeight(0)
        self.series_selector.setMaximumHeight(16777215)
        self.series_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # Shared model combo used by operation-specific model selectors.
        self.model_combo = QComboBox(self)

        # Subclasses create widgets here before build_* hooks run. A
        # subclass that builds its own form sets this in build_parameter_selector.
        self._parameter_form: QFormLayout | None = None
        self.init_operation_widgets()

        self._model_selector_widget = self.build_model_selector()
        self._parameter_selector_widget = self.build_parameter_selector()
        self._results_widget = self.build_results_pane()

        self._build_common_ui()
        self.connect_common_signals()
        self.connect_operation_signals()

        # Every subclass gets remembered entries and geometry for free; the
        # storage key is the class name, so two dialogs never share a slot.
        self._state_key = type(self).__name__
        # What Revert goes back to: the entries as built, before any memory.
        self._default_entries = dialog_entries(self, inputs_only=True)
        self._default_parameters = self.parameter_values()
        restore_window_geometry(self, self._state_key)
        restore_dialog_state(self, self._state_key)
        self._restore_extra_state()

    #: Where the declared parameters and the table's columns are stored beside
    #: the entries; "@" is in no attribute's name, so restoring skips them.
    _PARAMETERS_STATE = "@parameters"
    _TABLE_SOURCE_STATE = "@table_source"

    def _remember_state(self) -> None:
        """Persist entries, parameters and geometry to user.json.

        Called from every exit path - Apply, Close, the window button - because
        the useful moment to record a choice is when the user leaves, not when
        they make it.
        """
        extra: dict[str, Any] = {}
        if getattr(self, "_parameter_form_spec", None) is not None:
            extra[self._PARAMETERS_STATE] = self.parameter_values()
        source = getattr(self, "table_source", None)
        if source is not None and source.table():
            extra[self._TABLE_SOURCE_STATE] = {
                "table": source.table(), "x": source.x_column(), "y": source.y_columns(),
            }
        try:
            save_dialog_state(self, self._state_key, extra)
        except TypeError:
            # A parameter that does not go into JSON: the entries all the same.
            applogger.exception("Could not remember %s's parameters", self._state_key)
            save_dialog_state(self, self._state_key)
        save_window_geometry(self, self._state_key)

    def _restore_extra_state(self) -> None:
        """The parameters and the table's columns stored beside the entries."""
        stored = stored_dialog_state(self._state_key)
        form = getattr(self, "_parameter_form_spec", None)
        parameters = stored.get(self._PARAMETERS_STATE)
        if form is not None and isinstance(parameters, dict):
            try:
                form.set_values(parameters)
            except Exception:  # noqa: BLE001 - a stale value must not stop the window opening
                applogger.exception("Could not restore %s's parameters", self._state_key)
        source = getattr(self, "table_source", None)
        chosen = stored.get(self._TABLE_SOURCE_STATE)
        if source is not None and isinstance(chosen, dict):
            source.set_spec(str(chosen.get("table", "")), str(chosen.get("x", "")), chosen.get("y") or [])
        # Restored with its signals blocked: the page follows the radio here.
        if getattr(self, "_source_stack", None) is not None:
            self._source_stack.setCurrentIndex(1 if self.reads_table() else 0)

    def revert_entries(self) -> None:
        """Every entry back to its default, as JMP's Revert: what the window had when first opened."""
        set_dialog_entries(self, self._default_entries, quietly=False)
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.set_values(self._default_parameters)
        if getattr(self, "_source_stack", None) is not None:
            self._source_stack.setCurrentIndex(1 if self.reads_table() else 0)
        self._refresh_visibility()
        self.mark_results_stale()


    def _series_display_name(self, row: Any) -> str:
        return str(row["name"])

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------


    def set_row_visible(self, field_widget: QWidget, visible: bool, *, form: QFormLayout | None = None) -> None:
        """Show or hide a form row together with its label.

        Hiding only the field leaves its label behind, which reads as a bug.
        Two dialogs had their own copy of this; the ``form`` argument defaults
        to ``self._parameter_form``, which is the one every subclass uses.
        """
        layout = form if form is not None else getattr(self, "_parameter_form", None)
        if layout is None:
            return

        field_widget.setVisible(visible)
        label = layout.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)

    def set_doc_link(self, title: str, url: str) -> None:
        """Point this dialog's documentation label at a method's docs."""
        label = getattr(self, "_doc_link", None)
        if label is not None:
            set_doc_link(label, title, url)

    # ------------------------------------------------------------------
    # Builder hooks
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        """Create subclass widgets before build_model_selector/build_parameter_selector.

        The base constructor calls the builder hooks, so subclasses must create
        controls used by those hooks here rather than after super().__init__().
        """
        return None

    #: The models this operation offers, in the order the Model combo lists
    #: them: name -> OperationModel. Setting it gives the operation the
    #: standard selector - "Model:" with the combo, "Docs:" with a link kept
    #: on the selected model - without writing build_model_selector.
    MODELS: ClassVar[Mapping[str, OperationModel]] = {}
    #: The combo's tooltip, when MODELS builds it.
    MODEL_TOOLTIP: ClassVar[str] = "Choose the model."
    #: The combo's label, when MODELS builds it ("Operation:", "Method:"...).
    MODEL_LABEL: ClassVar[str] = "Model:"

    def build_model_selector(self) -> QWidget:
        """The Model combo and its Docs link, from MODELS.

        Without MODELS, just the shared model_combo, for an operation that
        fills it itself.
        """
        if not self.MODELS:
            panel = CardFrame(self, "operationModelCard")
            layout = panel.layout()
            stdSizeAndlayout(self.model_combo)
            layout.addWidget(self.model_combo)
            return panel

        if getattr(self, "_doc_link", None) is None:
            self._doc_link = create_doc_link(self)
        panel = QWidget(self)
        form = QFormLayout(panel)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # Connected before the items go in: the first addItems selects item 0
        # and emits, which is what puts the first model's link in place.
        self.model_combo.currentIndexChanged.connect(self._update_doc_link)
        self.model_combo.addItems(list(self.MODELS))
        # A line between groups, inserted from the end so earlier indices hold.
        groups = [model.group for model in self.MODELS.values()]
        for index in range(len(groups) - 1, 0, -1):
            if groups[index] != groups[index - 1]:
                self.model_combo.insertSeparator(index)
        self.model_combo.setToolTip(_(self.MODEL_TOOLTIP))
        form.addRow(_(self.MODEL_LABEL), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)
        return panel

    def model_spec(self) -> OperationModel | None:
        """The OperationModel of the selected model, when MODELS has it."""
        return self.MODELS.get(self.current_model())

    def _update_doc_link(self, *_ignored: Any) -> None:
        spec = self.model_spec()
        if spec is not None:
            self.set_doc_link(*spec.doc)

    
    #: Whether the figure combo is disabled. True for every operation so far;
    #: an operation that genuinely means to move between figures would have to
    #: keep self._figure_id in step with it, which none currently does.
    LOCK_FIGURE_SELECTION: bool = True

    #: Whether the series list is shown. An operation that generates a series
    #: rather than transforming one has nothing to select, and a picker it
    #: never reads is worse than no picker.
    SHOWS_SERIES_SELECTOR: bool = True

    #: Whether the Axis / Series page appears in the toolbox at all. The
    #: selector itself is still built and still answers selected_axis_id(),
    #: because _run_operation needs an axis to write to - this only decides
    #: whether the user is shown a page they have no decision to make on.
    SHOWS_AXIS_SERIES_PAGE: bool = True

    #: Whether the data can come from a table's columns instead of a chart's
    #: series: "Data from: Chart | Table" on the Axis / Series page. The
    #: columns are drawn as a figure of their own first (see
    #: _prepare_table_source), so the operation runs on series as always.
    READS_TABLES: bool = True

    #: Declared parameters.  An operation that sets this gets its parameter
    #: form built, wired and read back for free; see ``parameter_spec``.
    #: Leaving it empty keeps the old behaviour, where the subclass overrides
    #: ``build_parameter_selector`` and builds the form by hand - which is
    #: still the right answer for a genuinely unusual control.
    PARAMS: tuple[Param, ...] = ()

    def build_parameter_selector(self) -> QWidget:
        """Return the parameter editor widget for the left panel.

        Builds itself from ``PARAMS`` when the operation declares any.  An
        operation that declares none must override this - hence the empty
        widget rather than NotImplementedError, which is what the dialogs
        written before ``PARAMS`` existed rely on.
        """
        if not self.PARAMS:
            return QWidget(self)

        self._parameter_form_spec = ParameterForm(
            self.PARAMS,
            self,
            on_change=self.mark_results_stale,
            context=self.parameter_context,
        )
        # Kept under the name the base already uses for the hand-built form, so
        # set_row_visible and the state helpers keep working unchanged.
        self._parameter_form = self._parameter_form_spec.layout
        return self._parameter_form_spec.widget

    def parameter_context(self) -> Mapping[str, Any]:
        """Return values a ``visible_for`` rule may reference but not own.

        The model combo by default, under the name ``model``: nearly every
        operation's parameter visibility depends on which model is selected,
        and the combo belongs to this base rather than to ``PARAMS``.
        """
        combo = getattr(self, "model_combo", None)
        if combo is None:
            return {}
        return {"model": self.current_model()}

    def parameter_values(self) -> dict[str, Any]:
        """Return the declared parameters' current values, keyed by name.

        Falls back to the declared defaults when there is no form yet, so an
        operation can call this from ``compute_results`` during construction
        without a special case.
        """
        form = getattr(self, "_parameter_form_spec", None)
        if form is None:
            return defaults(self.PARAMS)
        return form.values()

    def build_results_pane(self) -> QWidget:
        """Return the shared HTML results widget."""
        return self._results_view

    def build_extra_action_buttons(self, layout: QHBoxLayout) -> None:
        """Add operation-specific buttons to the bottom row, after Preview.

        Only one operation needs this so far - fitting, where optimising the
        parameters is a separate act from drawing them - but putting the button
        in the shared row is what keeps it in the same place as every other
        dialog's buttons.
        """
        del layout

    def connect_common_signals(self) -> None:
        """Wire signals common to selector-based operation dialogs."""
        selection_changed = getattr(self.series_selector, "selection_changed", None)
        if selection_changed is not None:
            selection_changed.connect(lambda *_args: self.mark_results_stale())
            selection_changed.connect(lambda *_args: self._update_origin_label())
        self._update_origin_label()
    
    def connect_operation_signals(self) -> None:
        """Connect the model selector; subclasses add their own widgets.

        A different model shows different parameters and makes the results
        on screen stale - true of every operation, so wired here once. An
        override that connects more calls super() first.
        """
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.mark_results_stale)

    def _refresh_visibility(self, *_ignored: Any) -> None:
        """Show the parameters the selected model uses: the PARAMS' visible_for rules.

        Eight operations had this same body as their own; one that builds
        its controls by hand overrides it (or calls this, then adds its own).
        """
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()

    # -- For operations that build their own controls ----------------------

    def model_form_card(self, object_name: str, rows: Sequence[tuple[str, QWidget]]) -> CardFrame:
        """A card with the model choices as a form, and the Docs link last.

        For an operation with more than the one model combo build_model_selector
        lays out (Cluster's family and algorithm, Smoothing's data type and
        method).
        """
        panel = CardFrame(self, object_name)
        form_widget = QWidget(panel)
        form = QFormLayout(form_widget)
        stdSizeAndlayout(form)
        for label, widget in rows:
            form.addRow(label, widget)
        if getattr(self, "_doc_link", None) is None:
            self._doc_link = create_doc_link(self)
        form.addRow(_("Docs:"), self._doc_link)
        panel.layout().addWidget(form_widget)
        return panel

    def add_field_row(
        self, form: QFormLayout, key: str, label: str, widget: QWidget, tooltip: str = ""
    ) -> None:
        """Add one parameter row that show_field_rows can show or hide by *key*."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        stdSizeAndlayout(row_layout)
        row_layout.addWidget(widget)
        row_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label_widget = QLabel(label)
        if tooltip:
            widget.setToolTip(tooltip)
            label_widget.setToolTip(tooltip)
        form.addRow(label_widget, row_widget)
        rows: dict[str, tuple[QWidget, QWidget]] = self.__dict__.setdefault("_field_rows", {})
        rows[key] = (label_widget, row_widget)

    def show_field_rows(self, visible: set[str] | frozenset[str]) -> None:
        """Show the rows add_field_row made whose key is in *visible*, hide the rest."""
        for key, (label_widget, row_widget) in getattr(self, "_field_rows", {}).items():
            label_widget.setVisible(key in visible)
            row_widget.setVisible(key in visible)

    # -- A result on an axis or figure of its own -------------------------

    def result_figure_name(self, results: Sequence[Any], what: str = "", source: str = "") -> str:
        """"<source> - <what was done>": what names a figure an operation makes.

        "Calculus 1" in a tab bar says nothing a week later; the series and
        the operation do. *source* defaults to the first result's source_name.
        """
        if not source and results:
            source = str(getattr(results[0], "source_name", "") or "")
        return f"{source} - {what or self.operation_label}".strip(" -") or self.operation_label

    def label_result_axis(self, *, title: str, x_label: str = "x", y_label: str = "y") -> None:
        """Title and label the axis this operation made for its results, if it made one."""
        axis_id = getattr(self, "_result_axis_id", None)
        if axis_id is None:
            return
        try:
            self._repo.update_axis_descriptor(axis_id=axis_id, title=title, x_label=x_label, y_label=y_label)
        except (DatabaseError, ValueError):
            applogger.exception(f"Failed to label the {self.operation_label} result axis")

    def current_model(self, default: str | None = None) -> str:
        """The selected model's key, or *default* while none is selected.

        The item's data when it has any - a combo that shows translated names
        keeps the untranslated one there, and that is what visible_for rules
        and the code compare against - otherwise its text. With no *default*,
        the first of MODELS: the one the combo starts on.
        """
        data = self.model_combo.currentData()
        if data is not None and str(data):
            return str(data)
        if default is None:
            default = next(iter(self.MODELS), "")
        return self.model_combo.currentText() or default

    def current_axis_name(self) -> str:
        """The name of the axis picked in the Data page."""
        return self.series_selector.selected_axis_name()

    # -- Small controls, for operations that build their own forms ---------

    @staticmethod
    def int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        """A spin box over [minimum, maximum], set to *value*."""
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def float_spin(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        """A decimal spin box, stepping by its last shown decimal (0.001 at most)."""
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setDecimals(decimals)
        widget.setSingleStep(10 ** -min(decimals, 3))
        return widget

    # -- Sampling rate: the "derive it from x, or type it" pair -----------

    def create_sampling_rate_widgets(self, check_text: str = "") -> None:
        """Create ``_fs_auto_check`` and ``_fs_spin``; call from init_operation_widgets."""
        self._fs_auto_check = QCheckBox(check_text, self)
        self._fs_spin = QDoubleSpinBox(self)

    def add_sampling_rate_rows(self, form: QFormLayout) -> None:
        """Set up the sampling-rate pair and add it to *form* as two rows."""
        self._fs_auto_check.setChecked(True)
        self._fs_auto_check.setToolTip(
            _("Take the sampling frequency from the spacing of the x role.")
        )
        form.addRow(_("Sampling rate:"), self._fs_auto_check)
        self._fs_spin.setRange(1e-9, 1e12)
        self._fs_spin.setDecimals(6)
        self._fs_spin.setValue(1.0)
        self._fs_spin.setToolTip(_("Samples per unit of x, used when not derived."))
        form.addRow(_("fs:"), self._fs_spin)

    def sampling_rate(self, x_values: np.ndarray, name: str) -> float:
        """fs in samples per unit of x: the typed one, or read off the x role."""
        if not self._fs_auto_check.isChecked():
            return float(self._fs_spin.value())
        fs, note = sampling_frequency(x_values)
        if note:
            applogger.warning(
                "Series '%s': %s.", name, note,
            )
        return fs

    # ------------------------------------------------------------------
    # Common layout
    # ------------------------------------------------------------------
    def _toolbox_page(self, content: QWidget) -> QWidget:
        """Wrap one left-panel widget as a tab page.

        The properties inspector's arrangement: the page is the grey ground
        (marked toolboxPage and inset in _build_common_ui), and the
        content floats on it inside one white card - the System Settings
        grouped-box look. Most operation panels are already
        ``create_card_widget`` cards; the ones that are not (the Axis / Series
        selector) get wrapped in one here so every page reads the same.
        """
        page = QWidget(self)
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        page_layout = QVBoxLayout(page)
        # Left null so apply_toolbox_page_metrics fills it with the page inset.
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        if content.property("card"):
            card: QWidget = content
        else:
            card = CardFrame(page, "operationToolboxCard")
            card.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            card_layout = card.layout()
            # Stretch: the Axis / Series selector carries an expanding series
            # list that has to take the height the card is given.
            card_layout.addWidget(content, 1)

        page_layout.addWidget(card, 1)
        return page

    def _build_common_ui(self) -> None:
        """Build the common shell: the inputs as tabs on the left, the results on the right."""
        # Tabs, as Chart properties: the accordion's three stacked headers took
        # height and showed one section at a time all the same.
        left_tabs = QTabWidget(self)
        left_tabs.setObjectName("operationTabs")
        left_tabs.setDocumentMode(True)
        left_tabs.setUsesScrollButtons(True)
        left_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        left_tabs.tabBar().setExpanding(False)
        left_tabs.setMinimumWidth(320)
        left_tabs.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        self.axis_series_panel = self._build_source_panel()
        self.model_panel = self._model_selector_widget
        self.parameters_panel = self._parameter_selector_widget

        # Each page fills its tab: every panel may expand vertically.
        for panel in (
            self.axis_series_panel,
            self.model_panel,
            self.parameters_panel,
        ):
            panel.setMinimumHeight(0)
            panel.setMaximumHeight(16777215)
            panel.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )

        # The Axis / Series tab is left out for an operation that selects nothing.
        pages = []
        if self.SHOWS_AXIS_SERIES_PAGE:
            pages.append((self.axis_series_panel, _("Data")))
        pages.append((self.model_panel, _("Model")))
        pages.append((self.parameters_panel, _("Parameters")))
        for content, title in pages:
            page = self._toolbox_page(content)
            # The ground and the inset a toolbox page had (grey on macOS,
            # white on Windows; see apply_toolbox_page_metrics).
            page.setProperty("toolboxPage", True)
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            page.layout().setContentsMargins(*MARGIN_TOOLBOX_PAGE)
            left_tabs.addTab(page, title)
        left_tabs.setCurrentIndex(0)
        self.left_tabs = left_tabs

        right = CardFrame(self, "operationResultsCard")
        right_layout = right.layout()

        results_title = create_section_title(_("Results"), right)
        right_layout.addWidget(results_title, 0)
        right_layout.addWidget(self._results_widget, 3)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        right.setMinimumWidth(380)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)

        self.preview_button = create_action_button(
                                  parent=self,
                                  action_id="preview",
                                  action=self.preview,
                                  layout=action_row,
                              )

        # Shown only while a calculation runs in the background: every
        # operation gets the same Stop, in the same place. See evaluate().
        self.stop_button = create_action_button(
                               parent=self,
                               action_id="operation_stop",
                               action=self.stop_evaluation,
                               layout=action_row,
                           )
        self.stop_button.setVisible(False)
        # Beside Stop and shown with it. Busy (no percentage) for one series;
        # a series at a time for several. See _show_progress.
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setVisible(False)
        action_row.addWidget(self.progress_bar)

        # Back to the defaults, beside Preview: it acts on the entries only.
        self.revert_button = create_action_button(
                                 parent=self,
                                 action_id="operation_revert",
                                 action=self.revert_entries,
                                 layout=action_row,
                             )

        # Operation-specific buttons go next to Preview, on the left: they act
        # on the dialog's own state, unlike Apply/Close which end it.
        self.build_extra_action_buttons(action_row)

        action_row.addStretch(1)

        self.copy_html_button = create_action_button(
                                    parent=self,
                                    action_id="copy",
                                    action=self._results_label.copy_html_to_clipboard,
                                    layout=action_row,
                                )

        self.apply_button = create_action_button(
                                parent=self,
                                action_id="apply",
                                action=self.ok,
                                layout=action_row,
                            )

        self.close_button = create_action_button(
                                parent=self,
                                action_id="close",
                                action=self.reject,
                                layout=action_row,
                            )

        root = QVBoxLayout(self)
        # One helper owns dialog padding and size for every dialog in the app.
        apply_dialog_shell(self, root, size=None)
        root.addWidget(splitter, 1)
        root.addLayout(action_row, 0)


    # ------------------------------------------------------------------
    # Data from a chart, or from a table
    # ------------------------------------------------------------------

    def _build_source_panel(self) -> QWidget:
        """The Data page: the chart's series, or a table's columns."""
        self.table_source: TableSourceSelector | None = None
        #: The figure the table's columns were drawn on, and what they were.
        self._table_figure_id: int | None = None
        self._table_figure_spec: tuple[str, str, tuple[str, ...]] | None = None
        #: The figure the dialog was opened on, to go back to.
        self._chart_figure_id = self._figure_id
        if not (self.READS_TABLES and self.SHOWS_SERIES_SELECTOR and self.SHOWS_AXIS_SERIES_PAGE):
            return self.series_selector

        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Data from:"), panel))
        self.source_chart_radio = QRadioButton(_("Chart"), panel)
        self.source_chart_radio.setToolTip(_("The series the chart draws."))
        self.source_table_radio = QRadioButton(_("Table"), panel)
        self.source_table_radio.setToolTip(_("A table's columns, without drawing them first."))
        group = QButtonGroup(panel)
        group.addButton(self.source_chart_radio)
        group.addButton(self.source_table_radio)
        self.source_chart_radio.setChecked(True)
        row.addWidget(self.source_chart_radio)
        row.addWidget(self.source_table_radio)
        row.addStretch(1)
        layout.addLayout(row)

        self.table_source = TableSourceSelector(self._repo, panel)
        self._source_stack = QStackedWidget(panel)
        self._source_stack.addWidget(self.series_selector)
        self._source_stack.addWidget(self.table_source)
        layout.addWidget(self._source_stack, 1)

        self.source_table_radio.toggled.connect(self._on_source_changed)
        self.table_source.changed.connect(self._on_table_columns_changed)
        return panel

    def reads_table(self) -> bool:
        """Whether this run takes its data from a table's columns."""
        return getattr(self, "table_source", None) is not None and self.source_table_radio.isChecked()

    def _on_source_changed(self, *_ignored: Any) -> None:
        self._source_stack.setCurrentIndex(1 if self.reads_table() else 0)
        if not self.reads_table() and self._table_figure_id is not None:
            # Back to the chart: whatever was drawn from the table goes.
            self.cancel_operation_changes(refresh=False)
            self._discard_table_figure()
            self._refresh_after_preview_state_change()
        self._on_table_columns_changed()

    def _on_table_columns_changed(self, *_ignored: Any) -> None:
        """Draw the chosen columns at once, not at Preview.

        Everything an operation reads before Preview - its own buttons (Fit's
        Estimate and Fit), the lists it fills from the selected series'
        columns - reads the series the dialog is on. Drawn only at Preview,
        those were still the chart's series, not the table's.
        """
        self.mark_results_stale()
        if not self.reads_table():
            return
        try:
            self._prepare_table_source()
        except ValueError:
            return  # no Y yet: drawn when there is one
        except Exception:
            applogger.exception("Could not draw the table's columns")
            return
        self._refresh_after_preview_state_change()

    def _use_figure(self, figure_id: int) -> None:
        """Run on *figure_id* from now on: its axes, its series, its grid."""
        self._figure_id = int(figure_id)
        self._original_grid = self._capture_current_grid()
        # Reloading the selector announces a new selection, and whatever
        # listens may ask for the selected series: not a reason to draw the
        # table again while this is the drawing (or the discarding) of it.
        self._switching_figure = True
        try:
            self.series_selector._load_figures()
            self.series_selector.set_figure_id(int(figure_id), select_all_series=True)
        finally:
            self._switching_figure = False

    def _prepare_table_source(self) -> None:
        """Draw the chosen columns as a figure of their own, once, and run on it.

        Before the preview's savepoint, which is rolled back by the next
        Preview: the figure stays until the window closes without Apply, or
        the columns change.
        """
        spec = self.table_source.spec() if self.table_source is not None else None
        if spec is None:
            raise ValueError(_("Choose a table and at least one Y column."))
        if spec == self._table_figure_spec and self._table_figure_id is not None:
            if self._repo.get_figure_descriptor(int(self._table_figure_id)) is not None:
                return
        self.cancel_operation_changes(refresh=False)
        self.discard_result_target()
        self._discard_table_figure()

        table, x, ys = spec
        x_label = x or _("Row")
        figure_id = int(self._repo.create_figure_descriptor(
            name=f"{table}: {', '.join(ys)}", nrows=1, ncols=1,
        ))
        axis_id = int(self._repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot", title=table,
            x_label=x_label, y_label=ys[0] if len(ys) == 1 else "", options={"grid": True},
        ))
        for index, y in enumerate(ys):
            self._repo.create_series_descriptor(
                axis_id=axis_id, series_index=index, name=y,
                sql_query=TableSourceSelector.series_sql(table, x, y), roles={"x": "x", "y": "y"},
            )
        applogger.info("%s: drew %s from table %s as figure %s.", self.operation_label, ", ".join(ys), table, figure_id)
        self._table_figure_id, self._table_figure_spec = figure_id, spec
        self._use_figure(figure_id)

    def _discard_table_figure(self) -> None:
        """Remove the figure drawn from a table, unless Apply kept it; back to the chart."""
        figure_id = getattr(self, "_table_figure_id", None)
        self._table_figure_id, self._table_figure_spec = None, None
        if figure_id is None:
            return
        if not getattr(self, "_applied", False):
            try:
                self._repo.delete_figure(int(figure_id))
            except Exception:
                applogger.exception("Failed to discard the figure drawn from a table")
        self._use_figure(self._chart_figure_id)

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

        if hasattr(self._results_label, "setContent"):
            self._results_label.setContent(content)
            return

        if looks_like_html(content):
            self._results_label.setHtml(content)
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
        """Human readable operation name used in messages/logs."""
        return self.windowTitle() or self.__class__.__name__

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        """Style key/value pairs identifying series generated by this dialog.

        Example::

            {"generated_smoothing": True, "smoothing_dialog": "series_smoothing"}

        ``remove_previous_generated_series`` deletes only descriptors whose
        ``style_json`` contains all these key/value pairs.

        Derived from the module name by default - smoothing_dialog.py gives
        the example above. Four operations predate the rule and keep their
        own values, which are stored in existing projects.
        """
        stem = type(self).__module__.rsplit(".", 1)[-1].removesuffix("_dialog")
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
    BUSY_DELAY_MS: ClassVar[int] = 300

    # -- Per series: the common case ------------------------------------

    def series_settings(self) -> Any:
        """Everything compute_series needs besides the data, read on the GUI thread."""
        return self.parameter_values()

    def read_series(self, row: Any, name: str) -> Any:
        """One selected series' data, read on the GUI thread; x and y by default."""
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
        return type(self).compute_series is not SeriesOperationDialogBase.compute_series

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

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    #: What this operation needs of its input.  Subclasses override the ones
    #: that matter to them; the defaults are the weakest useful set, so an
    #: operation that says nothing still gets the empty/length/NaN checks
    #: without having its input rejected for a shape it handles fine.
    #:
    #: See ``app/utils/series_validation.py`` for what each one catches.
    INPUT_MINIMUM_POINTS: int = 2
    INPUT_REQUIRES_SORTED_X: bool = False
    INPUT_REQUIRES_UNIQUE_X: bool = False
    INPUT_REQUIRES_UNIFORM_X: bool = False
    INPUT_REQUIRES_VARYING_Y: bool = False

    def validate_input_xy(
        self,
        x: Any,
        y: Any,
        *,
        label: str = "",
        raise_on_error: bool = True,
        repairable: bool = False,
    ) -> list[SeriesIssue]:
        """Check one series against this operation's requirements.

        Warnings are logged and the caller continues; errors are logged and,
        by default, raised - so an operation that does not check the return
        value still cannot run on data that would give a wrong answer.

        Pass ``raise_on_error=False`` to collect problems across several series
        and report them together, which is what the multi-series dialogs want:
        one bad series should not hide the six good ones.
        """
        issues = validate_xy(
            x,
            y,
            minimum_points=self.INPUT_MINIMUM_POINTS,
            require_sorted_x=self.INPUT_REQUIRES_SORTED_X,
            require_unique_x=self.INPUT_REQUIRES_UNIQUE_X,
            require_uniform_x=self.INPUT_REQUIRES_UNIFORM_X,
            require_varying_y=self.INPUT_REQUIRES_VARYING_Y,
            repairable=repairable,
            label=label,
        )

        for issue in issues:
            if issue.severity == "warning":
                applogger.warning(
                    issue.message
                )

        blocking = errors(issues)
        if blocking and raise_on_error:
            # One message, not one per issue: a dialog that pops three boxes
            # for one bad series trains the user to dismiss them unread.
            applogger.error(
                " ".join(issue.message for issue in blocking),
                raise_error=True,
            )

        return issues

    def prepare_input_xy(
        self,
        x: Any,
        y: Any,
        *,
        label: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate, then return the series cleaned to this operation's needs.

        The repair follows the declared requirements: an operation that needs
        sorted x gets sorted x, one that needs unique x gets duplicates
        averaged, and every operation gets non-finite points dropped.  Anything
        changed is reported, because quietly discarding a user's points is its
        own kind of wrong answer.
        """
        # repairable=True: clean_xy below fixes exactly what the declared
        # requirements ask for, so a problem it is about to solve is a warning
        # rather than a reason to stop.
        self.validate_input_xy(x, y, label=label, repairable=True)

        x_clean, y_clean, report = clean_xy(
            x,
            y,
            sort_x=self.INPUT_REQUIRES_SORTED_X,
            merge_duplicate_x=self.INPUT_REQUIRES_UNIQUE_X,
        )

        if report.changed:
            prefix = f"{label}: " if label else ""
            applogger.warning(
                f"{prefix}{report.describe()}.",
            )

        return x_clean, y_clean

    def series_xy(self, row: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Run one selected series' SQL and return its x/y, cleaned.

        Byte-identical copies of this lived in the calculus, control-chart
        and peaks dialogs, and a fourth was about to be written for the root
        finder - see todo.txt P2-14. Every operation needs the same three
        steps and makes the same two decisions, so both live here:

        *Which columns.* The series' own roles first, since that is what the
        chart draws. A role naming a column the query no longer returns
        falls back to the first two numeric columns rather than raising: the
        query is edited far more often than the roles are, and the numbers
        are usually still there under different names.

        *What to do with the rows.* :meth:`prepare_input_xy`, which repairs
        exactly what this operation's INPUT_* flags declare it needs and
        reports whatever it changed.
        """
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            raise ValueError("the series has no SQL query")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            raise ValueError("the series query returned no rows")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [
            str(column)
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
        ]

        x_col = resolve_role_column(columns, roles, "x") or ""
        y_col = resolve_role_column(columns, roles, "y") or ""
        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0]
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else x_col

        return self.prepare_input_xy(
            self.numeric_x(frame[x_col], name),
            self.numeric_y(frame[y_col]),
            label=name,
        )

    def numeric_x(self, column: Any, name: str = "") -> np.ndarray:
        """Return an x column as floats, timestamps included, and remember.

        Not ``pd.to_numeric``: a timestamp column run through that becomes
        all-NaN, and the operation then reports "0 usable points" about a
        series of sixty perfectly good ones. Every operation but the
        outlier one did that to every dated series -
        ``app/utils/coercion`` was written for exactly this and only that
        one dialog was calling it.

        Whether the column *was* temporal is remembered against the series
        name, so :meth:`restore_temporal_x` can put the result back on the
        axis it came from rather than at x = 1 700 000 000.
        """
        if name:
            _coerced, is_temporal = coerce_axis(column)
            self._temporal_x_sources[name] = bool(is_temporal)
        return to_numeric_axis(column)

    def series_origin(self, row: Any) -> str:
        """Return a short, human caption for what shape one series' data is.

        Purely informational: no operation decides what it can compute from
        this string, only the roles it reads directly.  It exists so a user
        opening, say, the peaks or roots dialog on a series with a ``z`` role
        is told - before pressing anything - whether they are about to search
        a 2D curve, a 3D grid, an interpolated 3D surface, or a vector field.

        Deciding "on a grid" vs. "scattered" needs the actual data (a SQL
        query), which is why this is relatively expensive and is meant to be
        called only when the selection changes - see
        ``connect_common_signals`` - not on every redraw.
        """
        roles = parse_roles(row_value(row, "roles", default={}))
        has_z = bool(roles.get("z"))
        has_u = bool(roles.get("u"))
        has_v = bool(roles.get("v"))

        if has_u and has_v:
            # Informational only: nothing in this task actually computes
            # curl/divergence yet, this just tells the user what they see.
            return _("vector field (x, y, u, v)")

        if not has_z:
            return _("2D (x, y)")

        try:
            name = self._series_display_name(row)
            x, y, z = self.series_xyz(row, name)
            frame = pd.DataFrame({"x": x, "y": y, "z": z})
            grid = pivot_to_grid(frame)
        except Exception:
            # Any failure to read/pivot the data still leaves a true fact
            # standing: there is a z role, so this is 3D of some kind. Which
            # kind is a detail the label can afford to get wrong here - the
            # operation itself will raise a clear error if it matters.
            return _("3D scattered (x, y, z)")

        return _("3D on a grid (x, y, z)") if grid is not None else _("3D scattered (x, y, z)")

    def _update_origin_label(self) -> None:
        """Show ``series_origin`` for the current selection, if there is one.

        Wired next to ``mark_results_stale`` in ``connect_common_signals``:
        the same event (selection changed) that invalidates a preview is the
        one point where the caption needs recomputing.
        """
        selector = getattr(self, "series_selector", None)
        if selector is None or not hasattr(selector, "set_origin_text"):
            return
        rows = self.selected_series()
        if not rows:
            selector.set_origin_text("")
            return
        try:
            selector.set_origin_text(self.series_origin(rows[0]))
        except Exception:
            applogger.exception("Failed to describe the selected series' data shape.")
            selector.set_origin_text("")

    def series_xyz(self, row: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one selected series' SQL and return its x/y/z, as parallel arrays.

        The scattered (not-yet-gridded) reading of a 3D series: one point per
        row, same idea as smoothing_dialog's ``_series_choice_from_row`` but
        centralised here so every 3D-aware operation reads its data the same
        way. x and y go through ``numeric_x``/``numeric_y`` (timestamp-aware,
        same as the 2D case); z is a computed/measured quantity, never a
        chart axis a result has to land back on, so it goes through
        ``to_numeric_axis`` directly.

        Raises a clear error when the series has no ``z`` role, or when that
        role no longer names a real column - callers are expected to have
        already decided (via ``series_origin`` or the roles themselves) that
        this series is 3D before calling this, not to find out from the
        exception.
        """
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            raise ValueError("the series has no SQL query")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            raise ValueError("the series query returned no rows")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        z_col = str(roles.get("z") or "")
        if not z_col or z_col not in columns:
            raise ValueError("the series has no z role")

        numeric = [
            str(column)
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        x_col = resolve_role_column(columns, roles, "x") or ""
        y_col = resolve_role_column(columns, roles, "y") or ""
        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0]
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else x_col

        x_values = np.asarray(self.numeric_x(frame[x_col], name), dtype=float).reshape(-1)
        y_values = np.asarray(self.numeric_y(frame[y_col]), dtype=float).reshape(-1)
        z_values = np.asarray(to_numeric_axis(frame[z_col]), dtype=float).reshape(-1)
        return x_values, y_values, z_values

    def series_grid_xyz(
        self,
        row: Any,
        name: str,
        *,
        resolution: int = 120,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return ``(X, Y, Z, interpolated)`` grids for one 3D series.

        Preference order, and why it is this order:

        1. **Exact.** ``pivot_to_grid`` (``app/charts/grids.py``) recognises a
           complete Cartesian product of the distinct x and y values and
           returns it untouched - no smoothing, no invented values. This is
           always tried first and always preferred, because it is the only
           answer that is exactly what was measured.
        2. **Honest interpolation.** When the points are not a complete grid
           (which is the common case for anything sampled freehand, or
           jittered), a regular ``resolution`` x ``resolution`` grid is built
           over the data's own x/y span and filled with
           ``scipy.interpolate.griddata(..., method="linear")``. Linear
           interpolation inside the convex hull of the original points is a
           defensible estimate; *outside* it, ``griddata`` correctly refuses
           to guess and leaves ``NaN`` - which this function deliberately
           does not paper over. A caller that averages, contours or searches
           this grid has to skip those cells rather than be handed a number
           that was invented to fill a hole no data ever supported.

        Never a third option: there is no "reasonable default" for a cell
        with no support and no neighbours close enough to interpolate from -
        that would be fabricated precision, which is worse than leaving it
        as ``NaN`` and saying so via the returned ``interpolated`` flag.
        """
        x, y, z = self.series_xyz(row, name)
        frame = pd.DataFrame({"x": x, "y": y, "z": z})

        grid = pivot_to_grid(frame)
        if grid is not None:
            x_grid, y_grid, z_grid = grid
            return x_grid, y_grid, z_grid, False

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x_finite, y_finite, z_finite = x[finite], y[finite], z[finite]
        if x_finite.size < 4 or np.unique(x_finite).size < 2 or np.unique(y_finite).size < 2:
            raise ValueError("not enough points to interpolate a surface")

        x_lin = np.linspace(float(np.min(x_finite)), float(np.max(x_finite)), int(resolution))
        y_lin = np.linspace(float(np.min(y_finite)), float(np.max(y_finite)), int(resolution))
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)

        z_grid = griddata(
            (x_finite, y_finite),
            z_finite,
            (x_grid, y_grid),
            method="linear",
        )
        z_grid = np.asarray(z_grid, dtype=float)
        if not np.any(np.isfinite(z_grid)):
            raise ValueError("not enough points to interpolate a surface")

        return x_grid, y_grid, z_grid, True

    def numeric_y(self, column: Any) -> np.ndarray:
        """Return a y column as floats, timestamps included.

        Nothing to remember: a result's y is a computed quantity - a
        derivative, a residual, a count - and is written as the number it
        is even when the source was a duration or a time of day.
        """
        return to_numeric_axis(column)

    def result_to_frame(self, result: Any) -> Any:
        """Return the pandas DataFrame to save for one result: its to_df().

        ``to_frame()`` is still accepted from a user-written result that
        predates OperationResult.
        """
        for name in ("to_df", "to_frame"):
            method = getattr(result, name, None)
            if callable(method):
                return method()
        raise NotImplementedError(f"{type(result).__name__} has no to_df()")

    def result_series_spec(self, axis_id: int, table_name: str, result: Any) -> ResultSeriesSpec:
        """Build the chart-series descriptor for one saved result table."""
        raise NotImplementedError

    def result_series_specs(
        self,
        axis_id: int,
        table_name: str,
        result: Any,
    ) -> Sequence[ResultSeriesSpec]:
        """Return every series one result should draw.

        One by default, which is what almost every operation wants and what
        ``result_series_spec`` already answers. A control chart is the
        exception: its result table carries the plotted values, the centre
        line and the control limits, and drawing only the first of those
        leaves the chart without the lines that make it a control chart.

        Overriding this rather than calling create_series_descriptor directly
        keeps preview and apply on the same path - a spec listed here is
        previewed, cleaned up and applied exactly like any other.
        """
        return [self.result_series_spec(axis_id, table_name, result)]

    #: Result tables are named ``<prefix>_axis<id>_<series>[_<variant>]``,
    #: where the variant is this attribute of the result (its model, its
    #: method...) or nothing when None. An operation sets the two instead of
    #: overriding result_table_name; the names are kept as they always were,
    #: since a re-run replaces the table of the same name.
    RESULT_TABLE_PREFIX: str = ""
    RESULT_TABLE_VARIANT: str | None = "model"

    def result_table_name(self, axis_id: int, result: Any) -> str:
        """Return a safe table name for one result.

        From RESULT_TABLE_PREFIX/RESULT_TABLE_VARIANT when the operation sets
        a prefix; otherwise the class name and whatever model or method the
        result carries.
        """
        prefix = self.RESULT_TABLE_PREFIX
        if prefix:
            raw = f"{prefix}_axis{axis_id}_{result.source_name}"
            if self.RESULT_TABLE_VARIANT is not None:
                raw += f"_{getattr(result, self.RESULT_TABLE_VARIANT)}"
            return generated_table_name(raw, fallback=f"{prefix}_Result")
        source_name = str(getattr(result, "source_name", "Series") or "Series")
        # "model" on most results, "method" on the clustering one: a result
        # without either used to make Apply fail outright.
        variant = getattr(result, "model", None) or getattr(result, "method", None) or "result"
        raw = f"{self.__class__.__name__}_axis{axis_id}_{source_name}_{variant}"
        return generated_table_name(raw, fallback="Series_Result")

    def format_results(self, results: Sequence[Any]) -> str:
        """Return text for the results pane after apply.

        Subclasses may override to reuse their richer preview formatting.
        """
        return f"{len(results)} result(s)"


    def store_cached_results(self, results: Sequence[Any]) -> None:
        """Store latest results using the existing dialog convention."""
        if hasattr(self, "_last_results"):
            setattr(self, "_last_results", list(results))

    def write_result_table(self, table_name: str, result: Any) -> None:
        """Persist one operation result to a normal SQLite table."""
        frame = self.restore_temporal_x(self.result_to_frame(result), result)
        self._repo.import_dataframe(frame, table_name=table_name, normalize_columns=False)

    def restore_temporal_x(self, frame: Any, result: Any) -> Any:
        """Write a result's x back in the units its source was read in.

        An operation does its arithmetic on seconds since the epoch, because
        subtracting two timestamps is what every one of them needs. Writing
        that back would put the result on the chart at x = 1 700 000 000
        while the series it came from is drawn as dates - the right answer,
        plotted somewhere unreadable.

        So a result computed from a dated series gets dated x values back.
        The renderers read them the same way they read the source (see
        ``coercion.coerce_axis``), which is what puts the two on one axis.
        """
        source = str(getattr(result, "source_name", "") or "")
        if not self._temporal_x_sources.get(source):
            return frame

        for column in ("x", "left_x", "right_x"):
            if column not in getattr(frame, "columns", []):
                continue
            try:
                frame[column] = pd.to_datetime(
                    to_numbers(frame[column]),
                    unit="s",
                    errors="coerce",
                )
            except (TypeError, ValueError):
                applogger.warning(
                    "Could not write %s back as timestamps; leaving it numeric.",
                    column,
                )
        return frame

    def remove_previous_generated_series(self, axis_id: int) -> int:
        """Delete generated descriptors for this operation on the selected axis."""
        removed = 0
        expected = dict(self.generated_style_filter)
        for row in self._repo.get_series(axis_id) or []:
            try:
                style = json.loads(str(row["style_json"] or "{}"))
            except Exception:
                style = {}
            if all(style.get(key) == value for key, value in expected.items()):
                self._repo.delete_series(int(row["id"]))
                removed += 1
        if removed:
            applogger.info(f"Removed {removed} previous {self.operation_label} generated series.")
        return removed

    def create_result_series(self, axis_id: int, table_name: str, result: Any) -> int | None:
        """Create every chart-series descriptor for a saved result table.

        Returns the first series id, for callers that only ever make one.
        """
        first_id: int | None = None
        for spec in self.result_series_specs(axis_id, table_name, result):
            series_id = self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=spec.name,
                sql_query=spec.sql_query,
                roles=dict(spec.roles),
                style=dict(spec.style),
            )
            applogger.info(f"Created series: {spec.name}")
            if first_id is None and series_id is not None:
                first_id = int(series_id)
        return first_id

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """Return the axis the results should be written to.

        The selected axis by default: a smoothed series belongs next to the
        series it smoothed.  A transform that changes the meaning of the x
        axis - a spectrum's x is frequency, not the source's x - overrides this
        to build a chart of its own, because putting frequencies on a time axis
        produces a picture that is simply wrong.
        """
        del results
        return selected_axis_id

    def create_result_axis(
        self,
        *,
        chart_type: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> int:
        """Add an axis to the current figure and return its id.

        For operations whose output cannot share the source's axes.  A spectrum
        turns a time axis into a frequency one, and a box plot of a sample has
        no x in common with the series it summarises - putting either on top of
        its input produces a picture that is simply wrong.

        A new axis rather than a new figure: the result belongs beside what it
        was computed from, in the same tab, and a chart tab per estimate would
        bury the original.  The figure's grid grows by one row to make room,
        because an axis outside the grid is never drawn.
        """
        figure_id = self._figure_id
        axis_index = self._repo.next_axis_index(figure_id)

        descriptor = self._repo.load_figure_descriptor(figure_id=figure_id)
        rows = int(getattr(descriptor, "nrows", 1) or 1)
        cols = int(getattr(descriptor, "ncols", 1) or 1)
        if axis_index >= rows * cols:
            self._repo.set_figure_grid(figure_id, nrows=axis_index + 1, ncols=cols)

        axis_id = int(
            self._repo.create_axis_descriptor(
                figure_id=figure_id,
                axis_index=axis_index,
                chart_type=chart_type,
                title=title,
                x_label=x_label,
                y_label=y_label,
                options=dict(options or {"grid": True}),
            )
        )
        applogger.info(
            "%s: created axis %s (%s) on figure %s.",
            self.operation_label,
            axis_id,
            chart_type,
            figure_id,
        )
        return axis_id

    # ------------------------------------------------------------------
    # Where results are drawn
    # ------------------------------------------------------------------

    #: The three places a generated result can go. Shared because the choice
    #: is the same wherever it appears, and two operations offering the same
    #: decision under different words would be worse than either.
    DEST_SAME_AXIS: str = "same_axis"
    DEST_NEW_AXIS: str = "new_axis"
    DEST_NEW_FIGURE: str = "new_figure"

    @classmethod
    def destination_param(
        cls,
        *,
        name: str = "destination",
        label: str = "Draw on:",
        tooltip: str = "",
        default: str | None = None,
    ) -> Param:
        """Return the declared destination choice, worded once.

        Operations differ in *why* they default one way or another - a
        derivative needs its own scale, a drawn function usually belongs
        beside the data it is being compared with - so the default and the
        tooltip are arguments, while the options themselves are not.
        """
        return ChoiceParam(
            name,
            label,
            tooltip=tooltip,
            choices=(
                ("New axis in this figure", cls.DEST_NEW_AXIS),
                ("Same axis as the source", cls.DEST_SAME_AXIS),
                ("New figure", cls.DEST_NEW_FIGURE),
            ),
            default_value=default or cls.DEST_NEW_AXIS,
        )

    def resolve_destination_axis(
        self,
        selected_axis_id: int,
        *,
        chart_type: str,
        title: str = "",
        figure_name: str = "",
        options: Mapping[str, Any] | None = None,
        parameter: str = "destination",
    ) -> int:
        """Return the axis to draw on, honouring the declared destination.

        Creates the axis or figure once and reuses it, so adjusting a
        parameter repeatedly does not leave a trail of empty axes behind. The
        ids are remembered on the dialog so ``discard_result_target`` can undo
        them if Apply never happens.
        """
        destination = str(self.parameter_values().get(parameter, self.DEST_NEW_AXIS))
        if destination == self.DEST_SAME_AXIS:
            return selected_axis_id

        if getattr(self, "_result_axis_id", None) is None:
            if destination == self.DEST_NEW_FIGURE:
                self._result_figure_id, self._result_axis_id = self.create_result_figure(
                    name=figure_name or title or self.operation_label,
                    chart_type=chart_type,
                    title=title,
                    options=options,
                )
            else:
                self._result_figure_id = None
                self._result_axis_id = self.create_result_axis(
                    chart_type=chart_type,
                    title=title,
                    options=options,
                )

        # Falls back to the selected axis when creation failed: an
        # operation that cannot make its own axis should still draw
        # somewhere rather than raise on the way to the chart.
        return (
            int(self._result_axis_id)
            if self._result_axis_id is not None
            else selected_axis_id
        )

    def _forget_rolled_back_target(self) -> None:
        """Drop the remembered result axis or figure if the rollback removed it.

        resolve_destination_axis makes the target once and reuses it. Made
        during a Preview, it is inside the preview's savepoint, and undoing
        the Preview undoes it too - the next Preview then drew on an axis
        that no longer existed ("FOREIGN KEY constraint failed").
        """
        axis_id = getattr(self, "_result_axis_id", None)
        figure_id = getattr(self, "_result_figure_id", None)
        if axis_id is None and figure_id is None:
            return
        try:
            if figure_id is not None and self._repo.get_figure_descriptor(int(figure_id)) is None:
                self._result_axis_id = self._result_figure_id = None
                return
            descriptor = self._repo.load_figure_descriptor(
                int(figure_id) if figure_id is not None else self._figure_id
            )
            axis_ids = {int(axis.id) for axis in (descriptor.axes if descriptor else [])}
        except Exception:
            applogger.exception("Could not check the result target after a rollback")
            axis_ids = set()
        if axis_id is not None and int(axis_id) not in axis_ids:
            self._result_axis_id = None
            self._result_figure_id = None

    def discard_result_target(self) -> None:
        """Remove an axis or figure this dialog made, when Apply never ran.

        A whole figure when that is what was made: deleting only its axis
        would leave an empty chart tab behind, which is worse than the axis it
        was meant to clean up.
        """
        if getattr(self, "_applied", False):
            return

        axis_id = getattr(self, "_result_axis_id", None)
        figure_id = getattr(self, "_result_figure_id", None)
        if axis_id is None and figure_id is None:
            return

        self._result_axis_id = None
        self._result_figure_id = None
        try:
            if figure_id is not None:
                self._repo.delete_figure(int(figure_id))
                applogger.info("Discarded the unapplied figure %s.", figure_id)
            elif axis_id is not None:
                self._repo.delete_axis(int(axis_id))
                applogger.info("Discarded the unapplied axis %s.", axis_id)
        except Exception:
            applogger.exception("Failed to discard the unapplied result target")

    def create_result_figure(
        self,
        *,
        name: str,
        chart_type: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> tuple[int, int]:
        """Create a figure with one axis and return ``(figure_id, axis_id)``.

        For results that do not belong in the source chart at all.  A new axis
        (``create_result_axis``) keeps the result beside its input, which is
        right when the two are meant to be compared; a new figure is right when
        they are not - a derivative kept for its own sake, or a result destined
        for a different report.

        The dialog does NOT follow the new figure: ``self._figure_id`` stays
        put, because the selector is locked to it and the axis combo would
        otherwise start listing axes of a figure the user cannot see. The main
        window picks the new figure up when it reloads its tabs.
        """
        figure_id = int(self._repo.create_figure_descriptor(name=str(name), nrows=1, ncols=1))
        axis_id = int(
            self._repo.create_axis_descriptor(
                figure_id=figure_id,
                axis_index=0,
                chart_type=chart_type,
                title=title,
                x_label=x_label,
                y_label=y_label,
                options=dict(options or {"grid": True}),
            )
        )
        applogger.info(
            "%s: created figure %s with axis %s (%s).",
            self.operation_label,
            figure_id,
            axis_id,
            chart_type,
        )
        return figure_id, axis_id

    def discard_operation_artifacts(self) -> None:
        """Undo anything the operation created outside the preview savepoint.

        Called from Close/Cancel.  The savepoint covers descriptor rows, but
        writing a result table commits (pandas' to_sql does), so a dialog that
        creates its own figure has to remove it here or an abandoned chart is
        left behind.

        By default, the axis or figure a result was drawn on when this dialog
        made it (see discard_result_target); nothing when Apply ran.
        """
        self.discard_result_target()

    def apply_results_to_axis(self, axis_id: int, results: Sequence[Any]) -> None:
        """Write every result as a table plus a chart series on *axis_id*.

        Every result, not the first: this runs inside the savepoint opened by
        ``_run_operation``, so anything raised here discards the whole batch.
        It used to end with ``optimize_db()``, whose VACUUM cannot run inside a
        transaction - the OperationalError aborted the operation part-way and
        left only the results already flushed to disk.  The caller optimizes
        after committing instead.
        """
        self.remove_previous_generated_series(axis_id)
        for result in results:
            if isinstance(result, OperationResult):
                result.apply(self, axis_id)
                continue
            # A user-written operation whose result predates OperationResult.
            table_name = self.result_table_name(axis_id, result)
            self.write_result_table(table_name, result)
            applogger.info(f"Saved result table: {table_name}")
            self.create_result_series(axis_id, table_name, result)

    @property
    def preview_style_filter(self) -> Mapping[str, Any]:
        return {**dict(self.generated_style_filter), "operation_preview": True, "preview_owner": self.__class__.__name__}

    def remove_previous_preview_series(self, axis_id: int) -> int:
        removed = 0
        expected = dict(self.preview_style_filter)
        for row in self._repo.get_series(axis_id) or []:
            try:
                style = json.loads(str(row["style_json"] or "{}"))
            except Exception:
                style = {}
            if all(style.get(key) == value for key, value in expected.items()):
                self._repo.delete_series(int(row["id"]))
                removed += 1
        return removed

    def preview_table_name(self, axis_id: int, result: Any) -> str:
        """Return the table name a preview writes to.

        Derived from the final name so the two are recognisably a pair, and
        prefixed like it: a preview is even more clearly not the user's data.
        """
        base = self.result_table_name(axis_id, result)
        return generated_table_name(f"{base}_Preview", fallback="Series_Preview")

    def _delete_preview_table(self, table_name: str) -> None:
        self._repo.delete_table(str(table_name))

    def remove_preview_tables(self, table_names: Sequence[str] | None = None) -> None:
        names = {str(name) for name in (table_names or self._preview_table_names) if str(name).strip()}
        for table_name in names:
            self._delete_preview_table(table_name)
            applogger.info(f"Deleted preview table: {table_name}")
        self._preview_table_names.difference_update(names)

    def remove_preview_artifacts(self, axis_id: int | None = None) -> None:
        if axis_id is None:
            for preview_axis_id in list(self._preview_axis_ids):
                self.remove_previous_preview_series(int(preview_axis_id))
        else:
            self.remove_previous_preview_series(int(axis_id))
        self.remove_preview_tables()

    def create_preview_series(self, axis_id: int, table_name: str, result: Any) -> int | None:
        first_id: int | None = None
        for spec in self.result_series_specs(axis_id, table_name, result):
            style = dict(spec.style)
            style.update(self.preview_style_filter)
            series_id = self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=f"Preview: {spec.name}",
                sql_query=spec.sql_query,
                roles=dict(spec.roles),
                style=style,
            )
            if first_id is None and series_id is not None:
                first_id = int(series_id)
        return first_id

    def preview_results_to_axis(self, axis_id: int, results: Sequence[Any]) -> None:
        self.remove_preview_artifacts(axis_id)
        for result in results:
            if isinstance(result, OperationResult):
                result.preview(self, axis_id)
                continue
            table_name = self.preview_table_name(axis_id, result)
            self.write_result_table(table_name, result)
            self.remember_preview_table(table_name)
            self.create_preview_series(axis_id, table_name, result)

    def remember_preview_table(self, table_name: str) -> None:
        """Note a table a preview wrote, so closing the dialog removes it."""
        self._preview_table_names.add(table_name)
        # No optimize_db() here: a preview also runs inside the savepoint, and
        # VACUUM cannot run in a transaction (see apply_results_to_axis).
    # TO IMPLEMENT!!!!
    def get_data(
        self,
        sql_query: str,
        roles: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return finite X/Y values and source rowids for Hide-aware queries.

        ``query_series_frame_for_hide`` exposes normalized ``x``/``y`` columns
        plus ``__rowid__``.  Do not read ``z`` here: the outlier dialog is a
        1D X/Y operation, and normal series frames often have no Z column.
        """
        source_df = self._repo.query_series_frame_for_hide(
            sql_query=sql_query,
            roles=roles,
        )
        # Not pd.to_numeric: a timestamp x would become all-NaN and the
        # caller would see an empty series (see utils.coercion).
        raw_x = to_numeric_axis(source_df["x"])
        raw_y = to_numeric_axis(source_df["y"])
        raw_rowids = source_df["__rowid__"].to_numpy(dtype=int)

        finite_mask = np.isfinite(raw_x) & np.isfinite(raw_y)
        return raw_x[finite_mask], raw_y[finite_mask], raw_rowids[finite_mask]


    def _refresh_after_preview_state_change(self) -> None:
        """Notify the owner window that chart/data panes must refresh."""
        try:
            self.applied.emit()
        except Exception:
            applogger.exception("Failed to emit preview refresh signal.")

    def _reset_preview_state_flags(self) -> None:
        """Clear in-memory preview bookkeeping after commit or rollback."""
        self._preview_active = False
        self._preview_axis_ids.clear()
        self._preview_table_names.clear()

    def _capture_current_grid(self) -> tuple[int, int] | None:
        """Return the figure grid as it was before this dialog previews.

        Preview-only operations can temporarily create extra axes.  Some repo
        paths update the figure grid separately from the axis descriptors, so a
        rollback/cleanup must restore both pieces of state.
        """
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id=self._figure_id)
            rows = int(getattr(descriptor, "nrows", 1) or 1)
            cols = int(getattr(descriptor, "ncols", 1) or 1)
            return max(1, rows), max(1, cols)
        except Exception:
            applogger.exception("Failed to capture original figure grid.")
            return None

    def _restore_original_grid(self) -> None:
        """Restore the figure grid captured when the dialog opened."""
        if self._original_grid is None:
            return
        rows, cols = self._original_grid
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id=self._figure_id)
            current_rows = int(getattr(descriptor, "nrows", 1) or 1)
            current_cols = int(getattr(descriptor, "ncols", 1) or 1)
            if (current_rows, current_cols) != (rows, cols):
                self._repo.set_figure_grid(self._figure_id, nrows=rows, ncols=cols)
        except Exception:
            applogger.exception("Failed to restore original figure grid.")

    # ------------------------------------------------------------------
    # Preview transaction
    # ------------------------------------------------------------------
    def begin_preview_transaction(self) -> None:
        """Open the SAVEPOINT that makes a Preview undoable.

        Why a database savepoint and not just artifact bookkeeping: removing the
        preview series and temp tables only undoes what the *generated-table*
        operations create.  Clustering writes a ClusterId column into the user's
        own source table and rewrites the selected series' SQL, and outlier
        removal flips hide flags on real rows - none of which the artifact
        cleanup can reach.  With every preview write inside one savepoint,
        Close rolls back whatever the operation did, whatever that was.

        SqliteRepo._commit() suppresses commits while the savepoint is open, so
        repository helpers that normally commit after each write stay inside it.
        """
        if self._repo.preview_savepoint_active:
            return
        self._repo.begin_preview_savepoint(
            f"preview_{self.__class__.__name__}"
        )

    def rollback_preview_transaction(self) -> bool:
        """Undo every database change made since the preview started."""
        if not self._repo.preview_savepoint_active:
            return False
        try:
            return bool(self._repo.rollback_preview_savepoint())
        except Exception:
            applogger.exception("Failed to roll back the preview savepoint")
            return False

    def commit_preview_transaction(self) -> None:
        """Make the preview changes permanent."""
        if not self._repo.preview_savepoint_active:
            return
        try:
            self._repo.release_preview_savepoint()
        except Exception:
            applogger.exception("Failed to commit the preview savepoint")

    def _finalize_successful_operation(
        self,
        *,
        results: Sequence[Any],
        verb: str,
    ) -> None:
        """Refresh UI and update the results pane after Preview/Apply."""
        self.store_cached_results(results)
        self._refresh_after_preview_state_change()
        formatted = self.format_results(results) or f"{verb}: {len(results)} result(s)."
        self.publish_results(formatted)

        # Applying is the point at which the report stops being scratch work,
        # so that is when it is handed to the chart it describes.  A preview is
        # deliberately excluded: it is undone on Close, and a note about
        # results that no longer exist is worse than no note.
        if verb == "Applied":
            self.results_published.emit(self.results_report_html(formatted, results))

    def results_report_html(self, formatted: str, results: Sequence[Any]) -> str:
        """Wrap one run's output in a titled block for the chart's notes pane.

        The heading is what makes several appended reports readable: without
        it, two runs of two different operations are one undifferentiated wall
        of text.
        """
        from datetime import datetime

        body = formatted if self.RESULTS_ARE_HTML else plain_to_html(formatted)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"<h3 style='margin:0 0 4px 0;'>{html.escape(self.operation_label)}</h3>"
            f"<p style='margin:0 0 8px 0;color:#666;font-size:9pt;'>"
            f"{stamp} - {len(results)} result(s)</p>"
            f"{body}"
        )

    def _run_operation(
        self, *, commit: bool, then: Callable[[bool], None] | None = None
    ) -> bool:
        """Run Preview or Apply through one simple pipeline.

        Preview first executes the same cleanup used by Close/Cancel, then writes
        new preview artifacts. Apply first removes any preview artifacts, then
        writes final artifacts and commits/exits through ``ok``.

        In the background (RUN_IN_BACKGROUND) the results are written once the
        calculation is back, and this returns True for "started"; *then* is
        told whether the operation went through, in either case, once it is
        over.
        """
        def finish(succeeded: bool) -> bool:
            if succeeded:
                self.operation_succeeded(commit=commit)
            if then is not None:
                then(succeeded)
            return succeeded

        try:
            if self.reads_table():
                self._prepare_table_source()
            axis_id_value = self.series_selector.selected_axis_id()
            if axis_id_value is None:
                # The catalogue holds the wording; the operation names the box.
                show_message(self, "series.no_axis_selected", title=self.operation_label)
                return finish(False)
            # Read now, not when the results are back: the window stays
            # editable while a background run computes.
            inputs = self.operation_inputs() if commit else None
            if self._runs_in_background():
                if self.evaluating:
                    return False
                started: list[bool] = []

                def deliver(results: Sequence[Any]) -> None:
                    started.append(True)
                    finish(self._deliver_results(list(results), int(axis_id_value), commit=commit, inputs=inputs))

                self.evaluate(deliver, background=True)
                if not self.evaluating and not started:
                    # Nothing to compute: the same answer the GUI thread gives.
                    show_message(self, "series.no_series_selected", title=self.operation_label)
                    return finish(False)
                return True
            results = list(self.compute_results())
        except Exception as exc:
            return finish(self._operation_failed(exc, commit=commit))
        return finish(self._deliver_results(results, int(axis_id_value), commit=commit, inputs=inputs))

    def _runs_in_background(self) -> bool:
        return bool(
            self.RUN_IN_BACKGROUND
            and SeriesOperationDialogBase.BACKGROUND_ENABLED
            and (self._computes_per_series()
                 or type(self).prepare_job is not SeriesOperationDialogBase.prepare_job)
        )

    def _operation_failed(self, exc: Exception, *, commit: bool) -> bool:
        self.cancel_operation_changes(refresh=True)
        action = "Apply" if commit else "Preview"
        if commit:
            applogger.critical(f"{action} failed: {exc}")
        else:
            applogger.error(f"{action} failed: {exc}")
        return False

    def _deliver_results(
        self,
        results: list[Any],
        axis_id_value: int,
        *,
        commit: bool,
        inputs: Mapping[str, Any] | None = None,
    ) -> bool:
        """Write computed results as a preview, or apply them; on the GUI thread.

        An Apply is also recorded in the project (``__operations__``, todo
        R-03) with the *inputs* read when it started.
        """
        try:
            if not results:
                show_message(self, "series.no_series_selected", title=self.operation_label)
                return False

            # Before resolve_target_axis_id, which may *create* the axis or
            # figure the results go on: a snapshot taken after it would have
            # nothing to say about that axis, and undo would leave an empty
            # one behind. The result tables are added to the same entry
            # below, once their names are known - they depend on which axis
            # this turned out to be.
            undo_entry: int | None = None
            if commit:
                undo_entry = self._repo.snapshot_for_undo(
                    self._repo.DESCRIPTOR_TABLES,
                    label=f"Apply {self.operation_label}",
                )

            # Most operations write back onto the axis their inputs came from.
            # One does not: see resolve_target_axis_id.
            axis_id = self.resolve_target_axis_id(axis_id_value, results)

            if commit:
                self._repo.snapshot_for_undo(
                    [self.result_table_name(axis_id, result) for result in results]
                    + [OPERATIONS_TABLE],
                    label=f"Apply {self.operation_label}",
                    entry_id=undo_entry,
                )
                self.cancel_operation_changes(refresh=False)
                self.begin_preview_transaction()
                self.apply_results_to_axis(axis_id, results)
                self.record_operation(axis_id, results, inputs or self.operation_inputs())
                # Release before optimize_db: VACUUM cannot run inside an open
                # transaction.
                self.commit_preview_transaction()
                self._repo.optimize_db()
                self._reset_preview_state_flags()
                self._finalize_successful_operation(
                    results=results,
                    verb="Applied",
                )
                return True

            self.cancel_operation_changes(refresh=False)
            self.begin_preview_transaction()
            self.preview_results_to_axis(axis_id, results)
            self._preview_axis_ids.add(axis_id)
            self._preview_active = True
            self.store_cached_results(results)
            self._refresh_after_preview_state_change()
            self.publish_results(
                self.format_results(results)
                or f"Preview updated: {len(results)} result(s)."
            )
            return True

        except Exception as exc:
            return self._operation_failed(exc, commit=commit)

    def operation_inputs(self) -> dict[str, Any]:
        """What an applied operation is recorded with: its parameters, entries and source series."""
        sources = []
        for row in self.selected_series():
            sources.append(
                {
                    "id": row_value(row, "id", default=None),
                    "name": str(row_value(row, "name", "series_name", default="")),
                    "sql_query": str(row_value(row, "sql_query", "query", "sql", default="")),
                    "roles": parse_roles(row_value(row, "roles", default={})),
                }
            )
        return {
            "parameters": self.parameter_values(),
            "entries": dialog_entries(self, inputs_only=True),
            "sources": sources,
        }

    def record_operation(self, axis_id: int, results: Sequence[Any], inputs: Mapping[str, Any]) -> None:
        """Add this Apply to the project's ``__operations__``, inside its transaction.

        A failure is logged, not raised: the record is about the result, and
        must never be the reason the result is lost.
        """
        try:
            written = [
                {
                    "table": self.result_table_name(axis_id, result),
                    "name": str(getattr(result, "result_name", "") or getattr(result, "output_name", "")),
                    "axis_id": int(axis_id),
                }
                for result in results
            ]
            report = self.format_results(results) or ""
        except Exception:
            applogger.exception(
                f"Could not record {self.operation_label} in the project's history",
            )
            return
        self.record_applied(written, report, inputs)

    def record_applied(
        self,
        written: Sequence[Mapping[str, Any]],
        report: str,
        inputs: Mapping[str, Any] | None = None,
    ) -> None:
        """Record an Apply as it is: what it read, wrote and reported.

        The common path above writes result tables; Outlier (rows hidden or
        coloured in the source tables) and Statistics (a report, nothing
        written) call this themselves when their own Apply is kept. Logged,
        never raised, for the same reason as record_operation.
        """
        try:
            inputs = inputs if inputs is not None else self.operation_inputs()
            self._repo.record_operation(
                operation=self.operation_label,
                dialog=f"{type(self).__module__.rsplit('.', 1)[-1]}:{type(self).__name__}",
                parameters=dict(inputs.get("parameters", {})),
                entries=dict(inputs.get("entries", {})),
                sources=list(inputs.get("sources", [])),
                results=[dict(item) for item in written],
                app_version=str(APP_VERSION),
                report=report,
            )
        except Exception:
            applogger.exception(
                f"Could not record {self.operation_label} in the project's history",
            )

    def operation_succeeded(self, *, commit: bool) -> None:
        """Called once a Preview (commit False) or an Apply has gone through."""

    def preview(self) -> bool:
        applogger.debug("Preview")
        return self._run_operation(commit=False)

    def ok(self) -> None:
        self.apply(then=lambda applied: self.accept() if applied else None)

    def cancel(self) -> None:
        self.cancel_operation_changes()
        self._discard_table_figure()
        super().reject()

    def cancel_operation_changes(self, *, refresh: bool = True) -> None:
        """Remove temporary Preview artifacts.

        This is intentionally the same cleanup used before creating a new
        Preview, so clicking Preview repeatedly behaves like Close followed by a
        fresh Preview.
        """
        rolled_back = self.rollback_preview_transaction()

        has_preview = bool(
            self._preview_active
            or self._preview_axis_ids
            or self._preview_table_names
        )
        if not rolled_back and not has_preview:
            return

        # The savepoint already undid everything it covered.  The artifact
        # sweep still runs when it did not, so a stale savepoint (or a preview
        # written before this dialog opened one) is not left behind.
        if not rolled_back:
            self.remove_preview_artifacts()
        else:
            self._forget_rolled_back_target()

        # Keep the grid consistent with the rolled-back axis descriptors.
        # This fixes preview flows that grow the grid (for example 1x1 -> 2x1)
        # and then cancel back to the original axis count.
        self._restore_original_grid()

        self._reset_preview_state_flags()

        if refresh:
            self._refresh_after_preview_state_change()

    def apply(self, then: Callable[[bool], None] | None = None) -> bool:
        """Run the operation and commit it; remembers the entries either way.

        *then* is told whether it was applied, once it has been - later, when
        the calculation runs in the background.
        """
        self._remember_state()

        def done(applied: bool) -> None:
            self._applied = self._applied or applied
            if then is not None:
                then(applied)

        return self._run_operation(commit=True, then=done)

    def reject(self) -> None:
        """Close/Cancel rejects temporary Preview changes."""
        self.stop_evaluation()
        self._remember_state()
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        self._discard_table_figure()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Window close button also rejects temporary Preview changes."""
        self.stop_evaluation()
        self._remember_state()
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        self._discard_table_figure()
        super().closeEvent(event)
