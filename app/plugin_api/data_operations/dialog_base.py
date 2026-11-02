"""The base of every analysis window: DataOperationDialogBase, and what an operation works with.

One QDialog subclass, one module. It is the window every data operation is (smoothing, interpolation,
outlier removal, fitting...) and the base of RolesOperationDialogBase (the table analyses), which reads
a table's columns cast into roles instead of a chart's series:

* the window: the inputs as tabs on the left (Data, Model, Parameters, Charts), the Results pane on the
  right, and one row of buttons - Preview, Revert, Stop and its bar, OK, Cancel;
* the Data page: a chart's series, or a table's columns drawn as a figure of their own first;
* reading and validating the input series, and computing - here, or on a worker thread with Stop;
* drawing the results (axes, figures, previews) inside a savepoint that Close rolls back;
* the project's history, and the window's memory (its size, its entries, its parameters).

Subclasses provide operation-specific controls by overriding the builder hooks, and declare what the
window shows and what it needs as the class attributes at the top of the class.

Beside it: the models (OperationModel), the result specs (ResultSeriesSpec), the jobs (CallJob,
SeriesJob) and the generated-table names (generated_table_name).
"""
from __future__ import annotations

import html
import inspect
import json
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from PySide6.QtCore import QEvent, QObject, QRegularExpression, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QRegularExpressionValidator, QShowEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app import APP_VERSION
from app.analysis import Stopped
from app.analysis.sampling import sampling_frequency
from app.data.data_source import parse_roles, resolve_role_column, row_value
from app.data.repo.operations import OPERATIONS_TABLE
from app.data.sqlite_repo import DatabaseError, SqliteRepo
from app.logs.logger import applogger
from app.plugin_api import reading_guide
from app.plugin_api.charts.grids import pivot_to_grid
from app.plugin_api.data_operations import next_steps
from app.plugin_api.data_operations.parameter_form import FIELD_MAX_WIDTH, ParameterForm
from app.plugin_api.data_operations.parameter_spec import BoolParam, FloatParam, Param, defaults
from app.plugin_api.data_operations.results import OperationResult
from app.plugin_api.operation_panels import merged_values
from app.styles.style import (
    MARGIN_TOOLBOX_PAGE,
    CardFrame,
    TitledCard,
    apply_dialog_shell,
    configure_combo_width,
    create_action_button,
    create_doc_link,
    icon_from_svg_source,
    set_doc_link,
    stdSizeAndlayout,
)
from app.utils import report_html
from app.utils.background import BackgroundTask, run_in_background
from app.utils.coercion import coerce_axis, to_numbers, to_numeric_axis
from app.utils.config import get_constant
from app.utils.dialog_state import (
    dialog_entries,
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
    set_dialog_entries,
    stored_dialog_state,
)
from app.utils.i18n import _
from app.utils.messages import show_message
from app.utils.series_validation import SeriesIssue, clean_xy, errors, validate_xy
from app.widgets.axis_series_selector import AxisSeriesSelector
from app.widgets.html_results import HtmlResultsView, looks_like_html, plain_to_html
from app.widgets.table_source_selector import TableSourceSelector

__all__ = [
    "GENERATED_TABLE_PREFIX",
    "CallJob",
    "DataOperationDialogBase",
    "OperationJob",
    "OperationModel",
    "ResultSeriesSpec",
    "SeriesJob",
    "SeriesOutcome",
    "generated_table_name",
]

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


def xy_columns(frame: pd.DataFrame, roles: Mapping[str, Any]) -> tuple[str, str]:
    """The x and y columns of a series' query: the ones its roles name, else the first two numeric ones."""
    columns = [str(column) for column in frame.columns]
    numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    x_col = resolve_role_column(columns, roles, "x") or ""
    y_col = resolve_role_column(columns, roles, "y") or ""
    if x_col not in columns:
        x_col = numeric[0] if numeric else columns[0]
    if y_col not in columns:
        y_col = numeric[1] if len(numeric) > 1 else x_col
    return x_col, y_col


class _PageScroll(QScrollArea):
    """A page that scrolls up and down, never sideways: as wide as its panel, and the panel as wide as it.

    A QScrollArea asks for no width of its own, so a page wider than the panel - a row of buttons, a
    combo with a minimum width - was cut at its edge, and moving the focus scrolled it sideways out of
    sight. Here the page's narrowest is the area's narrowest: the panel (and the window) cannot be made
    narrower than the controls, and with widgetResizable they stretch to the panel's width.
    """

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 - Qt's name
        super().setWidget(widget)
        widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt's name
        if watched is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(watched, event)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        hint = super().minimumSizeHint()
        page = self.widget()
        if page is None:
            return hint
        margins = self.contentsMargins()
        width = (page.minimumSizeHint().width() + margins.left() + margins.right()
                 + self.verticalScrollBar().sizeHint().width())
        return QSize(max(hint.width(), width), hint.height())


class _OperationTabs(QTabWidget):
    """The inputs' tabs: at least LEFT_MINIMUM wide, and as wide as the widest page needs.

    Not setMinimumWidth: a layout takes an explicit minimum over the hint, and the pages were cut at 320.
    """

    LEFT_MINIMUM = 320

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        hint = super().minimumSizeHint()
        return QSize(max(self.LEFT_MINIMUM, hint.width()), hint.height())


class DataOperationDialogBase(QDialog):
    """The window of every analysis: its layout, its buttons and its behaviour.

    Subclasses should:
    - declare their Name, Description, Icon and what they show (the attributes below);
    - create model controls in ``build_model_selector`` (or set MODELS);
    - create parameter controls in ``build_parameter_selector`` (or set PARAMS);
    - optionally override ``build_results_pane`` when a QLabel is not enough;
    - implement ``compute_results`` (or ``compute_series``, through a job);
    - provide generated-series metadata through the apply hooks;
    - connect operation-specific signals in ``connect_operation_signals``.
    """

    # ==================================================================
    # Declarations: what the operation is, shows and needs
    # ==================================================================

    #: How the operation presents itself.  All three live on the class rather
    #: than in config.json because an operation is a plugin: dropping one .py
    #: file into ``app/data_operations`` is meant to be the whole install,
    #: and presentation that had to be registered elsewhere made that untrue.
    #: The scanner reads them straight out of the source, so none of it costs
    #: an import.
    #: The data a chart-data operation picks: its series.
    DATA_PICKER: ClassVar[str] = "series"
    #: Whether the Graphics page is shown: the name of the table the result (the curve) is written to.
    #: The base builds the field only for an operation that shows the page; that operation reads _output_table_name().
    SHOW_OUTPUT: ClassVar[bool] = False
    Name: str = ""
    Description: str = ""
    #: The Data operations panel's section for it, and its place: sections are
    #: ordered by their lowest Order, an operation within one by its own. No
    #: Category lists it under "Other". Nothing else in the application names
    #: an operation, so a plugin file can be added or removed on its own.
    Category: str = ""
    Order: int = 0
    #: Words to find it by in the search (synonyms, English and Italian), and the
    #: guided analysis goals it serves (keys of app.analysis.catalog.GOALS).
    Keywords: str = ""
    Goals: list[str] = []

    #: Shown in the results pane whenever a control changes, until Preview or
    #: OK actually computes something. Controls no longer trigger compute_results()
    #: as a side effect of being edited - only Preview and OK (through
    #: _run_operation) do, so a heavy operation never runs on an intermediate
    #: value while a spin box is being dragged or typed into.
    PENDING_RESULTS_MESSAGE: str = "Press Preview or OK to see the result."

    #: The window's size when first opened; the size it is left at is remembered.
    WINDOW_SIZE: tuple[int, int] = (900, 640)
    #: Every series of the axis ticked at first: for an operation one usually runs on all of them.
    SELECTS_ALL_SERIES: bool = False

    #: The operation's icon, as an SVG document.  Inline rather than a path to
    #: one: a file beside the module is a second thing to copy and a second
    #: thing to lose.  ``style.icon_from_svg_source`` renders it; an operation
    #: that leaves this empty gets no icon and is otherwise unaffected.

    Icon = """
    <rect x="5" y="5" width="14" height="14" rx="3"/>
    <path d="M8.5 12h7"/>
    <path d="M12 8.5v7"/>
    """

    #: Emitted with the run's report, as HTML, once Apply has committed.
    #: The main window forwards it to the chart panel the operation ran on, so
    #: the numbers end up beside the chart instead of disappearing with the
    #: dialog that produced them.
    results_published = Signal(str)

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

    #: The panels the window shows. An operation turns one off when it has nothing to decide there: a
    #: function's Data page (SHOW_DATA), a model it does not have, parameters or charts it does not offer.
    SHOW_DATA: bool = True
    SHOW_MODEL: bool = True
    SHOW_PARAMETERS: bool = True
    SHOW_CHART: bool = True
    #: The data sources offered on the Data page: a table's columns (FROM_DATA), a chart's series
    #: (FROM_CHART). With one of them off, the other is the source and its choice is not shown.
    FROM_DATA: bool = True
    FROM_CHART: bool = True
    #: A stored function as the source. No analysis reads one yet, so the switch changes nothing.
    FROM_FUNCTION: bool = False

    # -- The window's inputs: the figure, the series, the table's columns --
    #: Whether the figure combo is disabled. True for every operation so far;
    #: an operation that genuinely means to move between figures would have to
    #: keep self._figure_id in step with it, which none currently does.
    LOCK_FIGURE_SELECTION: bool = True

    #: Whether the series list is shown. An operation that generates a series
    #: rather than transforming one has nothing to select, and a picker it
    #: never reads is worse than no picker.
    SHOWS_SERIES_SELECTOR: bool = True

    #: The charts a table's columns can be drawn as, the first the default:
    #: with more than one, the Data page offers them ("Chart:"). Scatter Plot
    #: reads x and y; a chart of one variable - Box Plot, Histogram - reads y
    #: as its value too (TableSourceSelector.chart_series).
    TABLE_CHARTS: tuple[str, ...] = ("Scatter Plot",)
    #: Where the table's columns are stored beside the entries ("@": no attribute's name).
    _TABLE_SOURCE_STATE = "@table_source"

    # -- Model and parameters, and where the result is drawn ---------------
    #: The models this operation offers, in the order the Model combo lists
    #: them: name -> OperationModel. Setting it gives the operation the
    #: standard selector - "Model:" with the combo, "Docs:" with a link kept
    #: on the selected model - without writing build_model_selector.
    MODELS: ClassVar[Mapping[str, OperationModel]] = {}

    #: The combo's tooltip, when MODELS builds it.
    MODEL_TOOLTIP: ClassVar[str] = "Choose the model."

    #: The combo's label, when MODELS builds it ("Operation:", "Method:"...).
    MODEL_LABEL: ClassVar[str] = "Model:"

    #: Declared parameters.  An operation that sets this gets its parameter
    #: form built, wired and read back for free; see ``parameter_spec``.
    #: Leaving it empty keeps the old behaviour, where the subclass overrides
    #: ``build_parameter_selector`` and builds the form by hand - which is
    #: still the right answer for a genuinely unusual control.
    PARAMS: tuple[Param, ...] = ()

    #: Where the result may be drawn, in the Charts tab's "Draw on" list (DEST_* values); empty: no
    #: choice - the operation decides (Fit draws on its data's axis). DEFAULT_DESTINATION is preselected.
    DESTINATIONS: tuple[str, ...] = ()
    DEFAULT_DESTINATION: str = ""

    # -- What the input must be (INPUT_*, read by validate_input_xy) -------
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

    # -- How results are written and drawn ---------------------------------
    #: How result_series_spec draws a result: a line through its points.
    RESULT_STYLE: Mapping[str, Any] = {"linestyle": "-", "linewidth": 1.6, "marker": ""}

    #: Result tables are named ``<prefix>_axis<id>_<series>[_<variant>]``,
    #: where the variant is this attribute of the result (its model, its
    #: method...) or nothing when None. An operation sets the two instead of
    #: overriding result_table_name; the names are kept as they always were,
    #: since a re-run replaces the table of the same name.
    RESULT_TABLE_PREFIX: str = ""
    RESULT_TABLE_VARIANT: str | None = "model"

    #: The Results pane as one table, a row per result (result_row gives the cells): (heading, alignment) per
    #: column. Empty: a section per result instead (result_section).
    RESULT_COLUMNS: tuple[tuple[str, str], ...] = ()

    # -- Computing: on which thread, and whether per series ----------------
    #: Whether Preview and Apply compute on a worker thread. Only for an
    #: operation that has a job (prepare_job or compute_series): the window
    #: stays responsive, Stop appears with a progress bar if the run takes
    #: more than a moment, and closing the window stops it.
    RUN_IN_BACKGROUND: bool = False

    #: Off switch for every operation at once - the tests turn it off so a
    #: Preview has finished by the time it returns. See dev/tests/conftest.py.
    BACKGROUND_ENABLED: ClassVar[bool] = True

    #: Whether a series with a z role is read as a surface: read_series then
    #: gives read_curve_or_surface's ``(is_surface, data)`` instead of x and y.
    READS_SURFACES: bool = False

    # -- State a hook may need before (or without) the part that sets it --------
    # Defaults on the class, not in __init__: a subclass may set one before calling it, and a hook
    # the constructor calls may read one before its widget is built.
    #: True once Apply has written the results; discard_result_target and the dialogs' own cleanup
    #: leave an applied result alone.
    _applied: bool = False
    #: The figure the main window shows, for "Draw on: new axis in the current figure"; None: no such choice.
    current_figure_id: int | None = None
    #: The Data page's "Data from:" radios (build_source_choice); None when the page offers no choice.
    source_table_radio: QRadioButton | None = None
    source_chart_radio: QRadioButton | None = None
    #: A table's columns as the source (the Data page), and the stack that switches it with the series.
    table_source: TableSourceSelector | None = None
    _source_stack: QStackedWidget | None = None
    _table_chart_form: QFormLayout | None = None
    _table_charts_index: int | None = None
    _table_figure_id: int | None = None
    _table_figure_spec: tuple[Any, ...] | None = None
    _switching_figure: bool = False
    #: The inputs' tabs (_build_common_ui).
    left_tabs: QTabWidget | None = None
    #: The Charts tab's "Draw on" list, and the form of its ``tab="charts"`` parameters.
    destination_combo: QComboBox | None = None
    chart_form: ParameterForm | None = None
    #: The Parameters tab's form when PARAMS builds it; the Docs link when there is one.
    _parameter_form_spec: ParameterForm | None = None
    _doc_link: QLabel | None = None
    _bare_model_combo: bool = False
    #: The axis or figure "Draw on" made for the results, and what it was made for (resolve_destination_axis).
    _result_axis_id: int | None = None
    _result_figure_id: int | None = None
    _result_destination: str | None = None
    _result_made_for: tuple[str, str, dict[str, Any]] | None = None
    _preview_figure_shown: int | None = None
    #: The calculation running in the background (run_job), and its [done, total].
    _job: BackgroundTask | None = None
    _job_progress: list[int] | None = None

    # ==================================================================
    # Construction
    # ==================================================================

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt's name
        """Hide a results view no layout adopted.

        The base dialog creates its HTML results view as a child of itself and
        a layout normally takes it over. A dialog that shows its results some
        other way (Geometry draws a preview chart) never adds it, and a child
        of a dialog with no layout is shown at the dialog's top-left corner:
        the white rectangle there. Whatever still has the dialog itself as its
        parent when it opens is in nobody's layout, so it is hidden.
        """
        if self._results_label.parent() is self:
            self._results_label.hide()
        super().showEvent(event)

    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        title: str | None = None,
        parent: QWidget | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if repo is None:
            applogger.error("%s requires a repository instance.", type(self).__name__)
        super().__init__(parent)

        # The operation's own Name, as the Analysis panel, the catalogue and
        # the report call it (todo H-09) - not a second name of its own.
        self.setWindowTitle(_(title or self.Name or "Data Operation"))
        self.resize(width or self.WINDOW_SIZE[0], height or self.WINDOW_SIZE[1])

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
        self._results_label.link_requested.connect(self._follow_link)
        #: The question a "What next?" link asked for: the main window opens it once this one is closed.
        self.next_request: next_steps.NextStep | None = None
        self._preview_axis_ids: set[int] = set()
        self._preview_table_names: set[str] = set()
        self._preview_active = False
        #: The last run's results.
        self._last_results: list[Any] = []
        #: The figures the last Preview or OK drew (create_report_figure keeps it).
        self.created_figure_ids: list[int] = []
        # The busy indicator: Stop and the bar appear once a run has taken BUSY_DELAY_MS (_show_progress).
        self._busy_since = 0
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(100)
        self._busy_timer.timeout.connect(self._show_progress)
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
        # The name of what the operation writes (its tables, its functions), on the Charts tab of every
        # operation: built before the subclass hooks so that the default can be set from them (Fit's "_curve").
        # A generated name starts with "_" - the validator lets no other be typed; empty is the automatic one.
        self._output_table_edit = QLineEdit(self)
        self._output_table_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^(_[^\s\"'`;]*)?$"), self._output_table_edit))
        self._output_table_edit.setPlaceholderText(_("automatic"))
        self._output_table_edit.setToolTip(_(
            "The name of the tables (and functions) the operation writes, starting with _ as every generated "
            "one does; each result adds its series' name to it. Empty: named automatically."))
        self._output_warning = QLabel(
            _("The name does not start with _: it is written as a generated table all the same."), self)
        self._output_warning.setWordWrap(True)
        self._output_warning.setVisible(False)
        self._output_table_edit.textChanged.connect(self._check_output_name)

        # Subclasses create widgets here before build_* hooks run. A
        # subclass that builds its own form sets this in build_parameter_selector.
        self._parameter_form: QFormLayout | None = None
        self.init_operation_widgets()

        self._model_selector_widget = self.build_model_selector()
        self._parameter_selector_widget = self.build_parameter_selector()
        self._results_widget = self.build_results_pane()

        self._build_common_ui()
        if self.model_combo.parent() is self:
            # Not placed by the model selector (an operation with combos of its own): not a stray box at 0, 0.
            self.model_combo.setVisible(False)
        self.connect_common_signals()
        self.connect_operation_signals()

        # Every subclass gets remembered entries and geometry for free (restore_window, keyed by state_key).
        # What Revert goes back to: the entries as built, before any memory.
        self._default_entries = dialog_entries(self, inputs_only=True)
        self._default_parameters = self.parameter_values()
        self.restore_window()
        # Restored with its signals blocked: the model's documentation link
        # still named the first model's ("Huber" linked to RANSAC's page).
        if self._doc_link is not None:
            self._update_doc_link()
        if self.SELECTS_ALL_SERIES:
            self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.mark_results_stale()

    def parameter_forms(self) -> list[ParameterForm]:
        """The Parameters tab's form and the Charts tab's, whose values are kept."""
        return [form for form in (self._parameter_form_spec, self.chart_form) if form is not None]

    def state_extra(self) -> dict[str, Any]:
        """The table the data came from, and its columns."""
        source = self.table_source
        if source is not None and source.table():
            return {self._TABLE_SOURCE_STATE: {"table": source.table(), "x": source.x_column(),
                                               "y": source.y_columns()}}
        return {}

    def restore_state_extra(self, stored: Mapping[str, Any]) -> None:
        """The table's columns back in the Data tab."""
        source = self.table_source
        chosen = stored.get(self._TABLE_SOURCE_STATE)
        if source is not None and isinstance(chosen, dict):
            source.set_spec(str(chosen.get("table", "")), str(chosen.get("x", "")), chosen.get("y") or [])
        # Restored with its signals blocked: the page follows the radio here.
        self._show_source_page()

    def revert_entries(self) -> None:
        """Every entry back to its default, as JMP's Revert: what the window had when first opened."""
        set_dialog_entries(self, self._default_entries, quietly=False)
        if self._parameter_form_spec is not None:
            self._parameter_form_spec.set_values(self._default_parameters)
        self._show_source_page()
        self._refresh_visibility()
        self.mark_results_stale()

    def _series_display_name(self, row: Any) -> str:
        return str(row["name"])

    def set_row_visible(self, field_widget: QWidget, visible: bool, *, form: QFormLayout | None = None) -> None:
        """Show or hide a form row together with its label.

        Hiding only the field leaves its label behind, which reads as a bug.
        Two dialogs had their own copy of this; the ``form`` argument defaults
        to ``self._parameter_form``, which is the one every subclass uses.
        """
        layout = form if form is not None else self._parameter_form
        if layout is None:
            return

        field_widget.setVisible(visible)
        label = layout.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)

    def set_doc_link(self, title: str, url: str) -> None:
        """Point this dialog's documentation label at a method's docs."""
        if self._doc_link is not None:
            set_doc_link(self._doc_link, title, url)

    # ------------------------------------------------------------------
    # "Draw on"
    # ------------------------------------------------------------------

    @classmethod
    def destination_label(cls, value: str) -> str:
        """The Charts tab's wording of a "Draw on" choice."""
        return {cls.DEST_SAME_AXIS: _("Same axis"), cls.DEST_NEW_AXIS: _("Same figure"),
                cls.DEST_NEW_FIGURE: _("New figure"), cls.DEST_NONE: _("None")}[value]

    def destination_chosen(self) -> str:
        """The "Draw on" choice in the Charts tab (a DEST_* value); "" when there is no such list."""
        combo = self.destination_combo
        return str(combo.currentData() or "") if combo is not None else ""

    def destination_combo_for(self, parent: QWidget, values: Sequence[str], default: str,
                              on_change: Callable[[], None]) -> QComboBox:
        """The Charts tab's "Draw on" list: *values* (DEST_*), *default* selected, as wide as its words."""
        combo = QComboBox(parent)
        for value in values:
            combo.addItem(self.destination_label(value), value)
        combo.setCurrentIndex(max(0, combo.findData(default)))
        combo.setToolTip(_("Where the result is drawn: on the axis of its data (the first chart only), in the "
                           "current figure as new axes, or in a figure of its own."))
        # As narrow as the panel needs (its words elided, whole in the list that drops down), never wider
        # than a field: at its words' width it pushed the Charts page past a narrow window's edge.
        configure_combo_width(combo, follow_tooltip=False)
        combo.setMaximumWidth(FIELD_MAX_WIDTH)
        combo.currentIndexChanged.connect(lambda _index: on_change())
        return combo

    # ------------------------------------------------------------------
    # Where the data comes from
    # ------------------------------------------------------------------

    def build_source_choice(self, parent: QWidget, *, from_chart: bool,
                            on_change: Callable[[bool], None]) -> QHBoxLayout:
        """"Data from: (o) Table ( ) Chart" - every analysis reads either; *from_chart* checked at first.

        Sets source_table_radio and source_chart_radio; *on_change(from_chart)* follows the choice.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Data from:"), parent))
        table_radio = self.source_table_radio = QRadioButton(_("Table with roles"), parent)
        table_radio.setToolTip(_("A table's columns, each in the role the operation reads it as (X, Y)."))
        chart_radio = self.source_chart_radio = QRadioButton(_("Chart"), parent)
        chart_radio.setToolTip(_("The series a chart draws."))
        group = QButtonGroup(parent)
        for radio in (table_radio, chart_radio):
            group.addButton(radio)
            row.addWidget(radio)
        row.addStretch(1)
        # A source switched off (FROM_DATA, FROM_CHART) is not offered; the other one is the source.
        table_radio.setVisible(self.FROM_DATA)
        chart_radio.setVisible(self.FROM_CHART)
        chart_first = (from_chart and self.FROM_CHART) or not self.FROM_DATA
        (chart_radio if chart_first else table_radio).setChecked(True)
        chart_radio.toggled.connect(on_change)
        return row

    def reads_chart(self) -> bool:
        """Whether the data comes from a chart (Data from: Chart)."""
        return self.source_chart_radio is not None and self.source_chart_radio.isChecked()

    # ------------------------------------------------------------------
    # The Charts tab
    # ------------------------------------------------------------------

    def build_charts_page(
        self,
        chart_params: Sequence[Param],
        *,
        destinations: Sequence[str] = (),
        default_destination: str = "",
        on_change: Callable[[], None],
        context: Callable[[], Mapping[str, Any]] | None = None,
        on_destination_change: Callable[[], None] | None = None,
    ) -> QWidget | None:
        """The Charts tab: "Draw on" first (*destinations*), then the charts (*chart_params*); None if neither.

        A chart a choice elsewhere makes impossible is hidden by its own ``visible_for``, read against
        *context* (the response's type, the model...): the tab lists only what can be drawn.
        """
        if not (destinations or chart_params):
            return None
        panel = CardFrame(self, "operationChartsCard")
        layout = panel.layout()
        combo: QComboBox | None = None
        if destinations:
            # The charts a destination makes possible (none for "None") follow it.
            def destination_changed() -> None:
                if self.chart_form is not None:
                    self.chart_form.refresh_visibility()
                (on_destination_change or on_change)()

            combo = self.destination_combo = self.destination_combo_for(
                panel, destinations, default_destination or destinations[0], destination_changed)
        if chart_params:
            # The "Draw on" choice is read by the charts' visibility rules (a chart hidden for Same axis).
            def with_destination() -> Mapping[str, Any]:
                return {**(context() if context is not None else {}), "_destination": self.destination_chosen()}

            self.chart_form = ParameterForm(chart_params, self, on_change=on_change, context=with_destination)
            layout.addWidget(self.chart_form.widget)
        if combo is not None:
            # In the charts' own form when there are some: their boxes under the combo, in its column.
            if self.chart_form is not None:
                self.chart_form.layout.insertRow(0, _("Draw on:"), combo)
            else:
                row = QFormLayout()
                stdSizeAndlayout(row)
                row.addRow(_("Draw on:"), combo)
                layout.insertLayout(0, row)
        layout.addStretch(1)
        return panel

    def chart_value(self, name: str, default: Any = None) -> Any:
        """The Charts tab's *name* (a ``tab="charts"`` parameter); *default* when there is no such parameter."""
        values = self.chart_form.values() if self.chart_form is not None else {}
        return values.get(name, default)

    def chart_enabled(self, name: str) -> bool:
        """Whether the chart *name* is ticked and can be drawn (not hidden); True for one not declared."""
        form = self.chart_form
        if form is None or name not in form.values():
            return True
        return bool(form.values()[name]) and not form.field(name).isHidden()

    # ------------------------------------------------------------------
    # The window
    # ------------------------------------------------------------------

    def _lay_out_window(self, settings: QWidget) -> None:
        """Lay the window out: *settings* and the Results pane side by side, each titled, the buttons under them.

        Preview, Revert, the operation's own (build_extra_action_buttons), then Stop and its bar - shown only
        while a calculation runs - and, on the right, OK and Cancel. No Copy button: the report's right-click
        menu copies it, and exports it as HTML or PDF (HtmlResultsView).
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
        right.card.layout().addWidget(self._results_widget, 1)
        right.setMinimumWidth(380)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        row = QHBoxLayout()
        stdSizeAndlayout(row)
        self.preview_button = create_action_button(parent=self, action_id="preview", action=self.preview, layout=row)
        # Back to the defaults, beside Preview: it acts on the entries only.
        self.revert_button = create_action_button(parent=self, action_id="operation_revert",
                                                  action=self.revert_entries, layout=row)
        # The operation's own, on the left: they act on the dialog's state, unlike OK and Cancel.
        self.build_extra_action_buttons(row)
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
        self.apply_button = create_action_button(parent=self, action_id="apply", action=self.ok, layout=row)
        self.close_button = create_action_button(parent=self, action_id="close", action=self.reject, layout=row)

        root = QVBoxLayout(self)
        # One helper owns dialog padding and size for every dialog in the app.
        apply_dialog_shell(self, root)
        root.addWidget(splitter, 1)
        root.addLayout(row, 0)

    # ------------------------------------------------------------------
    # The Results pane
    # ------------------------------------------------------------------

    def results_widget(self) -> HtmlResultsView:
        """The Results pane."""
        return self._results_label

    def publish_results(self, formatted: str) -> None:
        """Show a run's report (format_results' output) in the Results pane; an operation may show it elsewhere."""
        self.show_results(formatted)

    def show_results(self, content: str) -> None:
        """Show a report in the Results pane: markup rendered, plain text kept as it is written."""
        text = str(content or "")
        if looks_like_html(text):
            self._results_label.setHtml(text)
        else:
            self._results_label.setContent(text)

    def show_failure(self, error: BaseException | str, *, headline: str = "") -> None:
        """Say in the Results pane why nothing came of the run: the log alone left the pane unchanged."""
        message = str(error) or type(error).__name__
        self._results_label.setHtml(
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

    def busy_widgets(self) -> list[QWidget]:
        """What is switched off while a calculation runs: whatever would start a second one."""
        return [self.preview_button, self.apply_button]

    @property
    def job_running(self) -> bool:
        """A calculation is running in the background."""
        return self._job is not None and self._job.running

    def run_job(self, work: Callable[[threading.Event], Any], on_done: Callable[[Any], None], *,
                progress: list[int] | None = None) -> bool:
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

        task = self._job = run_in_background(work, finished, self._job_failed)
        return True

    def stop(self) -> None:
        """Ask the calculation to stop at its next step (and the window to close: nothing is written)."""
        if self._job is not None and self._job.running:
            self._job.cancel()

    def stop_requested(self) -> bool:
        """True once Stop was pressed (or the window closed) during the calculation; poll it in long loops."""
        return self._job is not None and self._job.cancel_event.is_set()

    def _job_failed(self, error: BaseException) -> None:
        self._set_job_running(False)
        if isinstance(error, Stopped):
            self.show_results(_("Stopped. Nothing was changed."))
            return
        applogger.error("%s failed: %s", self.Name or type(self).__name__, error)
        self.show_failure(error, headline=_(self.FAILURE_HEADLINE))

    def _set_job_running(self, running: bool) -> None:
        # The buttons go off at once - a second click must not start a second run - but Stop and the
        # bar wait: most runs end before they would have been read.
        for widget in self.busy_widgets():
            widget.setEnabled(not running)
        if running:
            self._busy_since = 0
            self._busy_timer.start()
            return
        self._busy_timer.stop()
        self.stop_button.setVisible(False)
        self.progress_bar.setVisible(False)

    def _show_progress(self) -> None:
        """Every 100 ms of a run: Stop and the bar once it is slow, then the count."""
        self._busy_since += 100
        if self._busy_since < self.BUSY_DELAY_MS:
            return
        bar = self.progress_bar
        if not self.stop_button.isVisible():
            self.stop_button.setVisible(True)
            bar.setVisible(True)
            self.show_results(_("Working... press Stop to end it early."))
        progress = self._job_progress
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

    def remember_window(self) -> None:
        """Keep the entries, the parameters and the size - from every exit path, when the choice is final."""
        extra: dict[str, Any] = dict(self.state_extra())
        forms = self.parameter_forms()
        if forms:
            extra[self.PARAMETERS_STATE] = merged_values(forms)
        try:
            save_dialog_state(self, self.state_key(), extra)
        except TypeError:
            # A parameter that does not go into JSON: the entries all the same.
            applogger.exception("Could not remember %s's parameters", self.state_key())
            save_dialog_state(self, self.state_key())
        save_window_geometry(self, self.geometry_key())

    def restore_window(self) -> None:
        """Put back the size, the entries and the parameters remembered; a stale value is logged, not raised."""
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
        if default is None:
            default = next(iter(self.MODELS), "")
        data = self.model_combo.currentData()
        if data is not None and str(data):
            return str(data)
        return self.model_combo.currentText() or default


    def _leave(self) -> None:
        self.stop()
        self.remember_window()
        self.discard_unapplied()

    def reject(self) -> None:
        """Cancel or Close: the calculation stopped, the choices kept, the Preview taken back."""
        self._leave()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt's name
        """The window's close button: as Cancel."""
        self._leave()
        super().closeEvent(event)

    # ==================================================================
    # The preview transaction and report figures
    # ==================================================================

    def begin_preview_transaction(self, name: str | None = None) -> None:
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
        self._repo.begin_preview_savepoint(name or f"preview_{self.__class__.__name__}")

    def rollback_preview_transaction(self) -> bool:
        """Undo every database change made since the preview started."""
        if not self._repo.preview_savepoint_active:
            return False
        try:
            return bool(self._repo.rollback_preview_savepoint())
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to roll back the preview savepoint")
            return False

    def commit_preview_transaction(self) -> None:
        """Make the preview changes permanent."""
        if not self._repo.preview_savepoint_active:
            return
        try:
            self._repo.release_preview_savepoint()
        except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
            applogger.exception("Failed to commit the preview savepoint")

    def draw_on(self) -> str:
        """The Charts tab's choice; a new figure for one in the current figure when there is none to add to."""
        chosen = self.destination_chosen() or self.DEST_NEW_FIGURE
        if chosen in (self.DEST_NEW_AXIS, self.DEST_SAME_AXIS) and self.current_figure_id is None:
            return self.DEST_NEW_FIGURE
        return chosen

    def create_report_figure(self, name: str, notes_html: str, panels: Sequence[tuple[Any, ...]]) -> int | None:
        """A figure of *panels*, two to a row, with *notes_html* in its notes; its id.

        Each panel is ``(chart type, title, x label, y label, series)``, and
        optionally the axis options last (annotations, reference lines); each
        series ``(name, sql, roles, style)`` - see series_select.
        """
        from app.scanners.axis_renderer_scanner import get_renderer
        from app.widgets.chart_panel import FIGURE_VIEW_OPTIONS_KEY

        # Chart types are plugins that can be removed: a panel whose chart type
        # is not installed is left out (and logged) - the report is still made.
        available = []
        for panel in panels:
            if get_renderer(str(panel[0])) is None:
                applogger.warning(
                    "%s: the chart type %r is not installed; the panel %r is left out.",
                    self.Name, panel[0], panel[1],
                )
                continue
            available.append(panel)
        # The panels whose chart type is cleared in the Charts tab are left out.
        panels = [panel for panel in available if self.chart_enabled(str(panel[0]))]

        repo = self._repo
        draw_on = self.draw_on()
        if draw_on == self.DEST_SAME_AXIS:
            # Same axis: the first chart ticked only; the others are not drawn.
            panels = panels[:1]
        if draw_on == self.DEST_NONE or not panels:
            # The report is the result - no chart asked for, or every box cleared: no empty figure; the report
            # stays in Results and in the project's history.
            return None
        target = self.current_figure_id if draw_on != self.DEST_NEW_FIGURE else None
        descriptor = repo.load_figure_descriptor(int(target)) if target is not None else None
        onto = next((axis for axis in (descriptor.axes if descriptor else []) if axis.name == panels[0][0]), None) \
            if draw_on == self.DEST_SAME_AXIS else None
        if onto is not None and target is not None:
            # Same axis: the panel's series added to the figure's axis of its kind.
            for position, (series_name, sql, roles, style) in enumerate(panels[0][4], start=len(onto.series)):
                repo.create_series_descriptor(axis_id=int(onto.id), series_index=position, name=series_name,
                                              sql_query=sql, roles=roles, style=style)
            self.created_figure_ids.append(int(target))
            return int(target)
        if descriptor is not None and target is not None:
            # New axes after the figure's own, its grid grown to show them.
            figure_id = int(target)
            columns = max(1, int(descriptor.ncols or 1))
            first = max((int(axis.index) for axis in descriptor.axes), default=-1) + 1
            rows = max(int(descriptor.nrows or 1), math.ceil((first + len(panels)) / columns))
            repo.set_figure_grid(figure_id, nrows=rows, ncols=columns)
        else:
            first = 0
            columns = 2 if len(panels) > 1 else 1
            rows = max(1, math.ceil(len(panels) / columns))
            figure_id = int(repo.create_figure_descriptor(
                name=name, nrows=rows, ncols=columns,
                options={FIGURE_VIEW_OPTIONS_KEY: {"notes_html": notes_html}},
            ))
        for index, (chart, title, x_label, y_label, series, *options) in enumerate(panels, start=first):
            axis_id = int(repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=index, chart_type=chart,
                title=title, x_label=x_label, y_label=y_label, options=dict(options[0]) if options else {},
            ))
            for position, (series_name, sql, roles, style) in enumerate(series):
                repo.create_series_descriptor(
                    axis_id=axis_id, series_index=position, name=series_name,
                    sql_query=sql, roles=roles, style=style,
                )
        self.created_figure_ids.append(figure_id)
        return figure_id

    # ==================================================================
    # Builder hooks, layout and the Data page
    # ==================================================================

    # ------------------------------------------------------------------
    # Builder hooks
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        """Create subclass widgets before build_model_selector/build_parameter_selector.

        The base constructor calls the builder hooks, so subclasses must create
        controls used by those hooks here rather than after super().__init__().
        """

    def build_model_selector(self) -> QWidget:
        """The Model combo and its Docs link, from MODELS.

        Without MODELS, just the shared model_combo, for an operation that
        fills it itself.
        """
        if not self.MODELS:
            self._bare_model_combo = True
            panel = CardFrame(self, "operationModelCard")
            layout = panel.layout()
            stdSizeAndlayout(self.model_combo)
            layout.addWidget(self.model_combo)
            return panel

        if self._doc_link is None:
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

    def extra_tabs(self) -> list[tuple[QWidget, str]]:
        """Further (page, title) tabs after Parameters, for an operation with more to set."""
        return []

    # ------------------------------------------------------------------
    # The result's name, at the top of the Charts tab
    # ------------------------------------------------------------------
    def _add_output_name(self, charts_tab: QWidget) -> None:
        """"Result name:" first on the Charts tab - what the operation writes is named there, "_" first."""
        form = QFormLayout()
        stdSizeAndlayout(form)
        form.addRow(_("Result name:"), self._output_table_edit)
        form.addRow("", self._output_warning)
        charts_tab.layout().insertLayout(0, form)

    def output_stem(self) -> str:
        """The name typed for the results ("_" first), or "" for the automatic one."""
        text = self._output_table_edit.text().strip()
        return generated_table_name(text, fallback="Result") if text else ""

    def output_table_default(self) -> str:
        """The name the output starts with: a generated table named "curve" (translated)."""
        return generated_table_name(_("curve"), fallback="curve")

    def _refresh_default_output_name(self, force: bool = False) -> None:
        """Put the default name in the field: always when *force*, else only while it is empty."""
        if force or not self._output_table_edit.text().strip():
            self._output_table_edit.setText(self.output_table_default())

    def _output_table_name(self) -> str:
        """The name the result is written to: what the field holds, a generated table's prefix always added."""
        return generated_table_name(
            self._output_table_edit.text().strip() or self.output_table_default(),
            fallback="curve",
        )

    def _check_output_name(self, text: str) -> None:
        """Warn while the name lacks the generated table's prefix: it is added on save all the same."""
        text = text.strip()
        self._output_warning.setVisible(bool(text) and not text.startswith(GENERATED_TABLE_PREFIX))

    def build_charts_tab(self) -> QWidget | None:
        """The Charts tab: "Draw on" (DESTINATIONS), then the PARAMS marked tab="charts"; None when neither."""
        return self.build_charts_page(
            tuple(spec for spec in self.PARAMS if spec.tab == "charts"),
            destinations=self.DESTINATIONS, default_destination=self.DEFAULT_DESTINATION,
            on_change=self.mark_results_stale, context=self.parameter_context,
            on_destination_change=self.destination_changed,
        )

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
            tuple(spec for spec in self.PARAMS if spec.tab != "charts"),
            self,
            on_change=self.parameters_changed,
            context=self.parameter_context,
        )
        # Kept under the name the base already uses for the hand-built form, so
        # set_row_visible and the state helpers keep working unchanged.
        self._parameter_form = self._parameter_form_spec.layout
        return self._parameter_form_spec.widget

    def parameters_changed(self) -> None:
        """A parameter changed: the results on screen are stale. An operation that follows a choice with
        others (the starting values a model takes) overrides it and calls this."""
        self.mark_results_stale()

    def parameter_context(self) -> Mapping[str, Any]:
        """Return values a ``visible_for`` rule may reference but not own.

        The model combo by default, under the name ``model``: nearly every
        operation's parameter visibility depends on which model is selected,
        and the combo belongs to this base rather than to ``PARAMS``.
        """
        context: dict[str, Any] = {"model": self.current_model()}
        if self.destination_combo is not None:
            # "Draw on": a chart shown only while there is somewhere to draw it.
            context["destination"] = str(self.destination_combo.currentData())
        return context

    def parameter_values(self) -> dict[str, Any]:
        """Return the declared parameters' current values, keyed by name.

        Falls back to the declared defaults when there is no form yet, so an
        operation can call this from ``compute_results`` during construction
        without a special case.
        """
        form = self._parameter_form_spec
        values = defaults(self.PARAMS) if form is None else dict(form.values())
        if self.chart_form is not None:
            values.update(self.chart_form.values())
        if self.DESTINATIONS:
            combo = self.destination_combo
            values["destination"] = (str(combo.currentData()) if combo is not None
                                     else self.DEFAULT_DESTINATION or self.DESTINATIONS[0])
        return values

    def build_results_pane(self) -> QWidget:
        """Return the shared HTML results widget."""
        return self._results_label

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
        self.series_selector.selection_changed.connect(lambda *_args: self.mark_results_stale())
        self.series_selector.selection_changed.connect(lambda *_args: self._update_origin_label())
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
        """Show the parameters (and the charts) the selected model uses: the PARAMS' visible_for rules.

        Eight operations had this same body as their own; one that builds
        its controls by hand overrides it (or calls this, then adds its own).
        """
        for form in self.parameter_forms():
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
        if self._doc_link is None:
            self._doc_link = create_doc_link(self)
        form.addRow(_("Docs:"), self._doc_link)
        panel.layout().addWidget(form_widget)
        return panel

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
        axis_id = self._result_axis_id
        if axis_id is None:
            return
        try:
            self._repo.update_axis_descriptor(axis_id=axis_id, title=title, x_label=x_label, y_label=y_label)
        except (DatabaseError, ValueError):
            applogger.exception(f"Failed to label the {self.operation_label} result axis")

    def current_axis_name(self) -> str:
        """The name of the axis picked in the Data page."""
        return self.series_selector.selected_axis_name()

    # -- Sampling rate: the "derive it from x, or type it" pair -----------

    @staticmethod
    def sampling_rate_params(visible_for: Mapping[str, Sequence[Any]]) -> tuple[Param, Param]:
        """The sampling rate, for PARAMS: read off the x role's spacing, or typed - both shown where *visible_for*
        says (the models that use one); read with sampling_rate."""
        return (
            BoolParam("fs_auto", "Sampling rate from x", default_value=True, visible_for=visible_for,
                      tooltip="Take the sampling frequency from the spacing of the x role."),
            FloatParam("fs", "fs:", default_value=1.0, minimum=1e-9, maximum=1e12, decimals=6, step=0.001,
                       visible_for={**visible_for, "fs_auto": (False,)},
                       tooltip="Samples per unit of x, used when not derived."),
        )

    def sampling_rate(self, x_values: np.ndarray, name: str) -> float:
        """fs in samples per unit of x: the typed one, or read off the x role."""
        params = self.parameter_values()
        if not params.get("fs_auto", True):
            return float(params["fs"])
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
        # Left null: the page's own inset (MARGIN_TOOLBOX_PAGE) is set where it is mounted.
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

    @staticmethod
    def _scrolled(page: QWidget, parent: QWidget) -> QScrollArea:
        """*page* in a frameless vertical scroll area on the same ground, as wide as the page needs."""
        scroll = _PageScroll(parent)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        # setWidget turns the page's own background fill on: a plain page (the roles' Data page) was then painted
        # in the window's grey, a darker box on the sheet's ground. Left off, the ground shows through.
        page.setAutoFillBackground(False)
        scroll.viewport().setProperty("toolboxPage", True)
        scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # The sheet paints it; filled with the palette's window grey it was a darker box
        # inside the Settings frame - a panel in a panel.
        scroll.viewport().setAutoFillBackground(False)
        return scroll

    def input_pages(self) -> list[tuple[QWidget, str]]:
        """The tabs before Charts, as (page, title): Data, Model, Parameters, Graphics, then extra_tabs.

        A page its SHOW_ switch turns off is left out (Data, for an operation that selects nothing), and so
        is Model when there is no model to pick, and Parameters when there is none to set.
        """
        pages = []
        if self.SHOW_DATA:
            pages.append((self.axis_series_panel, _("Data")))
        # The bare combo with nothing in it is no choice: an operation without models has no Model tab.
        if self.SHOW_MODEL and not (self._bare_model_combo and not self.model_combo.count()):
            pages.append((self.model_panel, _("Model")))
        # The bare widget of an operation without PARAMS has nothing on it: no Parameters tab either.
        if self.SHOW_PARAMETERS and self.parameters_panel.layout() is not None:
            pages.append((self.parameters_panel, _("Parameters")))
        pages.extend(self.extra_tabs())
        return pages

    def _build_common_ui(self) -> None:
        """Build the common shell: the inputs as tabs on the left, the results on the right."""
        # Tabs, as Chart properties: the accordion's three stacked headers took
        # height and showed one section at a time all the same.
        left_tabs = _OperationTabs(self)
        left_tabs.setObjectName("operationTabs")
        left_tabs.setDocumentMode(True)
        left_tabs.setUsesScrollButtons(True)
        left_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        left_tabs.tabBar().setExpanding(False)
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

        pages = self.input_pages()
        # A panel no page shows is still a child of the window: left as it is, it is drawn over the top-left corner.
        for panel in (self.axis_series_panel, self.model_panel, self.parameters_panel):
            if panel.parent() is self and all(panel is not content for content, _title in pages):
                panel.hide()
        charts_tab = self.build_charts_tab() if self.SHOW_CHART else None
        if charts_tab is None and self.SHOW_OUTPUT:
            charts_tab = CardFrame(self, "operationChartsCard")  # the result's name needs a page
        if charts_tab is not None:
            self._add_output_name(charts_tab)
        else:
            # No Charts tab, nothing named: the field is not on any page - not loose on the window either.
            self._output_table_edit.hide()
            self._output_warning.hide()
        # The Charts tab's index when a table's chart is all it holds: shown only while a table is read.
        self._table_charts_index = None
        table_charts_only = None
        if charts_tab is None and self.SHOW_CHART and self._offers_table_chart():
            # A table's chart is picked here even when the operation has no other chart to choose.
            charts_tab = table_charts_only = CardFrame(self, "operationChartsCard")
        if charts_tab is not None:
            pages.append((charts_tab, _("Charts")))
            self._add_table_chart_choice(charts_tab)
        data_index = 0  # the Data page when there is one - the dialog opens on it, else on Model
        for content, title in pages:
            if content is self.axis_series_panel:
                data_index = left_tabs.count()
            page = self._toolbox_page(content)
            # The ground and the inset a toolbox page had (grey on macOS,
            # white on Windows; the toolboxPage rule in each sheet).
            page.setProperty("toolboxPage", True)
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            page.layout().setContentsMargins(*MARGIN_TOOLBOX_PAGE)
            if content is not self.axis_series_panel:
                # Scrolled when the window is shorter than the page, as the
                # table analyses' pages: a long parameter form set the whole
                # window's minimum height, too tall for a laptop.
                page = self._scrolled(page, left_tabs)
            if content is table_charts_only:
                self._table_charts_index = left_tabs.count()
            left_tabs.addTab(page, title)
        left_tabs.setCurrentIndex(data_index)
        self.left_tabs = left_tabs
        self._show_table_charts_tab()

        self._lay_out_window(left_tabs)

    # ------------------------------------------------------------------
    # Data from a chart, or from a table
    # ------------------------------------------------------------------

    def _build_source_panel(self) -> QWidget:
        """The Data page: the chart's series, or a table's columns."""
        self.table_source = None
        # The figure the table's columns were drawn on, and what they were.
        self._table_figure_id = None
        self._table_figure_spec = None
        # The figure the dialog was opened on, to go back to.
        self._chart_figure_id = self._figure_id
        if not (self.FROM_DATA and self.SHOWS_SERIES_SELECTOR and self.SHOW_DATA):
            return self.series_selector

        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        # A table's columns are drawn as a figure of their own first (_prepare_table_source).
        layout.addLayout(self.build_source_choice(panel, from_chart=True, on_change=self._on_source_changed))

        source = self.table_source = TableSourceSelector(self._repo, panel, charts=tuple(self.TABLE_CHARTS))
        stack = self._source_stack = QStackedWidget(panel)
        stack.addWidget(self.series_selector)
        stack.addWidget(source)
        layout.addWidget(stack, 1)

        source.changed.connect(self._on_table_columns_changed)
        return panel

    def reads_table(self) -> bool:
        """Whether this run takes its data from a table's columns."""
        radio = self.source_table_radio
        return self.table_source is not None and radio is not None and radio.isChecked()

    def _offers_table_chart(self) -> bool:
        """Whether a table can be drawn as more than one chart, so the Charts tab must say which."""
        return self.table_source is not None and self.table_source.chart_combo.count() >= 2

    def _add_table_chart_choice(self, charts_tab: QWidget) -> None:
        """A table's chart choice (what its columns are drawn as) on the Charts tab, above Draw on."""
        if self.table_source is None or not self._offers_table_chart():
            return
        form = self._table_chart_form = QFormLayout()
        stdSizeAndlayout(form)
        form.addRow(_("Chart:"), self.table_source.chart_combo)
        charts_tab.layout().insertLayout(0, form)
        form.setRowVisible(self.table_source.chart_combo, self.reads_table())

    def _show_source_page(self) -> None:
        """The Data page shows the table's columns when the run reads them, else the chart's series."""
        if self._source_stack is not None:
            self._source_stack.setCurrentIndex(1 if self.reads_table() else 0)

    def _show_table_charts_tab(self) -> None:
        """A Charts tab that only picks a table's chart is empty while a chart is read: hidden then."""
        if self._table_charts_index is not None and self.left_tabs is not None:
            self.left_tabs.setTabVisible(self._table_charts_index, self.reads_table())

    def _on_source_changed(self, *_ignored: Any) -> None:
        self._show_source_page()
        self._show_table_charts_tab()
        if self._table_chart_form is not None and self.table_source is not None:
            self._table_chart_form.setRowVisible(self.table_source.chart_combo, self.reads_table())
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
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
        source = self.table_source
        columns = source.spec() if source is not None else None
        if source is None or columns is None:
            raise ValueError(_("Choose a table and at least one Y column."))
        chart = source.chart()
        spec = (*columns, chart)
        if spec == self._table_figure_spec and self._table_figure_id is not None:
            if self._repo.get_figure_descriptor(int(self._table_figure_id)) is not None:
                return
        self.cancel_operation_changes(refresh=False)
        self.discard_result_target()
        self._discard_table_figure()

        table, x, ys = columns
        one = ys[0] if len(ys) == 1 else ""
        # A box plot's values go up; a histogram's along.
        x_label, y_label = {"Box Plot": ("", one), "Violin Plot": ("", one),
                            "Histogram": (one, _("Count"))}.get(chart, (x or _("Row"), one))
        figure_id = int(self._repo.create_figure_descriptor(
            name=f"{table}: {', '.join(ys)}", nrows=1, ncols=1,
        ))
        axis_id = int(self._repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type=chart, title=table,
            x_label=x_label, y_label=y_label, options={"grid": True},
        ))
        for index, y in enumerate(ys):
            sql, roles = TableSourceSelector.chart_series(chart, table, x, y)
            self._repo.create_series_descriptor(
                axis_id=axis_id, series_index=index, name=y, sql_query=sql, roles=roles,
            )
        applogger.info("%s: drew %s from table %s as figure %s.", self.operation_label, ", ".join(ys), table, figure_id)
        self._table_figure_id, self._table_figure_spec = figure_id, spec
        self._use_figure(figure_id)

    def _discard_table_figure(self) -> None:
        """Remove the figure drawn from a table, unless Apply kept it; back to the chart."""
        figure_id = self._table_figure_id
        self._table_figure_id, self._table_figure_spec = None, None
        if figure_id is None:
            return
        if not self._applied:
            try:
                self._repo.delete_figure(int(figure_id))
            except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
                applogger.exception("Failed to discard the figure drawn from a table")
        self._use_figure(self._chart_figure_id)

    # ==================================================================
    # Inputs and computing
    # ==================================================================

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

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
        frame, roles = self._series_frame(row)
        x_col, y_col = xy_columns(frame, roles)
        return self.prepare_input_xy(
            self.numeric_x(frame[x_col], name),
            self.numeric_y(frame[y_col]),
            label=name,
        )

    def _series_frame(self, row: Any) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        """Run one series' SQL: its rows as a frame, and its roles. Raises when there is no query or no rows."""
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            raise ValueError("the series has no SQL query")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            raise ValueError("the series query returned no rows")
        return frame, parse_roles(row_value(row, "roles", default={}))

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
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
        selector = self.series_selector
        rows = self.selected_series()
        if not rows:
            selector.set_origin_text("")
            return
        try:
            selector.set_origin_text(self.series_origin(rows[0]))
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
        frame, roles = self._series_frame(row)
        columns = [str(column) for column in frame.columns]
        z_col = str(roles.get("z") or "")
        if not z_col or z_col not in columns:
            raise ValueError("the series has no z role")

        x_col, y_col = xy_columns(frame, roles)
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

        1. **Exact.** ``pivot_to_grid`` (``app/plugin_api/charts/grids.py``) recognises a
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

        from scipy.interpolate import griddata  # here, not at import: 0.5 s of startup

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

    def result_style(self, result: Any, **overrides: Any) -> dict[str, Any]:
        """A generated series' style: this dialog's marks (generated_style_filter), the series and the model it
        came from, then *overrides*."""
        return {**self.generated_style_filter, "source_name": getattr(result, "source_name", ""),
                "model": getattr(result, "model", ""), **overrides}

    def curve_spec(self, table_name: str, result: Any, *, y: str = "y", name: str = "",
                   **style: Any) -> ResultSeriesSpec:
        """The result table's *y* column against x, in x order, as one series - named after the result unless
        *name* says otherwise, styled by result_style and *style*."""
        column = "y" if y == "y" else f"{y} AS y"
        return ResultSeriesSpec(
            name=name or result.result_name,
            sql_query=f'SELECT x, {column} FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style=self.result_style(result, **style),
        )

    def result_series_spec(self, axis_id: int, table_name: str, result: Any) -> ResultSeriesSpec:
        """The chart series of one saved result table: its y against x, drawn as RESULT_STYLE says."""
        del axis_id
        return self.curve_spec(table_name, result, **self.RESULT_STYLE)

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

    def result_table_name(self, axis_id: int, result: Any) -> str:
        """Return a safe table name for one result.

        From RESULT_TABLE_PREFIX/RESULT_TABLE_VARIANT when the operation sets
        a prefix; otherwise the class name and whatever model or method the
        result carries.
        """
        stem = self.output_stem()
        if stem:
            # Named in the Charts tab: that name, and the series' so that several results do not collide.
            return generated_table_name(f"{stem}_{getattr(result, 'source_name', 'Series')}", fallback="Result")
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

    def next_steps(self, result: Any) -> list[next_steps.NextStep]:
        """The questions this result leads to, as "What next?" links; none by default."""
        del result
        return []

    def what_next(self, result: Any) -> str:
        """The "What next?" section of *result*'s report, empty when turned off or there is none."""
        return next_steps.section_html(self.next_steps(result)) if next_steps.enabled() else ""

    def results_document(self, subtitle: str, *sections: str) -> str:
        """The Results pane's document, the same frame for every operation: its name, *subtitle*, *sections*."""
        return report_html.document(_(self.operation_label), subtitle, *sections)

    def format_results(self, results: Sequence[Any]) -> str:
        """The Results pane, framed by results_document: the RESULT_COLUMNS table, else a section per result.

        An operation says what it has to say through result_row or result_section; the frame, the empty
        case and the subtitle are the same for all.
        """
        if not results:
            return report_html.note(_("No results."))
        if self.RESULT_COLUMNS:
            body = report_html.section(_("Results"), report_html.table(
                tuple(_(heading) for heading, _align in self.RESULT_COLUMNS),
                [self.result_row(result) for result in results],
                align=tuple(align for _heading, align in self.RESULT_COLUMNS),
                empty_message=_("No results for this selection."),
            ))
            return self.results_document(self.results_subtitle(results), body)
        return self.results_document(self.results_subtitle(results),
                                     *(self.result_section(result) for result in results))

    def how_to_read(self, results: Sequence[Any]) -> list[str]:
        """The "How to read this" box (H-04): the numbers the report hinges on, each in two sentences
        (app.plugin_api.reading_guide); none by default."""
        del results
        return []

    def results_report(self, results: Sequence[Any]) -> str:
        """The Results pane: format_results, with the "How to read this" box under its first section."""
        report = self.format_results(results)
        if not report or not results or not reading_guide.enabled():
            return report
        try:
            box = report_html.reading_guide(_("How to read this"), self.how_to_read(results))
        except Exception:  # noqa: BLE001 - the explanation is never the reason a report is not shown
            applogger.exception("%s: could not explain the report.", type(self).__name__)
            return report
        return report_html.with_reading_guide(report, box)

    def results_subtitle(self, results: Sequence[Any]) -> str:
        """The line under the operation's name: the model chosen for a table of results, else the first's."""
        if self.RESULT_COLUMNS:
            return self.current_model()
        return str(getattr(results[0], "model_name", "") or "")

    def result_row(self, result: Any) -> Sequence[Any]:
        """The cells of *result*'s row in the RESULT_COLUMNS table (HTML)."""
        raise NotImplementedError

    def result_section(self, result: Any) -> str:
        """*result*'s section: its series, model and settings (OperationResult.to_html), and its metrics."""
        series = ", ".join(getattr(result, "series", ()) or ()) or str(getattr(result, "source_name", "") or "")
        body = result.to_html() if hasattr(result, "to_html") else report_html.note(str(result))
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, Mapping) and metrics:
            body += report_html.summary_table((str(key), value) for key, value in metrics.items())
        return report_html.section(series or _("Result"), body)

    @staticmethod
    def details(values: Mapping[str, Any], *, leave_out: Sequence[str] = ()) -> str:
        """``key: value, ...`` of *values* that say something - the Detail cell of a results table."""
        return ", ".join(f"{key}: {value}" for key, value in values.items()
                         if key not in leave_out and value not in (None, ""))

    def store_cached_results(self, results: Sequence[Any]) -> None:
        """Store latest results using the existing dialog convention."""
        self._last_results = list(results)

    def write_result_table(self, table_name: str, result: Any) -> None:
        """Persist one operation result to a normal SQLite table."""
        frame = self.restore_temporal_x(self.result_to_frame(result), result)
        # An infinite value is no number to draw - a model overflowing far from its data (exp of a date in
        # seconds): written as missing, so the chart leaves it out instead of scaling to it.
        numeric = frame.select_dtypes(include="number").columns
        if len(numeric):
            frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
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
        # The series it was computed from: source_name, or the series every result names (a fit has
        # only that - its dated x was written as seconds, a chart of dates drawn at the year -215941).
        sources = {str(getattr(result, "source_name", "") or ""), *map(str, getattr(result, "series", ()) or ())}
        if not any(self._temporal_x_sources.get(source) for source in sources if source):
            return frame

        for column in ("x", "left_x", "right_x"):
            if column not in frame.columns:
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

    def _remove_series_styled(self, axis_id: int, expected: Mapping[str, Any]) -> int:
        """Delete the series of *axis_id* whose style has every key/value of *expected*; return how many."""
        removed = 0
        for row in self._repo.get_series(axis_id) or []:
            try:
                style = json.loads(str(row["style_json"] or "{}"))
            except (TypeError, ValueError):
                style = {}
            if all(style.get(key) == value for key, value in expected.items()):
                self._repo.delete_series(int(row["id"]))
                removed += 1
        return removed

    def remove_previous_generated_series(self, axis_id: int) -> int:
        """Delete generated descriptors for this operation on the selected axis."""
        removed = self._remove_series_styled(axis_id, self.generated_style_filter)
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
        if not self.DESTINATIONS:
            return selected_axis_id
        # "Draw on" (the Charts tab): a new axis or figure of the source axis' own kind.
        descriptor = self._repo.load_figure_descriptor(self._figure_id)
        chart_type = next((axis.name for axis in (descriptor.axes if descriptor else [])
                           if int(axis.id) == int(selected_axis_id)), "Scatter Plot")
        return self.resolve_destination_axis(selected_axis_id, chart_type=chart_type, title=self.operation_label)

    def create_result_axis(
        self,
        *,
        chart_type: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        options: Mapping[str, Any] | None = None,
        figure_id: int | None = None,
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
        figure_id = self._figure_id if figure_id is None else int(figure_id)
        axis_index = self._repo.next_axis_index(figure_id)

        rows, cols = self._figure_grid(figure_id)
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
    # Shared helpers
    # ------------------------------------------------------------------

    def selected_series(self) -> list[Any]:
        if self.reads_table() and not self._switching_figure:
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

    # -- Per series: the common case ------------------------------------

    def series_settings(self) -> Any:
        """Everything compute_series needs besides the data, read on the GUI thread.

        ``(model, parameters)`` for an operation with MODELS, else the parameters.
        """
        if self.MODELS:
            return self.current_model(), self.parameter_values()
        return self.parameter_values()

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
        failures: list[str] = []
        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                inputs.append((name, self.read_series(row, name)))
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                failures.append(f"{name}: {exc}")
        if not inputs and not failures:
            return None
        return SeriesJob(inputs, settings, self.compute_series, failures)

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


    def evaluate(
        self,
        on_results: Callable[[Sequence[Any]], None],
        *,
        background: bool | None = None,
        **options: Any,
    ) -> None:
        """Compute the results and hand them to *on_results*, on the GUI thread.

        In the background when *background* says so (RUN_IN_BACKGROUND when it does not), through
        run_job: the series are counted on the progress bar, Stop is polled between them.
        *on_results* is not called when the run is stopped or fails; the reason is shown instead.
        """
        if self.job_running:
            return
        job = self.prepare_job(**options)
        if job is None:
            return
        if not (self.RUN_IN_BACKGROUND if background is None else background):
            on_results(self.finish_job(job, job.run()))
            return
        self.run_job(lambda cancel: job.run(should_stop=cancel.is_set),
                     lambda outcome: on_results(self.finish_job(job, outcome)),
                     progress=getattr(job, "progress", None))  # a job that counts (SeriesJob)

    # ==================================================================
    # Where results are drawn, and Preview / Apply
    # ==================================================================

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
        wanted = (destination, str(chart_type), dict(options or {}))
        made_for = self._result_made_for
        if self._result_destination not in (None, destination) or (
                self._result_axis_id is not None and made_for not in (None, wanted)):
            # Made for another choice - another place, or another chart: a Surface Plot asked for after a
            # Preview drew a Wireframe kept the Wireframe's axis - not reused, taken back
            # (destination_changed does it at once for the place).
            self.discard_result_target()
        if destination == self.DEST_SAME_AXIS:
            return selected_axis_id

        if self._result_axis_id is None:
            self._result_destination = destination
            self._result_made_for = wanted
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
        axis_id = self._result_axis_id
        figure_id = self._result_figure_id
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
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Could not check the result target after a rollback")
            axis_ids = set()
        if axis_id is not None and int(axis_id) not in axis_ids:
            self._result_axis_id = None
            self._result_figure_id = None

    def offer_destinations(self, destinations: Sequence[str], default: str | None = None) -> None:
        """Offer only *destinations* in "Draw on": a surface, say, cannot go on the axis of a curve.

        The choice is kept while it is still offered; otherwise *default*, or the first, and the change is
        followed as one the user made.
        """
        combo = self.destination_combo
        wanted = [str(value) for value in destinations]
        if combo is None or not wanted or [combo.itemData(i) for i in range(combo.count())] == wanted:
            return
        chosen = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            for value in wanted:
                combo.addItem(self.destination_label(value), value)
            target = chosen if chosen in wanted else (default if default in wanted else wanted[0])
            combo.setCurrentIndex(max(0, combo.findData(target)))
        finally:
            combo.blockSignals(False)
        if combo.currentData() != chosen:
            if self.chart_form is not None:
                self.chart_form.refresh_visibility()
            self.destination_changed()

    def destination_changed(self) -> None:
        """Draw on changed: the figure or axis drawn for the old choice - a new figure, say - is taken back at
        once, its Preview with it, and the window drops its tab; the next Preview draws where now asked."""
        made = self._result_axis_id is not None or self._result_figure_id is not None
        if made or self._preview_active:
            self.cancel_operation_changes(refresh=False)
            self.discard_result_target()
            self.sync_preview_figure()
        self.mark_results_stale()

    def discard_result_target(self) -> None:
        """Remove an axis or figure this dialog made, when Apply never ran.

        A whole figure when that is what was made: deleting only its axis
        would leave an empty chart tab behind, which is worse than the axis it
        was meant to clean up.
        """
        if self._applied:
            return

        axis_id = self._result_axis_id
        figure_id = self._result_figure_id
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
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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

    def undo_tables(self, results: Sequence[Any]) -> list[str]:
        """Tables OK changes besides the descriptors and the result tables, in its undo step (none by default)."""
        del results
        return []

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
        return self._remove_series_styled(axis_id, self.preview_style_filter)

    def preview_table_name(self, axis_id: int, result: Any) -> str:
        """Return the table name a preview writes to.

        Derived from the final name so the two are recognisably a pair, and
        prefixed like it: a preview is even more clearly not the user's data.
        """
        base = self.result_table_name(axis_id, result)
        return generated_table_name(f"{base}_Preview", fallback="Series_Preview")

    def remove_preview_tables(self, table_names: Sequence[str] | None = None) -> None:
        names = {str(name) for name in (table_names or self._preview_table_names) if str(name).strip()}
        for table_name in names:
            self._repo.delete_table(table_name)
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
            self.sync_preview_figure()
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to emit preview refresh signal.")

    def sync_preview_figure(self) -> None:
        """Tell the window about the figure a Preview made: shown when there is one, dropped when it went."""
        figure_id = self._result_figure_id
        if figure_id is None:
            # A report figure (create_report_figure: Distribution, Is it normal?...) is in created_figure_ids.
            made = [int(f) for f in self.created_figure_ids if int(f) != int(self._figure_id)]
            figure_id = made[-1] if made else None
        if figure_id is not None and self._repo.get_figure_descriptor(int(figure_id)) is None:
            figure_id = None
        shown = self._preview_figure_shown
        if figure_id is not None and figure_id != shown:
            self._preview_figure_shown = int(figure_id)
            self.preview_drawn.emit(int(figure_id))
        elif figure_id is None and shown is not None:
            self._preview_figure_shown = None
            self.preview_drawn.emit(-1)

    def _reset_preview_state_flags(self) -> None:
        """Clear in-memory preview bookkeeping after commit or rollback."""
        self._preview_active = False
        self._preview_axis_ids.clear()
        self._preview_table_names.clear()

    def _figure_grid(self, figure_id: int) -> tuple[int, int]:
        """The (rows, columns) of *figure_id*'s subplot grid; 1 x 1 for a figure that is not there."""
        descriptor = self._repo.load_figure_descriptor(figure_id=figure_id)
        if descriptor is None:
            return 1, 1
        return max(1, int(descriptor.nrows or 1)), max(1, int(descriptor.ncols or 1))

    def _capture_current_grid(self) -> tuple[int, int] | None:
        """Return the figure grid as it was before this dialog previews.

        Preview-only operations can temporarily create extra axes.  Some repo
        paths update the figure grid separately from the axis descriptors, so a
        rollback/cleanup must restore both pieces of state.
        """
        try:
            return self._figure_grid(self._figure_id)
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to capture original figure grid.")
            return None

    def _restore_original_grid(self) -> None:
        """Restore the figure grid captured when the dialog opened."""
        if self._original_grid is None:
            return
        rows, cols = self._original_grid
        try:
            if self._figure_grid(self._figure_id) != (rows, cols):
                self._repo.set_figure_grid(self._figure_id, nrows=rows, ncols=cols)
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Failed to restore original figure grid.")

    # ------------------------------------------------------------------
    # Preview transaction
    # ------------------------------------------------------------------
    def _finalize_successful_operation(
        self,
        *,
        results: Sequence[Any],
        verb: str,
    ) -> None:
        """Refresh UI and update the results pane after Preview/Apply."""
        self.store_cached_results(results)
        self._refresh_after_preview_state_change()
        formatted = self.results_report(results) or f"{verb}: {len(results)} result(s)."
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
        body = formatted if looks_like_html(formatted) else plain_to_html(formatted)
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
            if not self.confirm_run(commit=commit):
                return finish(False)
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
                if self.job_running:
                    return False
                started: list[bool] = []

                def deliver(results: Sequence[Any]) -> None:
                    started.append(True)
                    finish(self._deliver_results(list(results), int(axis_id_value), commit=commit, inputs=inputs))

                self.evaluate(deliver, background=True)
                if not self.job_running and not started:
                    # Nothing to compute: the same answer the GUI thread gives.
                    show_message(self, "series.no_series_selected", title=self.operation_label)
                    return finish(False)
                return True
            results = list(self.compute_results())
        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            return finish(self._operation_failed(exc, commit=commit))
        return finish(self._deliver_results(results, int(axis_id_value), commit=commit, inputs=inputs))

    def _runs_in_background(self) -> bool:
        return bool(
            self.RUN_IN_BACKGROUND
            and DataOperationDialogBase.BACKGROUND_ENABLED
            and (self._computes_per_series()
                 or type(self).prepare_job is not DataOperationDialogBase.prepare_job)
        )

    def _operation_failed(self, exc: Exception, *, commit: bool) -> bool:
        self.cancel_operation_changes(refresh=True)
        action = "Apply" if commit else "Preview"
        if commit:
            applogger.critical(f"{action} failed: {exc}")
        else:
            applogger.error(f"{action} failed: {exc}")
        self.show_failure(exc)
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
            # The last Preview's savepoint goes first, for OK and for a Preview alike: an axis made or
            # synced below (resolve_target_axis_id - a new figure, Fit's extra charts) inside it was
            # rolled back a line later, its id kept - the next series on it "FOREIGN KEY constraint
            # failed", or the chart just never showed. And a snapshot cannot be taken inside a
            # transaction; OK writes everything again anyway.
            self.cancel_operation_changes(refresh=False)
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
                    + [OPERATIONS_TABLE, *self.undo_tables(results)],
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

            self.begin_preview_transaction()
            self.preview_results_to_axis(axis_id, results)
            self._preview_axis_ids.add(axis_id)
            self._preview_active = True
            self.store_cached_results(results)
            self._refresh_after_preview_state_change()
            self.publish_results(
                self.results_report(results)
                or f"Preview updated: {len(results)} result(s)."
            )
            return True

        except Exception as exc:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
            report = self.results_report(results) or ""
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
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
        inputs = inputs if inputs is not None else self.operation_inputs()
        self.record_in_history(
            operation=self.operation_label,
            parameters=inputs.get("parameters", {}),
            entries=inputs.get("entries", {}),
            sources=list(inputs.get("sources", [])),
            results=written,
            report=report,
        )

    def confirm_run(self, *, commit: bool) -> bool:
        """Whether Preview or Apply goes ahead: an operation that knows a run is heavy asks here first."""
        del commit
        return True

    def operation_succeeded(self, *, commit: bool) -> None:
        """Called once a Preview (commit False) or an Apply has gone through."""

    def preview(self) -> bool:
        """Compute and draw, to be looked at: undone by the next Preview, or by Cancel."""
        applogger.debug("Preview")
        return self._run_operation(commit=False)

    def ok(self) -> None:
        """Compute if needed, write for good and close."""
        self.apply(then=lambda applied: self.accept() if applied else None)

    def _follow_link(self, url: str) -> None:
        """A "What next?" link clicked in Results: this result is kept (OK), then the next analysis opens."""
        step = next_steps.parse_step(url)
        if step is None:
            return
        self.next_request = step
        if self._last_results:
            self.ok()
        else:
            self.reject()

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
        self.remember_window()

        def done(applied: bool) -> None:
            self._applied = self._applied or applied
            if then is not None:
                then(applied)

        return self._run_operation(commit=True, then=done)

    def discard_unapplied(self) -> None:
        """Close/Cancel: the Preview's changes rolled back, the axis or figure it made, the table's drawing."""
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        if not self._applied:
            # The axes an operation drew outside the savepoint (create_result_axis) grew the grid a row each:
            # deleted just now, the grid goes back too - else a 2x1 figure is left with one axis.
            self._restore_original_grid()
        self._discard_table_figure()
        self.sync_preview_figure()
