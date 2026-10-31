"""The base of every chart-data operation dialog (smoothing, interpolation, outlier removal, fitting...).

The window is OperationShell's (tabs on the left, results on the right, the buttons below). This class
adds the parts an operation has in common, in three mixins, one module per part:

- ui.py: the builder hooks, the layout, and the Data page (a chart's series or a table's columns)
- data.py: reading and validating the input series, computing, the result tables' formatting
- results.py: drawing the results (axes, figures, previews) and the preview transaction

Subclasses provide operation-specific controls by overriding the builder hooks, and declare what the window
shows and what it needs as the class attributes at the top of the class.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.plugin_api.analysis_base import AnalysisDialogBase
from app.plugin_api.data_operations.dialog_base.common import OperationModel
from app.plugin_api.data_operations.dialog_base.data import DataMixin
from app.plugin_api.data_operations.dialog_base.results import ResultsMixin
from app.plugin_api.data_operations.dialog_base.ui import WindowMixin
from app.plugin_api.data_operations.parameter_spec import Param
from app.plugin_api.data_operations import next_steps
from app.styles.style import icon_from_svg_source, set_doc_link
from app.utils.dialog_state import dialog_entries, set_dialog_entries
from app.utils.i18n import _
from app.widgets.axis_series_selector import AxisSeriesSelector
from app.widgets.html_results import HtmlResultsView


class DataOperationDialogBase(WindowMixin, DataMixin, ResultsMixin, AnalysisDialogBase):
    """Abstract shell for all chart-data operation dialogs.

    Subclasses should:
    - declare their Name, Description, Icon and what they show (the attributes below);
    - create model controls in ``build_model_selector`` (or set MODELS);
    - create parameter controls in ``build_parameter_selector`` (or set PARAMS);
    - optionally override ``build_results_pane`` when a QLabel is not enough;
    - implement ``compute_results`` (or ``compute_series``, through a job);
    - provide generated-series metadata through the apply hooks;
    - connect operation-specific signals in ``connect_operation_signals``.
    """
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

    def showEvent(self, event: Any) -> None:  # noqa: N802
        """Hide a results view no layout adopted.

        The base dialog creates its HTML results view as a child of itself and
        a layout normally takes it over. A dialog that shows its results some
        other way (Geometry draws a preview chart) never adds it, and a child
        of a dialog with no layout is shown at the dialog's top-left corner:
        the white rectangle there. Whatever still has the dialog itself as its
        parent when it opens is in nobody's layout, so it is hidden.
        """
        results = getattr(self, "_results_label", None)
        if results is not None and results.parent() is self:
            results.hide()
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
        self.setWindowTitle(_(title or str(getattr(type(self), "Name", "") or "Data Operation")))
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
        self._results_view = self._results_label
        self._results_view.link_requested.connect(self._follow_link)
        #: The question a "What next?" link asked for: the main window opens it once this one is closed.
        self.next_request: next_steps.NextStep | None = None
        self._preview_axis_ids: set[int] = set()
        self._preview_table_names: set[str] = set()
        self._preview_active = False
        #: True once Apply has written the results; discard_result_target and
        #: the dialogs' own cleanup leave an applied result alone. Kept when a
        #: subclass set it before calling this constructor.
        self._applied: bool = getattr(self, "_applied", False)
        #: The last run's results, and the axis or figure "Draw on" made for them (resolve_destination_axis).
        self._last_results: list[Any] = []
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
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
        # The name of the table the result is written to, built before the subclass hooks so that
        # the default can be set from them. Only with the Graphics page: a field nobody sees is not a
        # choice the operation's history should record.
        if self.SHOW_OUTPUT:
            self._output_table_edit = QLineEdit(self)
            self._output_table_edit.setToolTip(_("Name of the table the result is written to."))
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

        # Every subclass gets remembered entries and geometry for free; the
        # storage key is the class name, so two dialogs never share a slot.
        self._state_key = type(self).__name__
        # What Revert goes back to: the entries as built, before any memory.
        self._default_entries = dialog_entries(self, inputs_only=True)
        self._default_parameters = self.parameter_values()
        self.restore_window()
        # Restored with its signals blocked: the model's documentation link
        # still named the first model's ("Huber" linked to RANSAC's page).
        if getattr(self, "_doc_link", None) is not None:
            self._update_doc_link()
        if self.SELECTS_ALL_SERIES:
            self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.mark_results_stale()

    def parameter_forms(self) -> list[Any]:
        """The Parameters tab's form and the Charts tab's, whose values are kept."""
        return [form for form in (getattr(self, "_parameter_form_spec", None), self.chart_form)
                if form is not None]

    def state_extra(self) -> dict[str, Any]:
        """The table the data came from, and its columns."""
        source = getattr(self, "table_source", None)
        if source is not None and source.table():
            return {self._TABLE_SOURCE_STATE: {"table": source.table(), "x": source.x_column(),
                                               "y": source.y_columns()}}
        return {}

    def restore_state_extra(self, stored: Mapping[str, Any]) -> None:
        """The table's columns back in the Data tab."""
        source = getattr(self, "table_source", None)
        chosen = stored.get(self._TABLE_SOURCE_STATE)
        if source is not None and isinstance(chosen, dict):
            source.set_spec(str(chosen.get("table", "")), str(chosen.get("x", "")), chosen.get("y") or [])
        # Restored with its signals blocked: the page follows the radio here.
        self._show_source_page()

    def revert_entries(self) -> None:
        """Every entry back to its default, as JMP's Revert: what the window had when first opened."""
        set_dialog_entries(self, self._default_entries, quietly=False)
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.set_values(self._default_parameters)
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
