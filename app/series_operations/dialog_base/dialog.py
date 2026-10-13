"""Shared base dialog for chart-series operation panels.

The base class owns the common shell used by smoothing, interpolation, outlier
removal and fitting dialogs:

- left side: tabs with the Axis/Series selector, the model selector and the parameters
- right side: results pane and log output
- bottom action buttons

Subclasses provide operation-specific controls by overriding the builder hooks.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QSizePolicy, QWidget

from app.data.sqlite_repo import SqliteRepo
from app.widgets.axis_series_selector import AxisSeriesSelector
from app.styles.style import icon_from_svg_source, set_doc_link
from app.logs.logger import applogger
from app.utils.dialog_state import (
    dialog_entries,
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
    set_dialog_entries,
    stored_dialog_state,
)
from app.widgets.html_results import HtmlResultsView

from app.series_operations.dialog_base.hooks import BuilderHooksMixin
from app.series_operations.dialog_base.layout import LayoutMixin
from app.series_operations.dialog_base.evaluation import EvaluationMixin
from app.series_operations.dialog_base.inputs import InputsMixin
from app.series_operations.dialog_base.results import ResultsMixin
from app.series_operations.dialog_base.transaction import TransactionMixin


class SeriesOperationDialogBase(BuilderHooksMixin, LayoutMixin, EvaluationMixin, InputsMixin, ResultsMixin, TransactionMixin, QDialog):
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
