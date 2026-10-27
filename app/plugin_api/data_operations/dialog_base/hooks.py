"""The hooks a data operation fills in: its model selector, parameters, fields and sampling rate."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QWidget,
)
import numpy as np

from app.analysis.sampling import sampling_frequency
from app.data.sqlite_repo import DatabaseError
from app.styles.style import create_doc_link, CardFrame, stdSizeAndlayout
from app.logs.logger import applogger
from app.plugin_api.data_operations.parameter_form import ParameterForm
from app.plugin_api.data_operations.parameter_spec import BoolParam, FloatParam, Param, defaults
from app.utils.i18n import _

from app.plugin_api.data_operations.dialog_base.common import OperationModel


class BuilderHooksMixin:
    """The hooks a data operation fills in: its model selector, parameters, fields and sampling rate. A part of DataOperationDialogBase (see dialog.py)."""

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

    #: Where the result may be drawn, in the Charts tab's "Draw on" list (DEST_* values); empty: no
    #: choice - the operation decides (Fit draws on its data's axis). DEFAULT_DESTINATION is preselected.
    DESTINATIONS: tuple[str, ...] = ()
    DEFAULT_DESTINATION: str = ""

    def extra_tabs(self) -> list[tuple[QWidget, str]]:
        """Further (page, title) tabs after Parameters, for an operation with more to set."""
        return []

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
        context: dict[str, Any] = {}
        if getattr(self, "model_combo", None) is not None:
            context["model"] = self.current_model()
        destination = getattr(self, "destination_combo", None)
        if destination is not None:
            # "Draw on": a chart shown only while there is somewhere to draw it.
            context["destination"] = str(destination.currentData())
        return context

    def parameter_values(self) -> dict[str, Any]:
        """Return the declared parameters' current values, keyed by name.

        Falls back to the declared defaults when there is no form yet, so an
        operation can call this from ``compute_results`` during construction
        without a special case.
        """
        form = getattr(self, "_parameter_form_spec", None)
        values = defaults(self.PARAMS) if form is None else dict(form.values())
        charts = self.chart_form
        if charts is not None:
            values.update(charts.values())
        if self.DESTINATIONS:
            combo = getattr(self, "destination_combo", None)
            values["destination"] = (str(combo.currentData()) if combo is not None
                                     else self.DEFAULT_DESTINATION or self.DESTINATIONS[0])
        return values

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
        """Show the parameters (and the charts) the selected model uses: the PARAMS' visible_for rules.

        Eight operations had this same body as their own; one that builds
        its controls by hand overrides it (or calls this, then adds its own).
        """
        for form in (getattr(self, "_parameter_form_spec", None), self.chart_form):
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

    def current_axis_name(self) -> str:
        """The name of the axis picked in the Data page."""
        return self.series_selector.selected_axis_name()

    # -- Small controls, for operations that build their own forms ---------

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
