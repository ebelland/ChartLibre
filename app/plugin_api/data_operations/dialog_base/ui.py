"""The operation window: the builder hooks (model, parameters, charts), the layout, and the Data page's source.

A chart's series or a table's columns: the Data page follows the radio button, and a table is drawn as a
figure of its own (_prepare_table_source) before the operation reads it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.analysis.sampling import sampling_frequency
from app.data.sqlite_repo import DatabaseError
from app.logs.logger import applogger
from app.plugin_api.data_operations.parameter_form import ParameterForm
from app.plugin_api.data_operations.parameter_spec import BoolParam, FloatParam, Param, defaults
from app.styles.style import MARGIN_TOOLBOX_PAGE, CardFrame, TitledCard, create_doc_link, stdSizeAndlayout
from app.utils.i18n import _
from app.widgets.table_source_selector import TableSourceSelector

from app.plugin_api.data_operations.dialog_base.common import (
    GENERATED_TABLE_PREFIX,
    OperationModel,
    generated_table_name,
)


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


class WindowMixin:
    """The operation window: builder hooks, layout and the Data page. A part of DataOperationDialogBase."""

    # ------------------------------------------------------------------
    # Builder hooks
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        """Create subclass widgets before build_model_selector/build_parameter_selector.

        The base constructor calls the builder hooks, so subclasses must create
        controls used by those hooks here rather than after super().__init__().
        """
        return None

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

    def extra_tabs(self) -> list[tuple[QWidget, str]]:
        """Further (page, title) tabs after Parameters, for an operation with more to set."""
        return []

    # ------------------------------------------------------------------
    # Graphics: the name of the table the result is written to
    # ------------------------------------------------------------------
    def build_output_page(self) -> QWidget:
        """The Graphics page: the name of the curve's table, and the warning under it when it lacks the prefix."""
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        card = TitledCard(page, _("Curve"))
        card.card.layout().addWidget(self._output_table_edit)
        card.card.layout().addWidget(self._output_warning)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

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
        scroll.viewport().setProperty("toolboxPage", True)
        scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # The sheet paints it; filled with the palette's window grey it was a darker box
        # inside the Settings frame - a panel in a panel.
        scroll.viewport().setAutoFillBackground(False)
        return scroll

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

        # The Axis / Series tab is left out for an operation that selects nothing.
        pages = []
        if self.SHOW_DATA:
            pages.append((self.axis_series_panel, _("Data")))
        if self.SHOW_MODEL:
            pages.append((self.model_panel, _("Model")))
        if self.SHOW_PARAMETERS:
            pages.append((self.parameters_panel, _("Parameters")))
        if self.SHOW_OUTPUT:
            pages.append((self.build_output_page(), _("Graphics")))
        pages.extend(self.extra_tabs())
        charts_tab = self.build_charts_tab() if self.SHOW_CHART else None
        if charts_tab is None and self.SHOW_CHART and self._offers_table_chart():
            # A table's chart is picked here even when the operation has no other chart to choose.
            charts_tab = CardFrame(self, "operationChartsCard")
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
            left_tabs.addTab(page, title)
        left_tabs.setCurrentIndex(data_index)
        self.left_tabs = left_tabs

        self.build_window(
            left_tabs,
            self._results_widget,
            copy=self._results_label.copy_html_to_clipboard,
            revert=self.revert_entries,
            extra_buttons=self.build_extra_action_buttons,
            results_minimum_width=380,
        )

    # ------------------------------------------------------------------
    # Data from a chart, or from a table
    # ------------------------------------------------------------------

    def _build_source_panel(self) -> QWidget:
        """The Data page: the chart's series, or a table's columns."""
        self.table_source: TableSourceSelector | None = None
        #: The figure the table's columns were drawn on, and what they were.
        self._table_figure_id: int | None = None
        self._table_figure_spec: tuple[Any, ...] | None = None
        #: The figure the dialog was opened on, to go back to.
        self._chart_figure_id = self._figure_id
        if not (self.FROM_DATA and self.SHOWS_SERIES_SELECTOR and self.SHOW_DATA):
            return self.series_selector

        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        # A table's columns are drawn as a figure of their own first (_prepare_table_source).
        layout.addLayout(self.build_source_choice(panel, from_chart=True, on_change=self._on_source_changed))

        self.table_source = TableSourceSelector(self._repo, panel, charts=tuple(self.TABLE_CHARTS))
        self._source_stack = QStackedWidget(panel)
        self._source_stack.addWidget(self.series_selector)
        self._source_stack.addWidget(self.table_source)
        layout.addWidget(self._source_stack, 1)

        self.table_source.changed.connect(self._on_table_columns_changed)
        return panel

    def reads_table(self) -> bool:
        """Whether this run takes its data from a table's columns."""
        return getattr(self, "table_source", None) is not None and self.source_table_radio.isChecked()

    def _offers_table_chart(self) -> bool:
        """Whether a table can be drawn as more than one chart, so the Charts tab must say which."""
        return self.table_source is not None and self.table_source.chart_combo.count() >= 2

    def _add_table_chart_choice(self, charts_tab: QWidget) -> None:
        """A table's chart choice (what its columns are drawn as) on the Charts tab, above Draw on."""
        if not self._offers_table_chart():
            return
        form = QFormLayout()
        stdSizeAndlayout(form)
        form.addRow(_("Chart:"), self.table_source.chart_combo)
        charts_tab.layout().insertLayout(0, form)
        self._table_chart_form = form
        self._table_chart_combo = self.table_source.chart_combo
        self._table_chart_form.setRowVisible(self._table_chart_combo, self.reads_table())

    def _show_source_page(self) -> None:
        """The Data page shows the table's columns when the run reads them, else the chart's series."""
        if getattr(self, "_source_stack", None) is not None:
            self._source_stack.setCurrentIndex(1 if self.reads_table() else 0)

    def _on_source_changed(self, *_ignored: Any) -> None:
        self._show_source_page()
        if getattr(self, "_table_chart_form", None) is not None:
            self._table_chart_form.setRowVisible(self._table_chart_combo, self.reads_table())
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
        columns = self.table_source.spec() if self.table_source is not None else None
        if columns is None:
            raise ValueError(_("Choose a table and at least one Y column."))
        assert self.table_source is not None
        chart = self.table_source.chart()
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
        figure_id = getattr(self, "_table_figure_id", None)
        self._table_figure_id, self._table_figure_spec = None, None
        if figure_id is None:
            return
        if not getattr(self, "_applied", False):
            try:
                self._repo.delete_figure(int(figure_id))
            except Exception:  # noqa: BLE001 - cleanup: must not hide the outcome it follows
                applogger.exception("Failed to discard the figure drawn from a table")
        self._use_figure(self._chart_figure_id)
