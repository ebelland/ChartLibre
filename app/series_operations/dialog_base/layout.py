"""The operation window's layout, and where its data comes from: a chart's series or a table's columns."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.widgets.table_source_selector import TableSourceSelector
from app.styles.style import (
    apply_dialog_shell,
    MARGIN_TOOLBOX_PAGE,
    CardFrame,
    TitledCard,
    create_action_button,
    stdSizeAndlayout,
)
from app.logs.logger import applogger
from app.utils.i18n import _


class LayoutMixin:
    """The operation window's layout, and where its data comes from: a chart's series or a table's columns. A part of SeriesOperationDialogBase (see dialog.py)."""

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
        pages.extend(self.extra_tabs())
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

        # The title above the box, as in every other window.
        right = TitledCard(self, _("Results"), "operationResultsCard")
        right.layout().setStretchFactor(right.card, 1)
        right.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right.card.layout().addWidget(self._results_widget, 3)

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
