"""The base of every table operation: an analysis of a table's columns, not of a chart's series.

A data operation reads the x and y of the series it is given; a table
operation reads a whole table - several factors, numeric and categorical,
several responses - the way JMP's Fit Model does. Otherwise it works the
same way, so it reads the same: the inputs on the left (the table and its
columns cast into roles, the model, the parameters), the report on the
right, and Preview, Copy, OK and Close.

A subclass declares what it is and what it needs, as class attributes:

    Name, Description, Category, Icon   how the Analysis panel lists it
    ROLES                               the column roles it needs (ColumnRole)
    MODELS                              optional: the models it offers
    PARAMS                              optional: its parameters (Param)

and implements three methods:

    compute(frame, casting, params, model)   the calculation; runs on a worker
                                             thread, so it touches no widget
    format_results(result)                   the report, as HTML
    apply_results(result)                    what OK writes into the project

Hidden rows (``Hide`` = 1) are left out, as every chart leaves them out.
The casting is remembered per table; a table the DOE dialog made has its
design (``SqliteRepo.get_table_info(table)["doe"]``) for default_casting
to read.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.data import column_metadata
from app.data.sqlite_repo import ROW_ID, SqliteRepo
from app.logs.logger import applogger
from app.plugin_api import reading_guide
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase, OperationModel
from app.plugin_api.data_operations.parameter_form import ParameterForm
from app.plugin_api.data_operations.parameter_spec import BoolParam, Param
from app.plugin_api.operation_panels import merged_values
from app.plugin_api.data_operations import chart_source, next_steps
from app.plugin_api.data_operations.column_roles import (
    CONTINUOUS,
    ColumnCasting,
    ColumnRole,
    ColumnRolesWidget,
)
from app.styles.style import (
    stdSizeAndlayout,
)
from app.utils import report_html
from app.utils.config import get_section, set_section
from app.utils.dialog_state import (
    dialog_entries,
)
from app.utils.i18n import _

#: Columns the application keeps on every table for itself: never offered.
APPLICATION_COLUMNS: frozenset[str] = frozenset({"Hide", "Selected"})


#: What marks a table an analysis wrote (as the data operations' results): hidden from the Tables list by default.
GENERATED_PREFIX = "_"



def _hidden_by_same_axis(spec: Any, first: str, same_axis: str) -> Any:
    """*spec*, shown unless Same axis is chosen and it is not the first chart (the one that is drawn)."""
    shown_if = spec.visible_if

    def visible(values: Mapping[str, Any]) -> bool:
        if values.get("_destination") == same_axis and spec.name != first:
            return False
        return shown_if is None or bool(shown_if(values))

    return replace(spec, visible_if=visible)


class RolesOperationDialogBase(DataOperationDialogBase):
    """A data operation whose Data page casts a table's columns into roles: the same window and the same
    Preview, OK, undo and history as every data operation, the calculation on the cast columns."""

    #: A table analysis picks its data as roles (ROLES), not as a chart's series.
    DATA_PICKER: ClassVar[str] = "roles"
    Name: ClassVar[str] = ""
    Description: ClassVar[str] = ""
    Category: ClassVar[str] = ""
    #: Its place among the Analysis panel's entries: categories by their lowest Order.
    Order: ClassVar[int] = 0
    Icon: ClassVar[str] = ""

    ROLES: ClassVar[tuple[ColumnRole, ...]] = ()
    #: The roles whose columns the report's subtitle names after the table.
    SUBTITLE_ROLES: ClassVar[tuple[str, ...]] = ("response",)
    #: Roles that may not share a column: (role, other role, message); "{columns}" in the message names them.
    DISTINCT_ROLES: ClassVar[tuple[tuple[str, str, str], ...]] = ()
    MODELS: ClassVar[Mapping[str, OperationModel]] = {}
    PARAMS: ClassVar[tuple[Param, ...]] = ()
    #: The tab the declared parameters go on: one of their own by default, or
    #: one the operation built (a model's settings).
    PARAMETERS_TAB: ClassVar[str] = "Parameters"
    #: Raised when the window's layout changes, so a remembered size made for
    #: the old one is left behind (4: the data operations' window).
    LAYOUT_VERSION: ClassVar[int] = 4
    WINDOW_SIZE = (1000, 560)
    #: The model is chosen beside the table, on the Data page.
    SHOW_MODEL = False
    PENDING_RESULTS_MESSAGE = "Cast the columns into their roles, then press Preview."

    #: The Results pane's headline when the analysis fails.
    FAILURE_HEADLINE: ClassVar[str] = "Nothing was computed."

    #: The chart types the analysis draws panels of: with two or more, the Charts tab has a box for each,
    #: and the report figure leaves out the panels whose box is cleared.
    CHART_PANELS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, repo: SqliteRepo, table: str | None = None, figure_id: int | None = None,
                 parent: QWidget | None = None) -> None:
        # Read by the hooks the base's constructor calls.
        self._frame = pd.DataFrame()
        self.row_ids = pd.Series(dtype="Int64")
        self._result: Any = None
        self._result_inputs: tuple[Any, ...] | None = None
        #: OK's undo step, which every snapshot of the same OK extends.
        self._undo_entry: int | None = None
        #: Reading a chart: the name its axis' data is written under.
        self._chart_table = ""
        self._cast_table = ""
        #: A Preview's tables and figure are written inside a savepoint, rolled back by the next Preview,
        #: by Cancel and before OK writes them for good.
        self._previewing = False
        # The size is the layout's: a new layout starts from its own default
        # rather than from a size remembered for the old one.
        self._geometry_key = f"{type(self).__name__}@{self.LAYOUT_VERSION}"
        super().__init__(repo, figure_id if figure_id is not None else 0, parent=parent)
        self.current_figure_id = figure_id
        self._results.setText(_(self.PENDING_RESULTS_MESSAGE))
        self._load_tables(table)

    # ------------------------------------------------------------------
    # What a subclass implements
    # ------------------------------------------------------------------

    def compute(self, frame: pd.DataFrame, casting: ColumnCasting, params: Mapping[str, Any], model: str) -> Any:
        """The calculation, on a worker thread: no widget, no database."""
        raise NotImplementedError

    def report_body(self, result: Any) -> str:
        """The report of *result* - its sections, as HTML - under the analysis' title."""
        raise NotImplementedError

    def report_subtitle(self, result: Any) -> str:
        """The line under the title: the table, then the columns of SUBTITLE_ROLES."""
        del result
        named = " / ".join(", ".join(self.columns(role)) for role in self.SUBTITLE_ROLES if self.columns(role))
        return f"{self.table} · {named}" if named else str(self.table)

    def report_document(self, result: Any) -> str:
        """The whole report: shown in Results, and kept as the notes of the figure OK draws."""
        body = self.report_body(result)
        box = ""
        if reading_guide.enabled():
            try:
                box = report_html.reading_guide(_("How to read this"), self.how_to_read(result))
            except Exception:  # noqa: BLE001 - the explanation is never the reason a report is not shown
                applogger.exception("%s: could not explain the report.", type(self).__name__)
        what_next = next_steps.section_html(self.next_steps(result)) if next_steps.enabled() else ""
        return report_html.document(_(self.Name), self.report_subtitle(result),
                                    report_html.with_reading_guide(body, box) + what_next)

    def how_to_read(self, result: Any) -> list[str]:
        """The "How to read this" box (H-04): the two or three numbers the report hinges on, each in two
        sentences (app.plugin_api.reading_guide); none by default."""
        del result
        return []

    def next_steps(self, result: Any) -> list[next_steps.NextStep]:
        """The questions this answer leads to, listed under "What next?" as links (G-02); none by default."""
        del result
        return []

    def next_step(self, label: str, operation: str, roles: Mapping[str, list[str]],
                  kinds: Mapping[str, str] | None = None) -> next_steps.NextStep:
        """A step on this dialog's table, the columns' types as they are cast here unless *kinds* says."""
        casting = self.roles_widget.casting()
        used = {column for columns in roles.values() for column in columns}
        types = {column: kind for column, kind in casting.kinds.items() if column in used}
        return next_steps.NextStep(label=label, operation=operation, table=str(self.table),
                                   roles={key: list(value) for key, value in roles.items()},
                                   kinds={**types, **dict(kinds or {})})

    def _follow_link(self, url: str) -> None:
        """A "What next?" link clicked in Results: this answer is kept (OK), then the next one opens."""
        step = next_steps.parse_step(url)
        if step is None:
            return
        self.next_request = step
        if self._result is not None:
            self.ok()
        else:
            self.reject()

    def format_results(self, result: Any) -> str:
        """The report of *result*, as HTML (report_document unless an analysis does more)."""
        return self.report_document(result)

    def apply_results(self, result: Any) -> list[dict[str, Any]]:
        """Write *result* into the project; return what was written, for the history.

        Each entry is a dict such as ``{"table": name}`` or ``{"figure": id}``;
        draw_report makes the figure and the entry.
        """
        raise NotImplementedError

    def column(self, role: str) -> str:
        """The one column cast in *role* ("" when none is)."""
        columns = self.columns(role)
        return columns[0] if columns else ""

    def columns(self, role: str) -> list[str]:
        """The columns cast in *role*."""
        return self.roles_widget.casting().columns(role)

    def build_extra_inputs(self, layout: QVBoxLayout) -> None:
        """Add the operation's own inputs under the roles (a model's effects, say); its parameters follow."""

    def default_casting(self, table: str, frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Roles and types for a table met for the first time: (roles, kinds). None by default."""
        del table, frame
        return {}, {}

    def chart_casting(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        """A chart's axis read as a table: its series are the response, its x the role after it - as drawn."""
        columns = [str(column) for column in frame.columns]
        response = next((role for role in self.ROLES if role.key == "response"), None)
        other = next((role for role in self.ROLES if role.key != "response"), None)
        if response is None or len(columns) < 2:
            return {}
        roles = {response.key: columns[1:][: response.maximum] if response.maximum else columns[1:]}
        if other is not None:
            roles[other.key] = columns[:1]
        return roles

    def cast_explanatory(self, columns: list[str]) -> None:
        """Cast the guided analysis's X: into this analysis's role after the response.

        Factors, groups, covariates, a time column - as many as the role
        takes. An analysis whose X are not a role (Fit Model's effects)
        overrides this.
        """
        role = next((role for role in self.ROLES if role.key != "response"), None)
        if role is None:
            return
        roles = dict(self.roles_widget.casting().roles)
        roles[role.key] = list(columns)[: role.maximum] if role.maximum else list(columns)
        self.roles_widget.set_casting(roles)

    def inputs_changed(self) -> None:
        """The table, the casting, the model or a parameter changed: the report is stale."""
        if self._result is not None:
            self._results.setText(_("The inputs changed: press Preview to compute again."))
        self._result = None
        # A different Y or model offers other charts (chart_context).
        for form in self.parameter_forms():
            form.refresh_visibility()

    # ------------------------------------------------------------------
    # The window
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        """The Data page, as JMP's launch window: where the columns come from, the table (and the model), its
        columns, and the boxes of the roles; then what the operation adds on pages of its own."""
        self._results = self._results_label
        self._inputs_panel = QWidget(self)
        inputs_layout = QVBoxLayout(self._inputs_panel)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        self.roles_widget = ColumnRolesWidget(self.ROLES, self._inputs_panel)
        self.roles_widget.changed.connect(self.inputs_changed)
        self.roles_widget.column_info_requested.connect(self._open_column_info)
        inputs_layout.addWidget(self.roles_widget, 1)
        # The table at the top of the columns' frame: the columns are its.
        table_row = QFormLayout()
        stdSizeAndlayout(table_row)
        self._table_combo = QComboBox(self._inputs_panel)
        self._table_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        # Short: the column is narrow, and the label must keep its room.
        self._table_combo.setMinimumContentsLength(3)
        self._table_combo.currentIndexChanged.connect(self._on_table_changed)
        table_label = self._table_label = QLabel(_("Table:"), self._inputs_panel)
        table_label.setMinimumWidth(table_label.sizeHint().width())
        table_row.addRow(table_label, self._table_combo)
        if self.MODELS:
            for key in self.MODELS:
                self.model_combo.addItem(_(key), key)
            table_row.addRow(QLabel(_("Model:"), self._inputs_panel), self.model_combo)
        self.roles_widget.columns_card_layout.insertLayout(0, table_row)
        # Above it, where the columns come from: a table, or a chart's axis read as one (chart_source).
        self.roles_widget.columns_card_layout.insertLayout(
            0, self.build_source_choice(self._inputs_panel, from_chart=False,
                                        on_change=lambda _chart: self._load_tables(None)))
        self._inputs_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def build_model_selector(self) -> QWidget:
        """No Model page: the model is chosen beside the table."""
        return QWidget(self)

    def build_parameter_selector(self) -> QWidget:
        """The operation's own inputs (build_extra_inputs), then its parameters, on PARAMETERS_TAB's page."""
        self.build_extra_inputs(self.roles_widget.side_layout)
        parameters_only = tuple(spec for spec in self.PARAMS if spec.tab != "charts")
        if parameters_only:
            form = self._parameter_form_spec = ParameterForm(parameters_only, self, on_change=self.inputs_changed,
                                                             context=self.chart_context)
            self._parameter_form = form.layout
            self.roles_widget.add_tab(_(self.PARAMETERS_TAB)).addWidget(form.widget)
        self.roles_widget.finish_tabs()
        return self.roles_widget.take_page(_(self.PARAMETERS_TAB)) or QWidget(self)

    def _build_source_panel(self) -> QWidget:
        """The Data page: the table's columns cast into roles."""
        self.table_source = None
        self._table_figure_id, self._table_figure_spec = None, None
        self._chart_figure_id = self._figure_id
        # The engine's series selector is not on this page: left a child of the window, it was drawn over
        # its top-left corner.
        self.series_selector.hide()
        # Scrolled: with several roles the page is taller than a laptop's window.
        return self._scrolled(self._inputs_panel, self)

    def input_pages(self) -> list[tuple[QWidget, str]]:
        """Data, the operation's own pages (a model's effects), then the parameters' page."""
        pages = [(self.axis_series_panel, _("Data"))] if self.SHOW_DATA else []
        pages.extend(self.roles_widget.pages())
        if self.SHOW_PARAMETERS and self.parameters_panel.layout() is not None:
            pages.append((self.parameters_panel, _(self.PARAMETERS_TAB)))
        return pages

    # ------------------------------------------------------------------
    # The table and its columns
    # ------------------------------------------------------------------

    def _load_tables(self, preferred: str | None) -> None:
        """The tables - or, reading a chart, the current figure's axes ("Figure · axis")."""
        combo = self._table_combo
        combo.blockSignals(True)
        combo.clear()
        if self.reads_chart():
            self._table_label.setText(_("Chart:"))
            for label, data in chart_source.chart_axes(self._repo, self.current_figure_id):
                combo.addItem(label, data)
        else:
            self._table_label.setText(_("Table:"))
            names = [str(name) for name in self._repo.list_user_tables()["Table"]]
            combo.addItems(names)
            if preferred and preferred in names:
                combo.setCurrentIndex(names.index(preferred))
        combo.blockSignals(False)
        self._on_table_changed()

    @property
    def table(self) -> str:
        """The table read - or, reading a chart, the table its axis is written as (_write_chart_source)."""
        if self.reads_chart():
            return self._chart_table
        return self._table_combo.currentText()

    def source_frame(self) -> pd.DataFrame:
        """What the analysis reads: the table's visible rows, or the chosen axis' series (chart_source)."""
        if self.reads_chart():
            chosen = self._table_combo.currentData()
            return chart_source.axis_frame(self._repo, chosen[1]) if chosen else pd.DataFrame()
        return self.table_frame(self.table) if self.table else pd.DataFrame()

    def _write_chart_source(self) -> None:
        """Reading a chart: its axis written as a table, which the analysis reads and writes beside.

        In OK's undo step, and rolled back with a Preview. Then the analysis is as on any table: its charts
        select from it, its results are named after it, and written beside it.
        """
        if not self.reads_chart() or not self.table:
            return
        self.snapshot_for_undo([self.table])
        self._repo.import_dataframe(self._frame, table_name=self.table, normalize_columns=False)
        ids = self._repo.query_df(f"SELECT * FROM {self.quoted(self.table)}", row_id=True)
        self.row_ids = ids[ROW_ID].astype("Int64") if ROW_ID in ids.columns else pd.Series(dtype="Int64")

    def table_frame(self, table: str) -> pd.DataFrame:
        """*table*'s visible rows, without the application's own columns.

        Each row's id (R-07 ``_id``) is kept aside in ``row_ids``, in the
        frame's order: what an analysis writes back into the table's own rows
        (a cluster, a prediction) finds them by it.
        """
        frame = self._repo.query_df(f"SELECT * FROM {self.quoted(table)}", row_id=True)
        if "Hide" in frame.columns:
            frame = frame[pd.to_numeric(frame["Hide"], errors="coerce").fillna(0) == 0]
        frame = frame.reset_index(drop=True)
        self.row_ids = frame[ROW_ID].astype("Int64") if ROW_ID in frame.columns else pd.Series(dtype="Int64")
        frame = frame.drop(columns=[c for c in frame.columns if c in APPLICATION_COLUMNS or c == ROW_ID])
        # Column Info (R-08): missing codes read as empty, value labels in place of their codes.
        return column_metadata.apply_to_frame(frame, column_metadata.column_info(self._repo, table))

    def _open_column_info(self, column: str) -> None:
        """Column Info (R-08) from the column list: the description, then the table is read again.

        A type forced there, a missing-value code or a value label changes what the
        analysis sees, so the frame and the column types are reloaded.
        """
        from app.dialogs.column_info_dialog import ColumnInfoDialog

        table = self.table
        if not table or self.reads_chart():
            return  # a chart's data is a table only once the analysis writes it
        if ColumnInfoDialog(self._repo, table, column, self).exec():
            self._on_table_changed()

    def _on_table_changed(self, *_args: Any) -> None:
        self._remember_casting()
        chosen = self._table_combo.currentData() if self.reads_chart() else None
        # A free name now, the same through Preview and OK: what the axis' data is written as.
        self._chart_table = self._repo.free_table_name(chart_source.axis_table_name(*chosen)) if chosen else ""
        table = self.table
        self._frame = self.source_frame()
        remembered = get_section(self._state_key).get("casting", {}).get(table) if table else None
        # Remembered with every role empty is nothing remembered: the defaults are better than no columns.
        if remembered and any(remembered.get("roles", {}).values()):
            roles, kinds = remembered.get("roles", {}), remembered.get("kinds", {})
        else:
            roles, kinds = self.default_casting(table, self._frame) if table else ({}, {})
            if chosen and not roles:
                roles = self.chart_casting(self._frame)
        declared = {str(row[1]): str(row[2] or "") for row in self._repo.table_info(table)} if table else {}
        # A modelling type forced in Column Info wins over the one remembered or guessed.
        forced = column_metadata.forced_kinds(column_metadata.column_info(self._repo, table)) if table else {}
        self.roles_widget.set_columns(self._frame, {**dict(kinds), **forced}, declared)
        self.roles_widget.set_casting(roles)
        self._cast_table = table
        self.inputs_changed()

    def _remember_casting(self) -> None:
        table = getattr(self, "_cast_table", "")
        if not table:
            return
        section = get_section(self._state_key)
        casting = self.roles_widget.casting()
        stored = dict(section.get("casting", {}))
        stored[table] = {"roles": casting.roles, "kinds": casting.kinds}
        set_section(self._state_key, {**section, "casting": stored})

    def doe_design(self, table: str) -> dict[str, Any] | None:
        """The design the DOE dialog recorded with *table*, if it made it."""
        design = self._repo.get_table_info(table).get("doe") if table else None
        return design if isinstance(design, dict) else None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def parameter_values(self) -> dict[str, Any]:
        """The parameters and the Charts tab's choices."""
        return merged_values(self.parameter_forms())

    def problems(self) -> list[str]:
        """Why the operation cannot run as it stands; empty when it can."""
        if not self.table:
            if self.reads_chart():
                return [_("The current figure draws no series: open the analysis on a chart, or read a table.")]
            return [_("Choose a table.")]
        if self._frame.empty:
            return [_("The table has no visible rows.")]
        found = self.roles_widget.problems()
        if found:
            return found
        for role, other, message in self.DISTINCT_ROLES:
            overlap = set(self.columns(role)) & set(self.columns(other))
            if overlap:
                return [_(message).format(columns=", ".join(sorted(overlap)))]
        # A chart destination with every chart cleared would draw an empty window: say so instead.
        if self.CHART_PANELS and self.draw_on() != self.DEST_NONE \
                and not any(self.chart_enabled(chart) for chart in self.CHART_PANELS):
            return [_("No chart is ticked in the Charts tab: tick one, or choose not to draw a chart.")]
        return []

    def _inputs(self) -> tuple[Any, ...]:
        casting = self.roles_widget.casting()
        return (self.table, repr(casting.roles), repr(casting.kinds), repr(self.parameter_values()), self.current_model())

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _run_operation(self, *, commit: bool, then: Callable[[bool], None] | None = None) -> bool:
        """Preview or OK (commit): compute on the cast columns - again only when the inputs changed since the
        last run - then draw inside the preview's savepoint, or write for good with OK's undo step.

        *then* is told whether it went through, once the calculation is back.
        """
        def finish(succeeded: bool) -> bool:
            if then is not None:
                then(succeeded)
            return succeeded

        problems = self.problems()
        if problems:
            self._results.setText("\n".join(problems))
            return finish(False)
        if commit and self._result is not None and self._result_inputs == self._inputs():
            return finish(self._apply(self._result))
        frame, casting = self._frame.copy(), self.roles_widget.casting()
        params, model, inputs = self.parameter_values(), self.current_model(), self._inputs()

        def done(result: Any) -> None:
            self._result, self._result_inputs = result, inputs
            self.show_results(self.format_results(result))
            finish(self._apply(result) if commit else self._draw_preview(result))

        # compute() takes no cancel argument (its signature is the plugins' contract): a long compute
        # polls stop_requested() and raises app.analysis.Stopped.
        return self.run_job(lambda _cancel: self.compute(frame, casting, params, model), done)

    def _draw_preview(self, result: Any) -> bool:
        """Write the result's tables and figure inside a savepoint, and have the window show them."""
        self._discard_preview()
        try:
            self.begin_preview_transaction(f"preview_{self._state_key}")
            self._previewing = True
            self.created_figure_ids = []
            self._write_chart_source()
            self.apply_results(result)
        except Exception as exc:  # noqa: BLE001 - shown like a failed run; the preview is undone
            self._previewing = False
            self._preview_active = True
            self._discard_preview()
            applogger.exception("%s: the preview could not be drawn", self.Name)
            self.show_failure(exc)
            return False
        self._previewing = False
        self._preview_active = True
        self.preview_drawn.emit(self.created_figure_ids[0] if self.created_figure_ids else -1)
        return True

    def _discard_preview(self) -> None:
        """Roll the Preview's tables and figure back, if there is one; the window drops them."""
        if not self._preview_active:
            return
        self._preview_active = False
        self.rollback_preview_transaction()
        self.created_figure_ids = []
        self.preview_drawn.emit(-1)

    def busy_widgets(self) -> list[QWidget]:
        """Off while a calculation runs: the buttons, and the inputs it was started with."""
        return [self.preview_button, self.apply_button, self.left_tabs]

    def _apply(self, result: Any) -> bool:
        """OK: the preview's drawing goes, and the result is written again for good, with its undo step."""
        self._discard_preview()
        try:
            self.created_figure_ids = []
            self._undo_entry = None
            self._write_chart_source()
            written = self.apply_results(result)
        except Exception as exc:  # noqa: BLE001 - reported, the dialog stays open
            applogger.exception("%s could not be written", self.Name)
            self.show_failure(exc, headline=_("Nothing was written."))
            return False
        self._record(written, result)
        self.applied.emit()
        return True

    def _record(self, written: Sequence[Mapping[str, Any]], result: Any) -> None:
        """The OK in the project's history, as every data operation's (OperationShell.record_in_history)."""
        casting = self.roles_widget.casting()
        self.record_in_history(
            operation=self.Name,
            parameters={**self.parameter_values(), "model": self.current_model()},
            entries=dialog_entries(self, inputs_only=True),
            sources=[{"table": self.table, "roles": casting.roles, "kinds": casting.kinds}],
            results=written,
            report=self.format_results(result),
        )

    def mark_results_stale(self, *_ignored: Any) -> None:
        """A choice changed (Draw on, the model): as any input."""
        self.inputs_changed()

    # ------------------------------------------------------------------
    # Results as a figure: charts of the tables written, the report in its notes
    # ------------------------------------------------------------------

    @staticmethod
    def quoted(name: str) -> str:
        """*name* as an SQL identifier."""
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def literal(text: str) -> str:
        """*text* as an SQL string."""
        return "'" + str(text).replace("'", "''") + "'"

    @staticmethod
    def report_rows(frame: pd.DataFrame, *, p_columns: Sequence[str] = (), digits: int = 6) -> list[list[str]]:
        return report_html.frame_rows(frame, p_columns=p_columns, digits=digits)

    def report_table(self, frame: pd.DataFrame, *, p_columns: Sequence[str] = (), digits: int = 6) -> str:
        """*frame* as a report table, its column names translated."""
        return report_html.frame_table(frame, _, p_columns=p_columns, digits=digits)

    def snapshot_for_undo(self, tables: Sequence[str]) -> None:
        """OK's undo step: *tables* as they were, before the analysis writes them ("<analysis> on '<table>'").

        Not for a Preview: it is rolled back, and OK writes it again with its step - and a snapshot cannot be
        taken inside the preview's savepoint anyway.
        """
        if not self._previewing:
            # One step for all: the chart's data written as a table, the results and the figure.
            entry = self._repo.snapshot_for_undo(list(tables), label=f"{self.Name} on '{self.table}'",
                                                 entry_id=self._undo_entry)
            self._undo_entry = entry if entry is not None else self._undo_entry

    def result_table_name(self, key: str) -> str:
        """A free name for a table the analysis writes: "_penguins_pls_vip".

        The leading underscore marks it generated, as the data operations'
        results are: the Tables list leaves it out until Show generated tables.
        """
        return self._repo.free_table_name(f"{GENERATED_PREFIX}{self.table}_{key}")

    def write_result_tables(self, frames: Mapping[str, pd.DataFrame], *, also: Sequence[str] = ()) -> dict[str, str]:
        """Each frame as a new table named after the source and its key ("_penguins_pls_vip"), one undo step.

        Returns {key: the name it got} - a free one, so an earlier run's tables
        stay. The figure's own tables are in the same step: Undo takes the
        tables and the figure the analysis draws away together.
        """
        repo = self._repo
        # A table with no columns (Tukey on groups of one value each) is no table: SQLite refuses "CREATE TABLE t ()".
        frames = {key: frame for key, frame in frames.items() if len(frame.columns)}
        names = {key: self.result_table_name(key) for key in frames}
        # *also*: tables the analysis changes besides, in the same undo step (the source's ClusterId).
        self.snapshot_for_undo([*names.values(), *also, *repo.DESCRIPTOR_TABLES])
        for key, frame in frames.items():
            repo.import_dataframe(frame, table_name=names[key], normalize_columns=False)
        return names

    def series_select(self, table: str, roles: Mapping[str, str], where: str = "") -> tuple[str, dict[str, str]]:
        """A series query naming each role's column as the role, the way every series reads its data."""
        columns = ", ".join(f"{self.quoted(column)} AS {self.quoted(role)}" for role, column in roles.items())
        sql = f"SELECT {columns} FROM {self.quoted(table)}" + (f" WHERE {where}" if where else "")
        return sql, {role: role for role in roles}

    def grouped_series(self, table: str, x: str, y: str, by: str = "", *, max_groups: int = 20) -> list[tuple]:
        """The rows of *table* as one scatter series, or one per value of *by* (up to *max_groups*)."""
        if by and by in self._frame.columns:
            levels = list(pd.unique(self._frame[by].dropna()))
            if 1 < len(levels) <= max_groups:
                series = []
                for level in sorted(levels, key=str):
                    sql, roles = self.series_select(table, {"x": x, "y": y},
                                                    where=f"{self.quoted(by)} = {self.literal(str(level))}")
                    series.append((str(level), sql, roles, {}))
                return series
        sql, roles = self.series_select(table, {"x": x, "y": y})
        return [(_("Rows"), sql, roles, {})]

    def draw_report(self, result: Any, title: str, panels: Sequence[tuple], *,
                    written: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
        """The report figure of *result*: *panels*, with the report in its notes, named "<analysis> · <title>".

        Returns the history entries: *written* (the tables), then the figure.
        """
        figure_id = self.create_report_figure(f"{_(self.Name)} · {title}", self.report_document(result), panels)
        figure = [] if figure_id is None else [{"figure": figure_id, "name": title}]
        return [*(dict(entry) for entry in written), *figure]

    # ------------------------------------------------------------------
    # Leaving
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Charts tab
    # ------------------------------------------------------------------

    def build_charts_tab(self) -> QWidget | None:
        """The Charts tab: where the report is drawn - a new figure, or new axes in the current one - then the
        charts: the PARAMS marked tab="charts", or a box per chart type (CHART_PANELS) when it draws several."""
        charts = tuple(spec for spec in self.PARAMS if spec.tab == "charts")
        if not charts and len(self.CHART_PANELS) > 1:
            # Only the charts the columns cast can draw are offered (charts_drawn): a box for the others
            # promised a chart that never came.
            charts = tuple(BoolParam(chart, chart, default_value=True, tab="charts",
                                     visible_if=lambda values, chart=chart: values.get("_charts") is None
                                     or chart in values["_charts"])
                           for chart in self.CHART_PANELS)
        # Same axis draws the first chart only: while it is chosen, the other charts are hidden.
        if charts:
            first = charts[0].name
            charts = tuple(_hidden_by_same_axis(spec, first, self.DEST_SAME_AXIS) for spec in charts)
        # A table analysis has no axis of its own: the choices of where its report is drawn.
        destinations = (self.DEST_SAME_AXIS, self.DEST_NEW_AXIS, self.DEST_NEW_FIGURE, self.DEST_NONE)
        return self.build_charts_page(charts, destinations=destinations, default_destination=self.DEST_NEW_FIGURE,
                                      on_change=self.inputs_changed, context=self.chart_context)

    def chart_context(self) -> dict[str, Any]:
        """What a chart's (or a parameter's) visible_for may read besides the form: the model, and the type
        of the Y - "continuous", "categorical", or "any" while none is cast - so that only the charts that
        can be drawn are offered."""
        casting = self.roles_widget.casting()
        response = self.column("response")
        kind = casting.kinds.get(response) if response else None
        y = "any" if kind is None else ("continuous" if kind == CONTINUOUS else "categorical")
        return {"model": self.current_model(), "y": y, "_charts": self.charts_drawn()}

    def charts_drawn(self) -> set[str] | None:
        """The CHART_PANELS the columns as cast can draw - only those are offered in the Charts tab; None
        (the default) for all of them."""
        return None

    def geometry_key(self) -> str:
        """The size is the layout's: a new layout starts from its own default, not from the old one's size."""
        return self._geometry_key

    def parameter_context(self) -> Mapping[str, Any]:
        return self.chart_context()

    def remember_window(self) -> None:
        """The shell's memory, and the columns' roles cast on this table."""
        self._remember_casting()
        super().remember_window()

    def discard_unapplied(self) -> None:
        self._discard_preview()

