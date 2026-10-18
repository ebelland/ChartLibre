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

import html

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pandas as pd
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app import APP_VERSION
from app.data.sqlite_repo import SqliteRepo
from app.analysis import Stopped
from app.logs.logger import applogger
from app.plugin_api.data_operations.dialog_base import OperationModel
from app.plugin_api.data_operations.parameter_form import ParameterForm
from app.plugin_api.data_operations.parameter_spec import Param
from app.styles.style import (
    TitledCard,
    apply_dialog_shell,
    create_action_button,
    icon_from_svg_source,
    stdSizeAndlayout,
)
from app.plugin_api.table_operations.column_roles import ColumnCasting, ColumnRole, ColumnRolesWidget
from app.utils.background import BackgroundTask, run_in_background
from app.utils.config import get_section, set_section
from app.utils.dialog_state import (
    dialog_entries,
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
)
from app.utils import report_html
from app.utils.i18n import _
from app.widgets.html_results import HtmlResultsView

#: Columns the application keeps on every table for itself: never offered.
APPLICATION_COLUMNS: frozenset[str] = frozenset({"Hide", "Selected"})


#: What marks a table an analysis wrote (as the data operations' results): hidden from the Tables list by default.
GENERATED_PREFIX = "_"


class TableOperationDialogBase(QDialog):
    """Inputs on the left, the report on the right; Preview, Copy, OK, Close."""

    Name: ClassVar[str] = ""
    Description: ClassVar[str] = ""
    Category: ClassVar[str] = ""
    #: Its place among the Analysis panel's entries: categories by their lowest Order.
    Order: ClassVar[int] = 0
    Icon: ClassVar[str] = ""

    ROLES: ClassVar[tuple[ColumnRole, ...]] = ()
    MODELS: ClassVar[Mapping[str, OperationModel]] = {}
    PARAMS: ClassVar[tuple[Param, ...]] = ()
    #: The tab the declared parameters go on: one of their own by default, or
    #: one the operation built (a model's settings).
    PARAMETERS_TAB: ClassVar[str] = "Parameters"
    #: Raised when the window's layout changes, so a remembered size made for
    #: the old one is left behind (2: the inputs in tabs, 880 x 460).
    LAYOUT_VERSION: ClassVar[int] = 3

    #: Something was written into the project (an OK): the window refreshes.
    applied = Signal()

    def __init__(self, *, repo: SqliteRepo, table: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self._frame = pd.DataFrame()
        self._result: Any = None
        self._result_inputs: tuple[Any, ...] | None = None
        self._task: BackgroundTask | None = None
        self._state_key = type(self).__name__
        # The size is the layout's: a new layout starts from its own default
        # rather than from a size remembered for the old one.
        self._geometry_key = f"{self._state_key}@{self.LAYOUT_VERSION}"
        self.setWindowTitle(_(self.Name))
        if self.Icon:
            self.setWindowIcon(icon_from_svg_source(self.Icon, size=32))
        self._build_ui()
        restore_window_geometry(self, self._geometry_key)
        restore_dialog_state(self, self._state_key)
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
        """The line under the title: the table, by default; an analysis adds its columns."""
        del result
        return str(self.table)

    def report_document(self, result: Any) -> str:
        """The whole report: shown in Results, and kept as the notes of the figure OK draws."""
        return report_html.document(_(self.Name), self.report_subtitle(result), self.report_body(result))

    def format_results(self, result: Any) -> str:
        """The report of *result*, as HTML (report_document unless an analysis does more)."""
        return self.report_document(result)

    def apply_results(self, result: Any) -> list[dict[str, Any]]:
        """Write *result* into the project; return what was written, for the history.

        Each entry is a dict such as ``{"table": name}`` or ``{"figure": id}``.
        """
        raise NotImplementedError

    def build_extra_inputs(self, layout: QVBoxLayout) -> None:
        """Add the operation's own inputs under the roles (a model's effects, say); its parameters follow."""

    def default_casting(self, table: str, frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Roles and types for a table met for the first time: (roles, kinds). None by default."""
        del table, frame
        return {}, {}

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

    # ------------------------------------------------------------------
    # The window
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # One page, as JMP's launch window: the table, its columns on the left,
        # the roles - and whatever the operation adds, a model's effects, its
        # parameters - in a column beside them. Tabs hid half the inputs and
        # asked for the factors twice.
        self._inputs_panel = QWidget(self)
        inputs_layout = QVBoxLayout(self._inputs_panel)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        self.roles_widget = ColumnRolesWidget(self.ROLES, self._inputs_panel)
        self.roles_widget.changed.connect(self.inputs_changed)
        inputs_layout.addWidget(self.roles_widget, 1)
        # The table at the top of the columns' frame: the columns are its.
        table_row = QFormLayout()
        stdSizeAndlayout(table_row)
        self._table_combo = QComboBox(self._inputs_panel)
        self._table_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        # Short: the column is narrow, and the label must keep its room.
        self._table_combo.setMinimumContentsLength(3)
        self._table_combo.currentIndexChanged.connect(self._on_table_changed)
        table_label = QLabel(_("Table:"), self._inputs_panel)
        table_label.setMinimumWidth(table_label.sizeHint().width())
        table_row.addRow(table_label, self._table_combo)
        if self.MODELS:
            self.model_combo = QComboBox(self._inputs_panel)
            for key in self.MODELS:
                self.model_combo.addItem(_(key), key)
            self.model_combo.currentIndexChanged.connect(lambda _i: self.inputs_changed())
            table_row.addRow(QLabel(_("Model:"), self._inputs_panel), self.model_combo)
        self.roles_widget.columns_card_layout.insertLayout(0, table_row)
        self.build_extra_inputs(self.roles_widget.side_layout)
        self._parameter_form: ParameterForm | None = None
        if self.PARAMS:
            self._parameter_form = ParameterForm(self.PARAMS, self, on_change=self.inputs_changed)
            parameters = TitledCard(self._inputs_panel, _("Parameters"))
            parameters.title_label.hide()  # the tab says it, as in the data operations
            parameters.card.layout().addWidget(self._parameter_form.widget)
            self.roles_widget.add_tab(_(self.PARAMETERS_TAB)).addWidget(parameters)
        self.roles_widget.finish_tabs()
        self.roles_widget.tabs.setCurrentIndex(0)
        # The standard ground under the frames, not the window's grey.
        self._inputs_panel.setProperty("toolboxPage", True)
        self._inputs_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # A standard frame: the title above the box, as the columns' and the roles'.
        right = TitledCard(self, _("Results"), "tableOperationResultsCard")
        right.layout().setStretchFactor(right.card, 1)
        self._results = HtmlResultsView(self)
        self._results.setText(_("Cast the columns into their roles, then press Preview."))
        right.card.layout().addWidget(self._results)
        right.setMinimumWidth(260)

        # In a titled frame like Results beside it, as the data operations'
        # Settings (B-14): one box with the title above it, its cards
        # flattened by the sheets (#operationSettingsCard) - it was a square
        # block with no title, stuck to Results (B-15).
        left = TitledCard(self, _("Settings"), "operationSettingsCard", margins=(6, 6, 6, 6))
        left.layout().setStretchFactor(left.card, 1)
        left.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.card.layout().addWidget(self._inputs_panel, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self._inputs_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        buttons = QHBoxLayout()
        stdSizeAndlayout(buttons)
        self.preview_button = create_action_button(parent=self, action_id="preview", action=self.preview, layout=buttons)
        self.stop_button = create_action_button(parent=self, action_id="operation_stop", action=self.stop, layout=buttons)
        self.stop_button.setVisible(False)
        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(160)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        buttons.addWidget(self._progress)
        buttons.addStretch(1)
        create_action_button(parent=self, action_id="copy", action=self._results.copy_html_to_clipboard, layout=buttons)
        self.apply_button = create_action_button(parent=self, action_id="apply", action=self.ok, layout=buttons)
        create_action_button(parent=self, action_id="close", action=self.reject, layout=buttons)

        root = QVBoxLayout(self)
        # Small enough for a laptop: the inputs need about 520 px, the report
        # takes the rest, and screen_fit shrinks it further on a smaller screen.
        # Wider since the roles sit under the columns: three columns of room.
        apply_dialog_shell(self, root, size=QSize(1000, 560))
        root.addWidget(splitter, 1)
        root.addLayout(buttons)

    # ------------------------------------------------------------------
    # The table and its columns
    # ------------------------------------------------------------------

    def _load_tables(self, preferred: str | None) -> None:
        names = [str(name) for name in self._repo.list_user_tables()["Table"]]
        self._table_combo.blockSignals(True)
        self._table_combo.clear()
        self._table_combo.addItems(names)
        self._table_combo.blockSignals(False)
        if preferred and preferred in names:
            self._table_combo.setCurrentIndex(names.index(preferred))
        self._on_table_changed()

    @property
    def table(self) -> str:
        return self._table_combo.currentText()

    def table_frame(self, table: str) -> pd.DataFrame:
        """*table*'s visible rows, without the application's own columns."""
        frame = self._repo.table_frame(table)
        if "Hide" in frame.columns:
            frame = frame[pd.to_numeric(frame["Hide"], errors="coerce").fillna(0) == 0]
        return frame.drop(columns=[c for c in frame.columns if c in APPLICATION_COLUMNS]).reset_index(drop=True)

    def _on_table_changed(self, *_args: Any) -> None:
        self._remember_casting()
        table = self.table
        self._frame = self.table_frame(table) if table else pd.DataFrame()
        remembered = get_section(self._state_key).get("casting", {}).get(table) if table else None
        if remembered:
            roles, kinds = remembered.get("roles", {}), remembered.get("kinds", {})
        else:
            roles, kinds = self.default_casting(table, self._frame) if table else ({}, {})
        declared = {str(row[1]): str(row[2] or "") for row in self._repo.table_info(table)} if table else {}
        self.roles_widget.set_columns(self._frame, kinds, declared)
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
        return dict(self._parameter_form.values()) if self._parameter_form is not None else {}

    def current_model(self) -> str:
        combo = getattr(self, "model_combo", None)
        return str(combo.currentData()) if combo is not None else ""

    def problems(self) -> list[str]:
        """Why the operation cannot run as it stands; empty when it can."""
        if not self.table:
            return [_("Choose a table.")]
        if self._frame.empty:
            return [_("The table has no visible rows.")]
        return self.roles_widget.problems()

    def _inputs(self) -> tuple[Any, ...]:
        casting = self.roles_widget.casting()
        return (self.table, repr(casting.roles), repr(casting.kinds), repr(self.parameter_values()), self.current_model())

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def preview(self) -> None:
        self._run(then=None)

    def ok(self) -> None:
        if self._result is not None and self._result_inputs == self._inputs():
            self._apply(self._result)
        else:
            self._run(then=self._apply)

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    def stop_requested(self) -> bool:
        """True once Stop was pressed (or the window closed) during compute(); poll it in long loops."""
        event = getattr(self, "_stop_event", None)
        return event is not None and event.is_set()

    def _run(self, then: Any) -> None:
        problems = self.problems()
        if problems:
            self._results.setText("\n".join(problems))
            return
        if self._task is not None:
            return
        frame, casting = self._frame.copy(), self.roles_widget.casting()
        params, model, inputs = self.parameter_values(), self.current_model(), self._inputs()

        def finished(result: Any) -> None:
            # Stop pressed too late for compute() to notice: the result is dropped all the same.
            stopped = self._task is not None and self._task.cancel_event.is_set()
            self._set_busy(False)
            if stopped:
                failed(Stopped())
                return
            self._result, self._result_inputs = result, inputs
            try:
                self._results.setContent(self.format_results(result))
            except Exception as exc:  # noqa: BLE001 - shown like a failed run
                failed(exc)
                return
            if then is not None:
                then(result)

        def failed(error: BaseException) -> None:
            self._set_busy(False)
            if isinstance(error, Stopped):
                self._results.setText(_("Stopped. Nothing was changed."))
                return
            applogger.warning("%s failed: %s", self.Name, error)
            self._results.setText(_("{operation} could not be computed: {error}").format(operation=_(self.Name), error=error))

        self._set_busy(True)
        # compute() takes no cancel argument (its signature is the plugins'
        # contract), so Stop is offered through stop_requested(): a long
        # compute polls it and raises app.analysis.Stopped.
        self._stop_event = None

        def work(cancel) -> Any:  # noqa: ANN001 - a threading.Event
            self._stop_event = cancel
            return self.compute(frame, casting, params, model)

        self._task = run_in_background(work, finished, failed)

    def _set_busy(self, busy: bool) -> None:
        if not busy:
            self._task = None
        self.stop_button.setVisible(busy)
        self._progress.setVisible(busy)
        for widget in (self.preview_button, self.apply_button, self._inputs_panel):
            widget.setEnabled(not busy)

    def _apply(self, result: Any) -> None:
        try:
            written = self.apply_results(result)
        except Exception as exc:  # noqa: BLE001 - reported, the dialog stays open
            applogger.exception("%s could not be written", self.Name)
            self._results.setText(_("{operation} could not be written: {error}").format(operation=_(self.Name), error=exc))
            return
        self._record(written, result)
        self.applied.emit()
        self._remember()
        super().accept()

    def _record(self, written: Sequence[Mapping[str, Any]], result: Any) -> None:
        """The Apply in the project's history, like a data operation's. Logged, never raised."""
        casting = self.roles_widget.casting()
        try:
            self._repo.record_operation(
                operation=self.Name,
                dialog=f"{type(self).__module__.rsplit('.', 1)[-1]}:{type(self).__name__}",
                parameters={**self.parameter_values(), "model": self.current_model()},
                entries=dialog_entries(self, inputs_only=True),
                sources=[{"table": self.table, "roles": casting.roles, "kinds": casting.kinds}],
                results=[dict(item) for item in written],
                app_version=str(APP_VERSION),
                report=self.format_results(result),
            )
        except Exception:  # noqa: BLE001 - a user action's boundary: reported, the window stays open
            applogger.exception("Could not record %s in the project's history", self.Name)

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
        """*frame*'s rows as a report table's cells: text escaped, whole numbers and DF as
        integers, the *p_columns* as p-values ("<.0001"), other numbers to *digits*."""
        rows = []
        for record in frame.itertuples(index=False):
            cells = []
            for column, value in zip(frame.columns, record):
                if column in p_columns:
                    cells.append(report_html.format_p_value(value))
                elif isinstance(value, str):
                    cells.append(html.escape(value))
                elif column == "DF" or (isinstance(value, int) and not isinstance(value, bool)):
                    cells.append("" if pd.isna(value) else str(int(value)))
                else:
                    cells.append(report_html.format_number(value, digits=digits))
            rows.append(cells)
        return rows

    def report_table(self, frame: pd.DataFrame, *, p_columns: Sequence[str] = (), digits: int = 6) -> str:
        """*frame* as a report table, its column names translated."""
        return report_html.table([_(str(c)) for c in frame.columns],
                                 self.report_rows(frame, p_columns=p_columns, digits=digits))

    def result_table_name(self, key: str) -> str:
        """A free name for a table the analysis writes: "_penguins_pls_vip".

        The leading underscore marks it generated, as the data operations'
        results are: the Tables list leaves it out until Show generated tables.
        """
        return self._repo.free_table_name(f"{GENERATED_PREFIX}{self.table}_{key}")

    def write_result_tables(self, frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
        """Each frame as a new table named after the source and its key ("_penguins_pls_vip"), one undo step.

        Returns {key: the name it got} - a free one, so an earlier run's tables
        stay. The figure's own tables are in the same step: Undo takes the
        tables and the figure the analysis draws away together.
        """
        repo = self._repo
        names = {key: self.result_table_name(key) for key in frames}
        repo.snapshot_for_undo([*names.values(), *repo.DESCRIPTOR_TABLES], label=f"{self.Name} on '{self.table}'")
        for key, frame in frames.items():
            repo.import_dataframe(frame, table_name=names[key], normalize_columns=False)
        return names

    def series_select(self, table: str, roles: Mapping[str, str], where: str = "") -> tuple[str, dict[str, str]]:
        """A series query naming each role's column as the role, the way every series reads its data."""
        columns = ", ".join(f"{self.quoted(column)} AS {self.quoted(role)}" for role, column in roles.items())
        sql = f"SELECT {columns} FROM {self.quoted(table)}" + (f" WHERE {where}" if where else "")
        return sql, {role: role for role in roles}

    def create_report_figure(self, name: str, notes_html: str, panels: Sequence[tuple]) -> int:
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
        panels = available

        repo = self._repo
        columns = 2 if len(panels) > 1 else 1
        rows = max(1, math.ceil(len(panels) / columns))
        figure_id = int(repo.create_figure_descriptor(
            name=name, nrows=rows, ncols=columns,
            options={FIGURE_VIEW_OPTIONS_KEY: {"notes_html": notes_html}},
        ))
        for index, (chart, title, x_label, y_label, series, *options) in enumerate(panels):
            axis_id = int(repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=index, chart_type=chart,
                title=title, x_label=x_label, y_label=y_label, options=dict(options[0]) if options else {},
            ))
            for position, (series_name, sql, roles, style) in enumerate(series):
                repo.create_series_descriptor(
                    axis_id=axis_id, series_index=position, name=series_name,
                    sql_query=sql, roles=roles, style=style,
                )
        return figure_id

    # ------------------------------------------------------------------
    # Leaving
    # ------------------------------------------------------------------

    def _remember(self) -> None:
        self._remember_casting()
        save_dialog_state(self, self._state_key)
        save_window_geometry(self, self._geometry_key)

    def reject(self) -> None:
        self.stop()
        self._remember()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop()
        self._remember()
        super().closeEvent(event)
