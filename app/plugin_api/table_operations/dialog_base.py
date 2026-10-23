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
    QCheckBox,
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

from app.data import column_metadata
from app.data.sqlite_repo import ROW_ID, SqliteRepo
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
from app.plugin_api.table_operations import next_steps
from app.plugin_api.table_operations.column_roles import ColumnCasting, ColumnRole, ColumnRolesWidget
from app.utils.config import get_section, set_section
from app.utils.dialog_state import (
    dialog_entries,
)
from app.utils import report_html
from app.utils.i18n import _
from app.widgets.html_results import HtmlResultsView
from app.plugin_api.operation_shell import OperationShell

#: Columns the application keeps on every table for itself: never offered.
APPLICATION_COLUMNS: frozenset[str] = frozenset({"Hide", "Selected"})


#: What marks a table an analysis wrote (as the data operations' results): hidden from the Tables list by default.
GENERATED_PREFIX = "_"


class TableOperationDialogBase(OperationShell, QDialog):
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
    #: The Results pane's headline when the analysis fails.
    FAILURE_HEADLINE: ClassVar[str] = "Nothing was computed."
    #: Preview drew its figure (its id), or took it away again (-1): the window shows or drops it.
    preview_drawn = Signal(int)

    #: The chart types the analysis draws panels of: with two or more, the Charts tab has a box for each,
    #: and the report figure leaves out the panels whose box is cleared.
    CHART_PANELS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, repo: SqliteRepo, table: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self._frame = pd.DataFrame()
        self.row_ids = pd.Series(dtype="Int64")
        self._result: Any = None
        self._result_inputs: tuple[Any, ...] | None = None
        #: The figure the main window shows, for "Draw on: new axis in the current figure"; None: no such choice.
        self.current_figure_id: int | None = None
        #: A Preview's tables and figure are written inside a savepoint, rolled back by the next Preview,
        #: by Cancel and before OK writes them for good.
        self._preview_active = False
        self._previewing = False
        #: The question a "What next?" link asked for: the main window opens it once this one is closed.
        self.next_request: next_steps.NextStep | None = None
        self._state_key = type(self).__name__
        # The size is the layout's: a new layout starts from its own default
        # rather than from a size remembered for the old one.
        self._geometry_key = f"{self._state_key}@{self.LAYOUT_VERSION}"
        self.setWindowTitle(_(self.Name))
        if self.Icon:
            self.setWindowIcon(icon_from_svg_source(self.Icon, size=32))
        self._build_ui()
        self.restore_window()
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
        return report_html.document(_(self.Name), self.report_subtitle(result),
                                    self.report_body(result) + next_steps.section_html(self.next_steps(result)))

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
        self._build_charts_tab()
        self.roles_widget.finish_tabs()
        self.roles_widget.tabs.setCurrentIndex(0)
        # The standard ground under the frames, not the window's grey.
        self._inputs_panel.setProperty("toolboxPage", True)
        self._inputs_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # A standard frame: the title above the box, as the columns' and the roles'.
        right = TitledCard(self, _("Results"), "tableOperationResultsCard")
        right.layout().setStretchFactor(right.card, 1)
        self._results = HtmlResultsView(self)
        self._results.link_requested.connect(self._follow_link)
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
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        buttons.addWidget(self.progress_bar)
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
        """*table*'s visible rows, without the application's own columns.

        Each row's id (R-07 ``_id``) is kept aside in ``row_ids``, in the
        frame's order: what an analysis writes back into the table's own rows
        (Cluster Analysis's ClusterId) finds them by it.
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
        if not table:
            return
        if ColumnInfoDialog(self._repo, table, column, self).exec():
            self._on_table_changed()

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
        self._run(then=self._draw_preview)

    def ok(self) -> None:
        # The preview's drawing goes; OK writes it again for good, with its undo step.
        self._discard_preview()
        if self._result is not None and self._result_inputs == self._inputs():
            self._apply(self._result)
        else:
            self._run(then=self._apply)

    def _draw_preview(self, result: Any) -> None:
        """Write the result's tables and figure inside a savepoint, and have the window show them."""
        self._discard_preview()
        try:
            self._repo.begin_preview_savepoint(f"preview_{self._state_key}")
            self._previewing = True
            self.apply_results(result)
        except Exception as exc:  # noqa: BLE001 - shown like a failed run; the preview is undone
            self._previewing = False
            self._preview_active = True
            self._discard_preview()
            applogger.exception("%s: the preview could not be drawn", self.Name)
            self.show_failure(exc)
            return
        self._previewing = False
        self._preview_active = True
        figures = list(getattr(self, "created_figure_ids", []) or [])
        self.preview_drawn.emit(int(figures[0]) if figures else -1)

    def _discard_preview(self) -> None:
        """Roll the Preview's tables and figure back, if there is one; the window drops them."""
        if not self._preview_active:
            return
        self._preview_active = False
        try:
            self._repo.rollback_preview_savepoint()
        except Exception:  # noqa: BLE001 - cleanup: must not hide what it follows
            applogger.exception("%s: the preview could not be rolled back", self.Name)
        if hasattr(self, "created_figure_ids"):
            self.created_figure_ids = []
        self.preview_drawn.emit(-1)

    def busy_widgets(self) -> list[QWidget]:
        """Off while a calculation runs: the buttons, and the inputs it was started with."""
        return [self.preview_button, self.apply_button, self._inputs_panel]

    def _run(self, then: Any) -> None:
        problems = self.problems()
        if problems:
            self._results.setText("\n".join(problems))
            return
        frame, casting = self._frame.copy(), self.roles_widget.casting()
        params, model, inputs = self.parameter_values(), self.current_model(), self._inputs()

        def done(result: Any) -> None:
            self._result, self._result_inputs = result, inputs
            self.show_results(self.format_results(result))
            if then is not None:
                then(result)

        # compute() takes no cancel argument (its signature is the plugins' contract): a long compute
        # polls stop_requested() and raises app.analysis.Stopped.
        self.run_job(lambda _cancel: self.compute(frame, casting, params, model), done)

    def _apply(self, result: Any) -> None:
        try:
            written = self.apply_results(result)
        except Exception as exc:  # noqa: BLE001 - reported, the dialog stays open
            applogger.exception("%s could not be written", self.Name)
            self.show_failure(exc, headline=_("Nothing was written."))
            return
        self._record(written, result)
        self.applied.emit()
        self._remember()
        super().accept()

    def _record(self, written: Sequence[Mapping[str, Any]], result: Any) -> None:
        """The Apply in the project's history, like a data operation's (OperationShell.record_in_history)."""
        casting = self.roles_widget.casting()
        self.record_in_history(
            operation=self.Name,
            parameters={**self.parameter_values(), "model": self.current_model()},
            entries=dialog_entries(self, inputs_only=True),
            sources=[{"table": self.table, "roles": casting.roles, "kinds": casting.kinds}],
            results=written,
            report=self.format_results(result),
        )

    def results_widget(self) -> Any:
        """The Results pane, for OperationShell."""
        return self._results

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

    def write_result_tables(self, frames: Mapping[str, pd.DataFrame], *, also: Sequence[str] = ()) -> dict[str, str]:
        """Each frame as a new table named after the source and its key ("_penguins_pls_vip"), one undo step.

        Returns {key: the name it got} - a free one, so an earlier run's tables
        stay. The figure's own tables are in the same step: Undo takes the
        tables and the figure the analysis draws away together.
        """
        repo = self._repo
        names = {key: self.result_table_name(key) for key in frames}
        # *also*: tables the analysis changes besides, in the same undo step (the source's ClusterId).
        # Not for a Preview: it is rolled back, and OK writes it again with its step.
        if not self._previewing:
            repo.snapshot_for_undo([*names.values(), *also, *repo.DESCRIPTOR_TABLES],
                                   label=f"{self.Name} on '{self.table}'")
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
        # The panels whose chart type is cleared in the Charts tab are left out.
        checks = getattr(self, "_panel_checks", {})
        panels = [panel for panel in available if checks.get(str(panel[0])) is None
                  or checks[str(panel[0])].isChecked()]

        repo = self._repo
        target = getattr(self, "current_figure_id", None) if self.draw_on() == self.DEST_NEW_AXIS else None
        descriptor = repo.load_figure_descriptor(int(target)) if target is not None else None
        if descriptor is not None:
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
        return figure_id

    # ------------------------------------------------------------------
    # Leaving
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Charts tab
    # ------------------------------------------------------------------

    def _build_charts_tab(self) -> None:
        """The Charts tab: where the report figure is drawn (a new figure, or new axes in the current one)."""
        box = TitledCard(self._inputs_panel, _("Charts"))
        box.title_label.hide()  # the tab says it
        form = QFormLayout()
        stdSizeAndlayout(form)
        # A table analysis has no axis of its own: a new figure, or new axes in the figure the window shows.
        self.draw_on_combo = self.destination_combo_for(box, (self.DEST_NEW_FIGURE, self.DEST_NEW_AXIS),
                                                        self.DEST_NEW_FIGURE, self.inputs_changed)
        form.addRow(_("Draw on:"), self.draw_on_combo)
        box.card.layout().addLayout(form)
        self._panel_checks: dict[str, QCheckBox] = {}
        if len(self.CHART_PANELS) > 1:
            for chart in self.CHART_PANELS:
                check = QCheckBox(_(chart), box)
                check.setChecked(True)
                check.toggled.connect(lambda _on: self.inputs_changed())
                self._panel_checks[chart] = check
                box.card.layout().addWidget(check)
        self.roles_widget.add_tab(_("Charts")).addWidget(box)

    def draw_on(self) -> str:
        """The Charts tab's choice; a new figure when there is no current one to add to."""
        combo = getattr(self, "draw_on_combo", None)
        if getattr(self, "current_figure_id", None) is None or combo is None:
            return self.DEST_NEW_FIGURE
        return str(combo.currentData() or self.DEST_NEW_FIGURE)

    def geometry_key(self) -> str:
        """The size is the layout's: a new layout starts from its own default, not from the old one's size."""
        return self._geometry_key

    def parameter_forms(self) -> list[Any]:
        return [self._parameter_form] if self._parameter_form is not None else []

    def _remember(self) -> None:
        self._remember_casting()
        self.remember_window()

    def reject(self) -> None:
        self.stop()
        self._discard_preview()
        self._remember()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.stop()
        self._discard_preview()
        self._remember()
        super().closeEvent(event)
