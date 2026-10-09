"""The base of every table operation: an analysis of a table's columns, not of a chart's series.

A series operation reads the x and y of the series it is given; a table
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
from app.logs.logger import applogger
from app.series_operations.dialog_base import OperationModel
from app.series_operations.parameter_form import ParameterForm
from app.series_operations.parameter_spec import Param
from app.styles.style import (
    CardFrame,
    TitledCard,
    apply_dialog_shell,
    create_action_button,
    create_section_title,
    icon_from_svg_source,
    stdSizeAndlayout,
)
from app.table_operations.column_roles import ColumnCasting, ColumnRole, ColumnRolesWidget
from app.utils.background import BackgroundTask, run_in_background
from app.utils.config import get_section, set_section
from app.utils.dialog_state import (
    dialog_entries,
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
)
from app.utils.i18n import _
from app.widgets.html_results import HtmlResultsView

#: Columns the application keeps on every table for itself: never offered.
APPLICATION_COLUMNS: frozenset[str] = frozenset({"Hide", "Selected"})


class TableOperationDialogBase(QDialog):
    """Inputs on the left, the report on the right; Preview, Copy, OK, Close."""

    Name: ClassVar[str] = ""
    Description: ClassVar[str] = ""
    Category: ClassVar[str] = ""
    Icon: ClassVar[str] = ""

    ROLES: ClassVar[tuple[ColumnRole, ...]] = ()
    MODELS: ClassVar[Mapping[str, OperationModel]] = {}
    PARAMS: ClassVar[tuple[Param, ...]] = ()
    #: The tab the declared parameters go on: one of their own by default, or
    #: one the operation built (a model's settings).
    PARAMETERS_TAB: ClassVar[str] = "Parameters"
    #: Raised when the window's layout changes, so a remembered size made for
    #: the old one is left behind (2: the inputs in tabs, 880 x 460).
    LAYOUT_VERSION: ClassVar[int] = 2

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

    def format_results(self, result: Any) -> str:
        """The report of *result*, as HTML."""
        raise NotImplementedError

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
            parameters.card.layout().addWidget(self._parameter_form.widget)
            self.roles_widget.add_tab(_(self.PARAMETERS_TAB)).addWidget(parameters)
        self.roles_widget.finish_tabs()
        self.roles_widget.tabs.setCurrentIndex(0)
        # The standard ground under the frames, not the window's grey.
        self._inputs_panel.setProperty("toolboxPage", True)
        self._inputs_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        right = CardFrame(self, "tableOperationResultsCard")
        right_layout = right.layout()
        assert right_layout is not None
        right_layout.addWidget(create_section_title(_("Results"), right))
        self._results = HtmlResultsView(self)
        self._results.setText(_("Cast the columns into their roles, then press Preview."))
        right_layout.addWidget(self._results)
        right.setMinimumWidth(260)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._inputs_panel)
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
        apply_dialog_shell(self, root, size=QSize(880, 460))
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
        self.roles_widget.set_columns(self._frame, kinds)
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
            self._set_busy(False)
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
            applogger.warning("%s failed: %s", self.Name, error)
            self._results.setText(_("{operation} could not be computed: {error}").format(operation=_(self.Name), error=error))

        self._set_busy(True)
        self._task = run_in_background(lambda _cancel: self.compute(frame, casting, params, model), finished, failed)

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
        """The Apply in the project's history, like a series operation's. Logged, never raised."""
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
        except Exception:
            applogger.exception("Could not record %s in the project's history", self.Name)

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
