"""Standalone Design of Experiments table creator.

Left pane: model and parameters. Right pane: live HTML preview.
OK creates a new SQLite source table; Cancel makes no changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import itertools
import re
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QFrame, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.i18n import _
from app.utils.screen_fit import fit_on_show

_TABLE_RE = re.compile(r"[^A-Za-z0-9_]+")
_MAX_PREVIEW_ROWS = 500


@dataclass(frozen=True, slots=True)
class Factor:
    name: str
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class DOERequest:
    model: str
    table_name: str
    factors: tuple[Factor, ...]
    responses: tuple[str, ...]
    levels: int
    center_points: int
    samples: int
    replicates: int
    randomize: bool
    seed: int
    lhs_algorithm: str


class DOEExperimentDialog(QDialog):
    """Create a DOE matrix as a new table, independently from charts."""

    Name = "Design of Experiments"
    Description = "Create a DOE test matrix as a new table."
    Category = "New Table"
    table_created = Signal(str)
    applied = Signal()  # Existing MainWindow refresh hook.

    def __init__(self, repo: SqliteRepo, figure_id: int | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        del figure_id  # Accepted for compatibility; DOE has no chart target.
        self._repo = repo
        self.created_table_name: str | None = None
        self._preview_frame: pd.DataFrame | None = None
        self._refresh_pending = False
        self._building = True
        self.setWindowTitle(_(self.Name))
        self.resize(1000, 700)
        self.setMinimumSize(850, 550)
        # 700 tall does not fit a laptop's 720 with its title bar: shrunk to
        # the screen on show, like every dialog built by apply_dialog_shell.
        fit_on_show(self)
        self._create_widgets()
        self._build_ui()
        self._connect_signals()
        self._building = False
        self._model_changed()

    @staticmethod
    def _spin(low: int, high: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setValue(value)
        return widget

    @staticmethod
    def _title(text: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    @staticmethod
    def _configure_form(form: QFormLayout) -> None:
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(7)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    def _create_widgets(self) -> None:
        self.model_combo = QComboBox(self)
        # Each label written inside _() so the catalogues find it.
        for label, code in (
            (_("Full factorial"), "full_factorial"),
            (_("Two-level factorial"), "two_level"),
            (_("Box-Behnken"), "box_behnken"),
            (_("Central composite, face-centered"), "ccd_face"),
            (_("Central composite, circumscribed"), "ccd_circumscribed"),
            (_("Latin hypercube"), "latin_hypercube"),
        ):
            self.model_combo.addItem(label, code)

        self.table_name = QLineEdit("DOE_Experiment", self)
        self.factor_count = self._spin(1, 20, 3)
        self.response_count = self._spin(1, 20, 1)
        self.levels = self._spin(2, 10, 2)
        self.center_points = self._spin(0, 100, 3)
        self.samples = self._spin(2, 100_000, 20)
        self.replicates = self._spin(1, 1_000, 1)
        self.seed = self._spin(0, 2_147_483_647, 12_345)
        for spin in (
            self.factor_count,
            self.response_count,
            self.levels,
            self.center_points,
            self.samples,
            self.replicates,
            self.seed,
        ):
            spin.setMinimumWidth(72)
            spin.setMaximumWidth(112)
        self.randomize = QCheckBox(_("Randomize run order"), self)
        self.randomize.setChecked(True)
        self.lhs_algorithm = QComboBox(self)
        self.lhs_algorithm.addItem(_("Random within strata"), "random")
        self.lhs_algorithm.addItem(_("Centered within strata"), "centered")
        self.lhs_algorithm.addItem(_("Maximin candidate search"), "maximin")

        self.factor_table = QTableWidget(3, 3, self)
        self.factor_table.setHorizontalHeaderLabels([_("Factor"), _("Low"), _("High")])
        self.factor_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.factor_table.setAlternatingRowColors(True)
        self.factor_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.factor_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.factor_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.factor_table.verticalHeader().hide()
        header = self.factor_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.factor_table.setColumnWidth(0, 118)
        self.factor_table.setMinimumWidth(310)
        for row in range(3):
            self._set_factor_row(row, f"Factor_{row + 1}", -1, 1)

        self.response_table = QTableWidget(1, 1, self)
        self.response_table.setHorizontalHeaderLabels([_("Response")])
        self.response_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.response_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.response_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.response_table.verticalHeader().hide()
        self.response_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._set_response_row(0, "Response_1")

        self.preview = QTextBrowser(self)
        self.preview.setObjectName("doePreview")
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self)
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText(_("OK"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(_("Cancel"))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        scroll = QScrollArea(splitter)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(410)
        scroll.setMaximumWidth(560)
        left = QWidget(scroll)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._title(_("DOE model"), left))
        model_host = QWidget(left)
        model_form = QFormLayout(model_host)
        self._configure_form(model_form)
        model_form.addRow(_("Model:"), self.model_combo)
        left_layout.addWidget(model_host)
        left_layout.addWidget(self._title(_("Parameters"), left))
        parameter_host = QWidget(left)
        parameter_columns = QHBoxLayout(parameter_host)
        parameter_columns.setContentsMargins(0, 0, 0, 0)
        parameter_columns.setSpacing(8)
        first_column = QWidget(parameter_host)
        second_column = QWidget(parameter_host)
        first_form = QFormLayout(first_column)
        second_form = QFormLayout(second_column)
        self._configure_form(first_form)
        self._configure_form(second_form)
        self.parameter_forms = (first_form, second_form)
        for label, widget in (
            (_("New table:"), self.table_name),
            (_("Factors:"), self.factor_count),
            (_("Responses:"), self.response_count),
            (_("Levels:"), self.levels),
            (_("Replicates:"), self.replicates),
        ):
            first_form.addRow(label, widget)
        for label, widget in (
            (_("Center points:"), self.center_points),
            (_("LHS samples:"), self.samples),
            (_("LHS algorithm:"), self.lhs_algorithm),
            (_("Random seed:"), self.seed),
            ("", self.randomize),
        ):
            second_form.addRow(label, widget)
        parameter_columns.addWidget(first_column, 1)
        parameter_columns.addWidget(second_column, 1)
        left_layout.addWidget(parameter_host)
        left_layout.addWidget(self._title(_("Factor names and bounds"), left))
        left_layout.addWidget(self.factor_table, 0)
        left_layout.addWidget(self._title(_("Response columns"), left))
        left_layout.addWidget(self.response_table)
        left_layout.addStretch(1)
        scroll.setWidget(left)

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.addWidget(self._title(_("Test matrix preview"), right))
        right_layout.addWidget(self.preview, 1)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([455, 545])
        root.addWidget(splitter, 1)
        root.addWidget(self.buttons)
        self._update_table_heights()

    def _connect_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.table_name.textChanged.connect(self._queue_preview)
        self.factor_count.valueChanged.connect(self._resize_factors)
        self.response_count.valueChanged.connect(self._resize_responses)
        self.factor_table.itemChanged.connect(self._queue_preview)
        self.response_table.itemChanged.connect(self._queue_preview)
        for widget in (self.levels, self.center_points, self.samples,
                       self.replicates, self.seed):
            widget.valueChanged.connect(self._queue_preview)
        self.randomize.toggled.connect(self._queue_preview)
        self.lhs_algorithm.currentIndexChanged.connect(self._queue_preview)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def current_model(self) -> str:
        return str(self.model_combo.currentData() or "full_factorial")

    def _model_changed(self, *_args: Any) -> None:
        model = self.current_model()
        for widget, visible in (
            (self.levels, model == "full_factorial"),
            (self.center_points, model in {"box_behnken", "ccd_face", "ccd_circumscribed"}),
            (self.samples, model == "latin_hypercube"),
            (self.lhs_algorithm, model == "latin_hypercube"),
        ):
            self._set_parameter_row_visible(widget, visible)
        self._queue_preview()

    def _set_parameter_row_visible(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        for form in self.parameter_forms:
            label = form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
                break

    def _queue_preview(self, *_args: Any) -> None:
        if self._building or self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(0, self._run_queued_preview)

    def _run_queued_preview(self) -> None:
        self._refresh_pending = False
        self.refresh_preview()

    def _resize_factors(self, count: int) -> None:
        old = self.factor_table.rowCount()
        self.factor_table.blockSignals(True)
        try:
            self.factor_table.setRowCount(count)
            for row in range(old, count):
                self._set_factor_row(row, f"Factor_{row + 1}", -1, 1)
        finally:
            self.factor_table.blockSignals(False)
        self._update_table_heights()
        self._queue_preview()

    def _resize_responses(self, count: int) -> None:
        old = self.response_table.rowCount()
        self.response_table.blockSignals(True)
        try:
            self.response_table.setRowCount(count)
            for row in range(old, count):
                self._set_response_row(row, f"Response_{row + 1}")
        finally:
            self.response_table.blockSignals(False)
        self._update_table_heights()
        self._queue_preview()

    def _update_table_heights(self) -> None:
        """Keep definition tables compact; scroll when row counts exceed caps."""
        self._fit_table_height(self.factor_table, maximum_visible_rows=5)
        self._fit_table_height(self.response_table, maximum_visible_rows=3)

    @staticmethod
    def _fit_table_height(table: QTableWidget, maximum_visible_rows: int) -> None:
        visible_rows = max(1, min(table.rowCount(), maximum_visible_rows))
        row_height = table.verticalHeader().defaultSectionSize()
        header_height = table.horizontalHeader().sizeHint().height()
        frame = 2 * table.frameWidth()
        height = header_height + visible_rows * row_height + frame + 4
        table.setMinimumHeight(height)
        table.setMaximumHeight(height)

    def _set_factor_row(self, row: int, name: str, low: float, high: float) -> None:
        for column, value in enumerate((name, f"{low:g}", f"{high:g}")):
            self.factor_table.setItem(row, column, QTableWidgetItem(value))

    def _set_response_row(self, row: int, name: str) -> None:
        self.response_table.setItem(row, 0, QTableWidgetItem(name))

    @staticmethod
    def _cell(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _read_request(self) -> DOERequest:
        raw_name = self.table_name.text().strip()
        table_name = _TABLE_RE.sub("_", raw_name).strip("_")
        if not table_name:
            raise ValueError(_("Enter a table name."))
        if table_name[0].isdigit():
            table_name = f"DOE_{table_name}"

        used = {"id"}
        factors: list[Factor] = []
        for row in range(self.factor_table.rowCount()):
            name = self._cell(self.factor_table, row, 0)
            if not name:
                raise ValueError(_("Factor {number} has no name.").format(number=row + 1))
            key = name.casefold()
            if key in used:
                raise ValueError(_("Duplicate column name: {name}").format(name=name))
            try:
                low = float(self._cell(self.factor_table, row, 1))
                high = float(self._cell(self.factor_table, row, 2))
            except ValueError as exc:
                raise ValueError(_("Factor '{name}' has invalid bounds.").format(name=name)) from exc
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(_("Factor '{name}' requires finite Low < High.").format(name=name))
            used.add(key)
            factors.append(Factor(name, low, high))

        responses: list[str] = []
        for row in range(self.response_table.rowCount()):
            name = self._cell(self.response_table, row, 0)
            if not name:
                raise ValueError(_("Response {number} has no name.").format(number=row + 1))
            key = name.casefold()
            if key in used:
                raise ValueError(_("Duplicate column name: {name}").format(name=name))
            used.add(key)
            responses.append(name)

        model = self.current_model()
        if model == "box_behnken" and len(factors) < 3:
            raise ValueError(_("Box-Behnken requires at least three factors."))
        return DOERequest(
            model=model, table_name=table_name, factors=tuple(factors),
            responses=tuple(responses), levels=self.levels.value(),
            center_points=self.center_points.value(), samples=self.samples.value(),
            replicates=self.replicates.value(), randomize=self.randomize.isChecked(),
            seed=self.seed.value(),
            lhs_algorithm=str(self.lhs_algorithm.currentData() or "random"),
        )

    @classmethod
    def generate_matrix(cls, request: DOERequest) -> pd.DataFrame:
        n = len(request.factors)
        rng = np.random.default_rng(request.seed)
        if request.model == "full_factorial":
            coded = np.asarray(list(itertools.product(
                *[np.linspace(-1.0, 1.0, request.levels) for _ in range(n)])),
                dtype=float)
        elif request.model == "two_level":
            coded = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=float)
        elif request.model == "box_behnken":
            coded = cls._box_behnken(n, request.center_points)
        elif request.model in {"ccd_face", "ccd_circumscribed"}:
            coded = cls._central_composite(
                n, request.center_points, request.model == "ccd_face")
        elif request.model == "latin_hypercube":
            coded = 2.0 * cls._latin_hypercube(
                request.samples, n, request.lhs_algorithm, rng) - 1.0
        else:
            raise ValueError(_("Unsupported DOE model: {model}").format(model=request.model))

        if request.replicates > 1:
            coded = np.repeat(coded, request.replicates, axis=0)
        if request.randomize:
            coded = coded[rng.permutation(len(coded))]

        data: dict[str, Any] = {"Id": np.arange(1, len(coded) + 1, dtype=int)}
        for column, factor in enumerate(request.factors):
            midpoint = (factor.low + factor.high) / 2.0
            half_range = (factor.high - factor.low) / 2.0
            data[factor.name] = midpoint + coded[:, column] * half_range
        for response in request.responses:
            data[response] = np.full(len(coded), np.nan)
        return pd.DataFrame(data)

    @staticmethod
    def _box_behnken(n: int, centers: int) -> np.ndarray:
        if n < 3:
            raise ValueError(_("Box-Behnken requires at least three factors."))
        rows: list[np.ndarray] = []
        for first in range(n - 1):
            for second in range(first + 1, n):
                for signs in itertools.product((-1.0, 1.0), repeat=2):
                    row = np.zeros(n, dtype=float)
                    row[first], row[second] = signs
                    rows.append(row)
        rows.extend(np.zeros(n, dtype=float) for _ in range(centers))
        return np.vstack(rows)

    @staticmethod
    def _central_composite(n: int, centers: int, face_centered: bool) -> np.ndarray:
        factorial = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=float)
        alpha = 1.0 if face_centered else float((2 ** n) ** 0.25)
        axial = np.zeros((2 * n, n), dtype=float)
        for column in range(n):
            axial[2 * column, column] = -alpha
            axial[2 * column + 1, column] = alpha
        return np.vstack((factorial, axial, np.zeros((centers, n), dtype=float)))

    @staticmethod
    def _latin_hypercube(samples: int, dimensions: int, algorithm: str,
                         rng: np.random.Generator) -> np.ndarray:
        def candidate(centered: bool) -> np.ndarray:
            matrix = np.empty((samples, dimensions), dtype=float)
            for dimension in range(dimensions):
                offset: float | np.ndarray = 0.5 if centered else rng.random(samples)
                points = (np.arange(samples) + offset) / samples
                matrix[:, dimension] = points[rng.permutation(samples)]
            return matrix

        if algorithm == "centered":
            return candidate(True)
        if algorithm != "maximin":
            return candidate(False)
        best: np.ndarray | None = None
        best_distance = -np.inf
        for _attempt in range(20):
            trial = candidate(False)
            delta = trial[:, None, :] - trial[None, :, :]
            distances = np.sqrt(np.sum(delta * delta, axis=2))
            np.fill_diagonal(distances, np.inf)
            minimum = float(np.min(distances))
            if minimum > best_distance:
                best, best_distance = trial, minimum
        if best is None:
            raise RuntimeError(_("Could not generate the Latin hypercube."))
        return best

    def refresh_preview(self) -> None:
        try:
            request = self._read_request()
            frame = self.generate_matrix(request)
        except Exception as exc:  # Validation is displayed in the preview.
            self._preview_frame = None
            self.ok_button.setEnabled(False)
            self.preview.setHtml(
                f"<h3>{html.escape(_('Preview unavailable'))}</h3>"
                f"<p style='color:#a40000'>{html.escape(str(exc))}</p>")
            return
        self._preview_frame = frame
        self.ok_button.setEnabled(True)
        self.preview.setHtml(self._preview_html(frame, request))

    def _preview_html(self, frame: pd.DataFrame, request: DOERequest) -> str:
        factor_names = {factor.name for factor in request.factors}
        response_names = set(request.responses)
        schema: list[dict[str, str]] = []
        for column in frame.columns:
            role = ("Id" if column == "Id" else "Factor" if column in factor_names
                    else "Response" if column in response_names else "Data")
            dtype = frame[column].dtype
            sql_type = ("INTEGER" if pd.api.types.is_integer_dtype(dtype)
                        else "REAL" if pd.api.types.is_numeric_dtype(dtype) else "TEXT")
            schema.append({
                _("Column"): str(column), _("SQLite type"): sql_type,
                _("Role"): role,
                _("Nullable"): _("Yes") if role == "Response" else _("No"),
            })
        shown = frame.head(_MAX_PREVIEW_ROWS)
        note = ""
        if len(shown) < len(frame):
            note = "<p><i>" + html.escape(
                _("Showing the first {count} rows.").format(count=len(shown))) + "</i></p>"
        return (
            "<style>body{font-family:sans-serif;color:#222}"
            "table{border-collapse:collapse;width:100%;margin-bottom:14px}"
            "th,td{border:1px solid #d0d0d0;padding:4px 7px;text-align:right}"
            "th{background:#f2f2f2}td:first-child,th:first-child{text-align:left}</style>"
            f"<h3>{html.escape(_('DOE test matrix'))}</h3>"
            f"<p><b>{html.escape(_('Model'))}:</b> {html.escape(self.model_combo.currentText())}<br>"
            f"<b>{html.escape(_('New table'))}:</b> {html.escape(request.table_name)}<br>"
            f"<b>{html.escape(_('Runs'))}:</b> {len(frame)} &nbsp; "
            f"<b>{html.escape(_('Factors'))}:</b> {len(request.factors)} &nbsp; "
            f"<b>{html.escape(_('Responses'))}:</b> {len(request.responses)}</p>"
            f"<h4>{html.escape(_('Generated table schema'))}</h4>"
            + pd.DataFrame(schema).to_html(index=False, border=0, escape=True)
            + f"<h4>{html.escape(_('Generated data'))}</h4>" + note
            + shown.to_html(index=False, border=0, na_rep="", escape=True))

    def _unique_table_name(self, requested: str) -> str:
        name, suffix = requested, 1
        while self._repo.check_if_table_exists(name):
            name = f"{requested}_{suffix}"
            suffix += 1
        return name

    def accept(self) -> None:
        try:
            request = self._read_request()
            frame = self.generate_matrix(request)  # Never save a stale preview.
            table_name = self._unique_table_name(request.table_name)
            self._repo.import_dataframe(
                frame, table_name=table_name, normalize_columns=False)
        except Exception as exc:
            applogger.error("DOE table creation failed: %s", exc)
            QMessageBox.warning(self, _(self.Name), str(exc))
            return
        self._preview_frame = frame
        self.created_table_name = table_name
        self.table_created.emit(table_name)
        self.applied.emit()
        super().accept()
