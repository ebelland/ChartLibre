"""Function plotting dialog with Library, Expression and Code models.

Extends the original SeriesFunctionDialog while retaining the existing Library
scanner logic. User-defined models can be saved in the current project's
SQLite database table ``__sympy__`` (app.data_operations.sympy_store) and
imported/exported as text files.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import sympy as sp
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QMessageBox, QPushButton, QPlainTextEdit, QSizePolicy, QTableWidget,
    QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from app.analysis.function_plot import (
    SPACING_LINEAR, SPACING_LOG, build_range, evaluate_curve, evaluate_surface,
)
from app.scanners.functions_scanner import FunctionScanner, SurfaceFunctionScanner
from app.plugin_api.data_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.plugin_api.data_operations.results import TableResult
from app.plugin_api.data_operations.dialog_base import (
    ResultSeriesSpec, DataOperationDialogBase, generated_table_name,
)
from app.styles.style import create_doc_link, mark_editor_panel, set_doc_link
from app.utils.i18n import _
from app.utils import report_html
from app.data.data_source import DataSource
from app.data.repo.function_sql import GridRange
from app.data_operations.sympy_store import (
    UntrustedFunction,
    _scientific_import,
    ensure_table,
    fingerprint as stored_fingerprint,
    get as get_stored,
    rows as stored_rows,
    save as save_stored,
    source_sql as stored_source_sql,
    stored_values,
    untrusted as stored_untrusted,
)


@dataclass(slots=True)
class FunctionResult(TableResult):
    function_name: str
    result_name: str
    x: np.ndarray
    y: np.ndarray
    params: dict[str, float]
    expression: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    z: np.ndarray | None = None
    #: A stored function (Expression or Code): the function as saved, and the query of its live series - the
    #: formula evaluated by SQLite at each draw, nothing written but the function (Q2 of the function data
    #: source plan). Empty for a library function: its values are a table, as before.
    stored: dict[str, Any] | None = None
    source_sql: str = ""

    @property
    def model_name(self) -> str:
        return self.function_name

    def preview(self, dialog: Any, axis_id: int) -> None:
        if not self.source_sql:
            TableResult.preview(self, dialog, axis_id)
            return
        dialog.store_function(self)
        dialog.create_preview_series(axis_id, "", self)

    def apply(self, dialog: Any, axis_id: int) -> None:
        if not self.source_sql:
            TableResult.apply(self, dialog, axis_id)
            return
        dialog.store_function(self)
        dialog.create_result_series(axis_id, "", self)

    def to_df(self) -> pd.DataFrame:
        if self.z is not None:
            return pd.DataFrame({"x": self.x, "y": self.y, "z": self.z})
        return pd.DataFrame({"x": self.x, "y": self.y})


class _CallableModel:
    """Small adapter matching the callable shape expected by evaluate_curve."""
    def __init__(self, func: Callable[..., Any]):
        self._func = func

    def __call__(self, x: np.ndarray, *params: float) -> Any:
        return self._func(x, *params)

    # Some evaluators use eval() rather than __call__().
    def eval(self, x: np.ndarray, *params: float) -> Any:
        return self._func(x, *params)


class SeriesFunctionDialog(DataOperationDialogBase):
    DESTINATIONS = (
        DataOperationDialogBase.DEST_SAME_AXIS,
        DataOperationDialogBase.DEST_NEW_AXIS,
        DataOperationDialogBase.DEST_NEW_FIGURE,
    )
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_SAME_AXIS
    Name = "Function"
    Description = "Plot a function"
    INPUT_MINIMUM_POINTS = 0
    SHOWS_SERIES_SELECTOR = False
    SHOWS_AXIS_SERIES_PAGE = False

    # Kept as declared parameters so the existing ParameterForm logic, value
    # validation and build_range() path remain untouched. The widget is moved
    # from Parameters to Charts by build_parameter_selector/build_chart_selector.
    PARAMS = (
        FloatParam("start", "X from:", tooltip="Start of the range to evaluate over.",
                   default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
                   tab="charts"),
        FloatParam("stop", "X to:", tooltip="End of the range to evaluate over.",
                   default_value=10.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
                   tab="charts"),
        IntParam("points", "X points:", tooltip="How densely to sample.",
                 default_value=200, minimum=2, maximum=1_000_000, step=50, tab="charts"),
        FloatParam("y_start", "Y from:", tooltip="Start of the y range for z = f(x, y).",
                   default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
                   tab="charts"),
        FloatParam("y_stop", "Y to:", tooltip="End of the y range for z = f(x, y).",
                   default_value=10.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
                   tab="charts"),
        IntParam("y_points", "Y points:", tooltip="Number of y samples for z = f(x, y).",
                 default_value=200, minimum=2, maximum=1_000_000, step=50, tab="charts"),
        ChoiceParam("spacing", "Spacing:",
                    tooltip="Logarithmic spacing requires a range strictly above zero.",
                    choices=(("Linear", SPACING_LINEAR), ("Logarithmic", SPACING_LOG)),
                    tab="charts"),
        ChoiceParam(
            "chart_type", "Chart type:",
            tooltip="New axes and figures use this chart type. Same axis keeps the selected axis.",
            default_value="Scatter Plot",
            choices=(
                ("Scatter Plot", "Scatter Plot"),
                ("Surface Plot", "Surface Plot"),
                ("Wireframe", "Wireframe"),
                ("3D Scatter", "3D Scatter"),
                ("Contour", "Contour"),
                ("Filled Contour", "Filled Contour"),
            ),
            tab="charts",
        ),
    )

    Icon = '<path d="M4 20c4 0 4-16 8-16"/><path d="M12 4c4 0 4 16 8 16"/>'
    Keywords = "funzione formula expression espressione plot function grafico di funzione"
    Goals: list[str] = []
    Category = "Graph"
    Order = 5
    WINDOW_SIZE = (900, 760)

    def __init__(self, **kwargs: Any) -> None:
        self._scanner = FunctionScanner()
        self._surface_scanner = SurfaceFunctionScanner()
        self._selected_function: dict[str, Any] = {}
        self._range_widget: QWidget | None = None
        super().__init__(**kwargs)
        self.axis_series_panel.setVisible(False)
        self._reload_function_tree()
        self.mark_results_stale()

    # ---------- UI ----------
    def init_operation_widgets(self) -> None:
        self._doc_link = cast(QLabel, create_doc_link(self))
        self._function_tree = QTreeWidget(self)
        self._params_table = QTableWidget(0, 2, self)
        self._expression_label = QLabel("", self)

        self._model_tabs = QTabWidget(self)
        self._current_function_name = ""
        self._expr_symbols = QLineEdit(self)
        self._expr_description = QPlainTextEdit(self)
        self._expr_description.setFixedHeight(72)
        self._expr_description.setPlaceholderText(_("Optional description"))
        self._expr_text = QPlainTextEdit(self)
        self._expr_text.setPlaceholderText("x**2 + a*x + b")
        self._expr_text.setFixedHeight(90)

        self._code_description = QPlainTextEdit(self)
        self._code_description.setFixedHeight(72)
        self._code_description.setPlaceholderText(_("Optional description"))
        self._code_text = QPlainTextEdit(self)
        self._code_text.setPlaceholderText(
            "def MyFunction(a, b):\n"
            "    f_numeric = a*x**2 + b\n"
            "    return f_numeric"
        )
        self._code_text.setMinimumHeight(230)

        self._stored_combo = QComboBox(self)
        self._stored_combo.setMinimumWidth(220)
        self._stored_combo.currentIndexChanged.connect(self._stored_selected)

        for widget in (self._expr_symbols, self._expr_description, self._expr_text,
                       self._code_description, self._code_text):
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._custom_model_changed)
            else:
                widget.textChanged.connect(self._custom_model_changed)

    def build_model_selector(self) -> QWidget:
        self.model_combo.setVisible(False)

        # Library: original logic preserved.
        library = QWidget(self)
        ll = QVBoxLayout(library); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(6)
        self._function_tree.setHeaderLabels([_("Function"), _("Description")])
        self._function_tree.setMinimumHeight(200)
        self._function_tree.currentItemChanged.connect(self._on_function_changed)
        mark_editor_panel(self._function_tree)
        ll.addWidget(self._function_tree, 1)
        self._expression_label.setWordWrap(True)
        self._expression_label.setTextFormat(Qt.TextFormat.RichText)
        self._expression_label.setProperty("muted", True)
        ll.addWidget(self._expression_label)
        docs = QWidget(library); form = QFormLayout(docs); form.setContentsMargins(0, 0, 0, 0)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.addRow(_("Docs:"), self._doc_link); ll.addWidget(docs)

        expression = QWidget(self)
        expression.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ef = QFormLayout(expression)
        ef.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        ef.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        ef.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        ef.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self._expr_symbols.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._expr_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._expr_description.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ef.addRow(_("Description:"), self._expr_description)
        ef.addRow(_("Parameters:"), self._expr_symbols)
        ef.addRow(_("Expression:"), self._expr_text)
        expr_note = QLabel(_("x, y and z are implicit coordinates and must not be listed as parameters."), expression)
        expr_note.setWordWrap(True)
        expr_note.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        expr_note.setProperty("muted", True)
        ef.addRow(expr_note)

        code = QWidget(self)
        code.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cf = QFormLayout(code)
        cf.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        cf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        cf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        cf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self._code_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._code_description.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cf.addRow(_("Description:"), self._code_description)
        cf.addRow(_("Code:"), self._code_text)
        code_note = QLabel(_("x, y and z are implicit coordinates and must not be listed as parameters."), code)
        code_note.setWordWrap(True)
        code_note.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        code_note.setProperty("muted", True)
        cf.addRow(code_note)

        self._model_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._model_tabs.addTab(library, _("Library"))
        self._model_tabs.addTab(expression, _("Expression"))
        self._model_tabs.addTab(code, _("Code"))
        self._model_tabs.currentChanged.connect(self._source_changed)

        root = QWidget(self); layout = QVBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._model_tabs, 1)

        buttons = QHBoxLayout()
        buttons.addWidget(QLabel(_("Stored:"), root))
        self._stored_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        buttons.addWidget(self._stored_combo, 1)
        for text, slot in ((_("New"), self._new_function), (_("Save"), self._save_current),
                           (_("Import..."), self._import_text), (_("Export..."), self._export_text)):
            btn = QPushButton(text, root); btn.clicked.connect(slot); buttons.addWidget(btn)
        buttons.addStretch(1); layout.addLayout(buttons)
        self._refresh_stored_combo()
        return root

    def build_parameter_selector(self) -> QWidget:
        """Only function parameters live on Parameters now."""
        widget = QWidget(self); layout = QVBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
        self._params_table.setHorizontalHeaderLabels([_("Parameter"), _("Value")])
        self._params_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._params_table.verticalHeader().setVisible(False)
        self._params_table.setMinimumHeight(140)
        self._params_table.itemChanged.connect(lambda *_a: self.mark_results_stale())
        mark_editor_panel(self._params_table)
        layout.addWidget(self._params_table, 1)
        return widget

    def connect_operation_signals(self) -> None:
        return None

    # ---------- sources ----------
    def _source_changed(self, _index: int) -> None:
        if self._model_tabs.currentIndex() == 0:
            self._on_function_changed(self._function_tree.currentItem())
        else:
            self._custom_model_changed()

    def _custom_model_changed(self, *_a: Any) -> None:
        idx = self._model_tabs.currentIndex()
        if idx == 1:
            names = self._parse_symbols(self._expr_symbols.text())
            self._rebuild_param_names(names)
        elif idx == 2:
            try:
                _name, params = self._validated_code_signature()
                self._rebuild_param_names(params)
            except Exception:
                pass
        self.mark_results_stale()

    @staticmethod
    def _parse_symbols(text: str) -> list[str]:
        names = [n.strip() for n in text.split(",") if n.strip()]
        if len(set(names)) != len(names) or any(not n.isidentifier() for n in names):
            raise ValueError("Parameters must be unique valid Python identifiers separated by commas.")
        if {"x", "y", "z"}.intersection(names):
            raise ValueError("x, y and z are implicit coordinates and cannot be model parameters.")
        return names

    def _validated_code_signature(self) -> tuple[str, list[str]]:
        code = self._code_text.toPlainText()
        wanted = self._current_function_name.strip()
        if not wanted or not wanted.isidentifier():
            raise ValueError("Function name must be a valid Python identifier.")
        tree = ast.parse(code, mode="exec")
        funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        node = next((n for n in funcs if n.name == wanted), None)
        if node is None or isinstance(node, ast.AsyncFunctionDef):
            raise ValueError(f"Code must define def {wanted}(...).")
        args = [a.arg for a in node.args.args]
        if {"x", "y", "z"}.intersection(args):
            raise ValueError("x, y and z are implicit coordinates and cannot be function parameters.")
        if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            raise ValueError("*args, **kwargs and keyword-only arguments are not supported.")
        returns_f_numeric = any(
            isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id == "f_numeric"
            for n in ast.walk(node)
        )
        if not returns_f_numeric:
            raise ValueError("The function must return f_numeric.")
        return wanted, args

    def _check_project_code(self) -> None:
        """The editor holds one of the project's stored functions, not trusted here: ask before running it.

        Code typed or imported here is the user's own; a stored function from a project made elsewhere runs
        only once trusted on this computer (todo ID47).
        """
        if self._model_tabs.currentIndex() == 1:
            current = {"source_type": "expression", "code_string": self._expr_text.toPlainText()}
        else:
            current = {"source_type": "code", "code_string": self._code_text.toPlainText()}
        wanted = stored_fingerprint(current)
        stored = next((row for row in stored_untrusted(self._repo) if stored_fingerprint(row) == wanted), None)
        if stored is None:
            return
        from app.dialogs.function_trust_dialog import allowed_to_run

        if not allowed_to_run(self, self._repo, stored):
            raise UntrustedFunction(_("Not run: the function '{name}' is not trusted on this computer.").format(
                name=stored.get("function_name", "")))

    def _check_without_running(self) -> None:
        """The model's text checked as text: an imported file's code is read here, not run."""
        if self._model_tabs.currentIndex() == 1:
            text = self._expr_text.toPlainText().strip()
            if not text:
                raise ValueError("Expression is required.")
            ast.parse(text.replace("^", "**"), mode="eval")
            return
        self._validated_code_signature()

    def _custom_callable(self) -> tuple[str, Callable[..., Any], str]:
        self._check_project_code()
        if self._model_tabs.currentIndex() == 1:
            name = self._current_function_name.strip()
            if not name:
                raise ValueError("Function name is required.")
            parameters = self._parse_symbols(self._expr_symbols.text())
            x_symbol = sp.Symbol("x")
            parameter_symbols = [sp.Symbol(param) for param in parameters]
            local_symbols = {"x": x_symbol, "y": sp.Symbol("y"), "z": sp.Symbol("z"), **dict(zip(parameters, parameter_symbols))}
            expr_text = self._expr_text.toPlainText().strip()
            expr = sp.sympify(expr_text, locals=local_symbols)
            return name, sp.lambdify((x_symbol, *parameter_symbols), expr, modules="numpy"), expr_text

        name, _params = self._validated_code_signature()
        # Restrictive globals: useful numerical/SymPy tools, no imports/builtins.
        env: dict[str, Any] = {"np": np, "numpy": np, "sp": sp, "sympy": sp, "__builtins__": {"__import__": _scientific_import}}
        exec(compile(self._code_text.toPlainText(), "<sympy>", "exec"), env, env)
        func = env.get(name)
        if not callable(func):
            raise ValueError(f"{name} is not callable.")
        def numeric(x_values: np.ndarray, *params: float) -> Any:
            env["x"] = x_values
            return func(*params)
        return name, numeric, self._code_text.toPlainText()

    # ---------- library (current logic) ----------
    def _catalog_data(self) -> dict[str, list[dict[str, Any]]]:
        merged = {category: list(models) for category, models in self._scanner.catalog().items()}
        for category, models in self._surface_scanner.catalog().items():
            merged.setdefault(category, []).extend(models)
            merged[category].sort(key=lambda item: str(item.get("name", "")).lower())
        return dict(sorted(merged.items(), key=lambda item: item[0].lower()))

    def _is_2d_function(self, payload: Mapping[str, Any] | None = None) -> bool:
        data = payload if payload is not None else self._selected_function
        return int((data or {}).get("ndim", 1)) == 2

    def _reload_function_tree(self) -> None:
        self._function_tree.blockSignals(True)
        try:
            self._function_tree.clear()
            for category, functions in self._catalog_data().items():
                parent = QTreeWidgetItem([category, ""])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                for payload in functions:
                    child = QTreeWidgetItem([str(payload.get("name", "")), str(payload.get("description", ""))])
                    child.setData(0, Qt.ItemDataRole.UserRole, payload); parent.addChild(child)
                self._function_tree.addTopLevelItem(parent)
            self._function_tree.expandAll(); self._function_tree.resizeColumnToContents(0)
        finally:
            self._function_tree.blockSignals(False)
        first = self._first_function_item()
        if first is not None: self._function_tree.setCurrentItem(first)

    def _first_function_item(self) -> QTreeWidgetItem | None:
        for index in range(self._function_tree.topLevelItemCount()):
            parent = self._function_tree.topLevelItem(index)
            if parent is not None and parent.childCount(): return parent.child(0)
        return None

    def _on_function_changed(self, current: Any, _previous: Any = None) -> None:
        if self._model_tabs.currentIndex() != 0: return
        payload = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        if not isinstance(payload, dict): return
        self._selected_function = payload
        self._rebuild_params_table(payload)
        self._expression_label.setText(str(payload.get("expression", "") or ""))
        set_doc_link(self._doc_link, str(payload.get("name", "")), str(payload.get("doc_url", "") or ""))
        self.mark_results_stale()

    def _rebuild_params_table(self, payload: Mapping[str, Any]) -> None:
        names = [str(n) for n in (payload.get("params") or [])]
        defaults = [float(v) for v in (payload.get("p0") or [])]
        defaults += [1.0] * max(0, len(names) - len(defaults))
        self._set_param_rows(names, defaults)

    def _rebuild_param_names(self, names: list[str]) -> None:
        old, vals = self._function_params(); previous = dict(zip(old, vals.tolist()))
        self._set_param_rows(names, [previous.get(n, 1.0) for n in names])

    def _set_param_rows(self, names: list[str], values: list[float]) -> None:
        self._params_table.blockSignals(True)
        try:
            self._params_table.setRowCount(len(names))
            for row, name in enumerate(names):
                label = QTableWidgetItem(name); label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._params_table.setItem(row, 0, label)
                self._params_table.setItem(row, 1, QTableWidgetItem(f"{values[row]:g}"))
        finally: self._params_table.blockSignals(False)

    def _function_params(self) -> tuple[list[str], np.ndarray]:
        names, values = [], []
        for row in range(self._params_table.rowCount()):
            label, cell = self._params_table.item(row, 0), self._params_table.item(row, 1)
            names.append(label.text() if label else f"p{row}")
            try: values.append(float(cell.text()) if cell else 0.0)
            except ValueError: values.append(0.0)
        return names, np.asarray(values, dtype=float)

    # ---------- persistence ----------
    def _refresh_stored_combo(self) -> None:
        current = self._current_function_name
        self._stored_combo.blockSignals(True)
        try:
            self._stored_combo.clear()
            self._stored_combo.addItem(_("Select stored function..."), "")
            try:
                ensure_table(self._repo)
                available = stored_rows(self._repo)
            except Exception as exc:  # Do not prevent the Function dialog from opening.
                available = []
                self._stored_combo.setToolTip(str(exc))
            else:
                self._stored_combo.setToolTip("")
            for row in available:
                self._stored_combo.addItem(row["function_name"], row["function_name"])
            if current:
                index = self._stored_combo.findData(current)
                self._stored_combo.setCurrentIndex(max(0, index))
        finally:
            self._stored_combo.blockSignals(False)

    def _stored_selected(self, index: int) -> None:
        if index <= 0:
            return
        name = str(self._stored_combo.itemData(index) or "")
        data = get_stored(self._repo, name)
        if data is not None:
            self._apply_serialised(data)

    def _new_function(self) -> None:
        name, ok = QInputDialog.getText(self, _("New function"), _("Function name:"))
        name = name.strip()
        if not ok or not name:
            return
        if not name.isidentifier():
            self._show_error(_("New function"), ValueError("Function name must be a valid Python identifier."))
            return
        if get_stored(self._repo, name) is not None:
            self._show_error(_("New function"), ValueError("A function with this name already exists. Select it from Stored."))
            return
        self._current_function_name = name
        if self._model_tabs.currentIndex() == 0:
            self._model_tabs.setCurrentIndex(1)
        self._expr_symbols.clear(); self._expr_description.clear(); self._expr_text.clear()
        self._code_description.clear(); self._code_text.clear()
        self._stored_combo.setCurrentIndex(0)
        self._custom_model_changed()

    def _current_serialised(self) -> dict[str, str]:
        name = self._current_function_name.strip()
        if not name:
            raise ValueError("Create a new function or select a stored function first.")
        idx = self._model_tabs.currentIndex()
        names, numbers = self._function_params()
        values = dict(zip(names, numbers.tolist()))
        if idx == 1:
            return {"values": values, "function_name": name, "source_type": "expression",
                    "symbols": self._expr_symbols.text().strip(), "code_string": self._expr_text.toPlainText(),
                    "description": self._expr_description.toPlainText(),
                    "from": str(self.parameter_values().get("start", 0.0)),
                    "to": str(self.parameter_values().get("stop", 10.0)),
                    "points": str(self.parameter_values().get("points", 200)),
                    "y_from": str(self.parameter_values().get("y_start", 0.0)),
                    "y_to": str(self.parameter_values().get("y_stop", 10.0)),
                    "y_points": str(self.parameter_values().get("y_points", 200))}
        if idx == 2:
            return {"values": values, "function_name": name, "source_type": "code",
                    "symbols": ",".join(self._validated_code_signature()[1]), "code_string": self._code_text.toPlainText(),
                    "description": self._code_description.toPlainText(),
                    "from": str(self.parameter_values().get("start", 0.0)),
                    "to": str(self.parameter_values().get("stop", 10.0)),
                    "points": str(self.parameter_values().get("points", 200)),
                    "y_from": str(self.parameter_values().get("y_start", 0.0)),
                    "y_to": str(self.parameter_values().get("y_stop", 10.0)),
                    "y_points": str(self.parameter_values().get("y_points", 200))}
        raise ValueError("Library functions are read-only. Use New to create a stored function.")

    def _save_current(self) -> None:
        try:
            data = self._current_serialised(); self._custom_callable()
            exists = get_stored(self._repo, data["function_name"]) is not None
            if exists:
                answer = QMessageBox.question(
                    self, _("Modify stored function"),
                    _("Function already exists. Confirm modifications?"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            save_stored(self._repo, data)
            self._refresh_stored_combo()
        except Exception as exc:
            self._show_error(_("Save function"), exc)

    def _apply_serialised(self, data: Mapping[str, str]) -> None:
        self._current_function_name = str(data.get("function_name", ""))
        kind = str(data.get("source_type", "")).lower()
        if kind == "expression":
            self._model_tabs.setCurrentIndex(1)
            symbols = [token.strip() for token in str(data.get("symbols", "")).split(",") if token.strip() not in {"", "x", "y", "z"}]
            self._expr_symbols.setText(",".join(symbols))
            self._expr_description.setPlainText(data.get("description", ""))
            self._expr_text.setPlainText(data.get("code_string", ""))
        elif kind == "code":
            self._model_tabs.setCurrentIndex(2)
            self._code_description.setPlainText(data.get("description", ""))
            self._code_text.setPlainText(data.get("code_string", ""))
        else:
            raise ValueError(f"Unsupported stored source type: {kind}")
        range_values = {
            "start": float(data.get("from", "0") or 0),
            "stop": float(data.get("to", "10") or 10),
            "points": int(float(data.get("points", "200") or 200)),
            "y_start": float(data.get("y_from", "0") or 0),
            "y_stop": float(data.get("y_to", "10") or 10),
            "y_points": int(float(data.get("y_points", "200") or 200)),
        }
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.set_values(range_values)
        chart_form = getattr(self, "chart_form", None)
        if chart_form is not None and hasattr(chart_form, "set_values"):
            chart_form.set_values(range_values)
        self._custom_model_changed()
        stored = stored_values(data)
        if stored:  # the parameters at the values kept with the function
            names, numbers = self._function_params()
            self._set_param_rows(names, [stored.get(name, float(value)) for name, value in zip(names, numbers)])

    def _export_text(self) -> None:
        try:
            data = self._current_serialised(); self._check_without_running()
            path, selected_filter = QFileDialog.getSaveFileName(self, _("Export function"), f'{data["function_name"]}.pysym.txt', "PySym text (*.pysym.txt *.txt)")
            if path:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("PYSYM/1\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        except Exception as exc: self._show_error(_("Export function"), exc)

    def _import_text(self) -> None:
        try:
            path, selected_filter = QFileDialog.getOpenFileName(self, _("Import function"), "", "PySym text (*.pysym.txt *.txt);;All files (*)")
            if not path: return
            with open(path, "r", encoding="utf-8") as fh: raw = fh.read()
            if not raw.startswith("PYSYM/1\n"): raise ValueError("Not a PYSYM/1 text file.")
            self._apply_serialised(json.loads(raw.split("\n", 1)[1]))
            self._check_without_running()  # validate, but deliberately do not auto-save
        except Exception as exc: self._show_error(_("Import function"), exc)

    def _show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, str(exc))

    # ---------- computation ----------
    def compute_results(self) -> list[FunctionResult]:
        if self._model_tabs.currentIndex() == 0:
            payload = self._selected_function
            if not payload: return []
            if self._is_2d_function(payload): return self._compute_surface_result(payload)
            params = self.parameter_values(); x_values = self._build_range(params)
            names, values = self._function_params()
            model = self._scanner.make_model(dict(payload)); curve = evaluate_curve(model, x_values, values)
            metadata: dict[str, Any] = {"spacing": str(params.get("spacing", SPACING_LINEAR)), "range": f"{x_values[0]:g} .. {x_values[-1]:g}"}
            if curve.undefined: metadata["undefined"] = curve.undefined
            name = str(payload.get("name", "function"))
            return [FunctionResult(name, name, curve.x, curve.y,
                                   dict(zip(names, map(float, values))), str(payload.get("expression", "") or ""), metadata)]

        params = self.parameter_values(); x_values = self._build_range(params)
        names, values = self._function_params(); name, func, expression = self._custom_callable()
        # Call directly so Expression/Code does not depend on scanner model classes.
        y = np.asarray(func(x_values, *values), dtype=float)
        if y.ndim == 0: y = np.full_like(x_values, float(y), dtype=float)
        try: y = np.broadcast_to(y, x_values.shape).astype(float, copy=False)
        except ValueError as exc: raise ValueError("Function result must be scalar or have the same shape as x.") from exc
        finite = np.isfinite(x_values) & np.isfinite(y)
        metadata = {"spacing": str(params.get("spacing", SPACING_LINEAR)), "range": f"{x_values[0]:g} .. {x_values[-1]:g}"}
        dropped = int((~finite).sum())
        if dropped: metadata["undefined"] = dropped
        stored, source = self._live_source(params)
        return [FunctionResult(name, name, x_values[finite], y[finite],
                               dict(zip(names, map(float, values))), expression, metadata,
                               stored=stored, source_sql=source)]

    def _live_source(self, params: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """The function in the editor as it will be stored, and its live series' query - over the range chosen.

        None and "" when it has no name to be kept under: drawn as a table then, as a library function is.
        """
        try:
            stored = self._current_serialised()
            grid = GridRange(float(params.get("start", 0.0)), float(params.get("stop", 10.0)),
                             int(params.get("points", 200)), str(params.get("spacing", SPACING_LINEAR)))
            return stored, stored_source_sql(stored, stored_values(stored), grid)
        except (ValueError, SyntaxError):
            return None, ""

    def store_function(self, result: FunctionResult) -> None:
        """Keep the function a live series draws: saved inside the operation, so Cancel takes it back too."""
        if result.stored is not None:
            save_stored(self._repo, result.stored)
            self._refresh_stored_combo()

    def _compute_surface_result(self, payload: Mapping[str, Any]) -> list[FunctionResult]:
        params = self.parameter_values()
        x_lin = self._build_range(params)
        y_lin = build_range(
            float(params.get("y_start", 0.0)),
            float(params.get("y_stop", 10.0)),
            int(params.get("y_points", 200)),
            str(params.get("spacing", SPACING_LINEAR)),
        )
        names, values = self._function_params()
        model = self._surface_scanner.make_model(dict(payload))
        x_grid, y_grid = np.meshgrid(x_lin, y_lin, indexing="xy")
        independent = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        z_flat = np.asarray(model(independent, values), dtype=float).reshape(-1)
        if z_flat.size == 1:
            z_flat = np.full(independent.shape[0], float(z_flat[0]), dtype=float)
        if z_flat.size != independent.shape[0]:
            raise ValueError("Surface result must be scalar or match the X by Y sampling grid.")
        finite = np.isfinite(independent[:, 0]) & np.isfinite(independent[:, 1]) & np.isfinite(z_flat)
        undefined = int((~finite).sum())
        metadata: dict[str, Any] = {
            "spacing": str(params.get("spacing", SPACING_LINEAR)),
            "range": (
                f"x: {x_lin[0]:g} .. {x_lin[-1]:g} ({x_lin.size}); "
                f"y: {y_lin[0]:g} .. {y_lin[-1]:g} ({y_lin.size})"
            ),
        }
        if undefined:
            metadata["undefined"] = undefined
        name = str(payload.get("name", "function"))
        return [FunctionResult(
            name, name, independent[finite, 0], independent[finite, 1],
            dict(zip(names, map(float, values))),
            str(payload.get("expression", "") or ""), metadata, z_flat[finite],
        )]

    @staticmethod
    def _build_range(params: Mapping[str, Any]) -> np.ndarray:
        return build_range(float(params.get("start", 0.0)), float(params.get("stop", 10.0)),
                           int(params.get("points", 200)), str(params.get("spacing", SPACING_LINEAR)))

    # ---------- unchanged result integration ----------
    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        name = str(results[0].function_name) if results else "Function"
        is_surface = bool(results and results[0].z is not None)
        requested = str(self.parameter_values().get("chart_type", "Scatter Plot"))
        destination = str(self.parameter_values().get("destination", self.DEFAULT_DESTINATION))

        # Same axis must preserve the existing chart. For a newly-created target,
        # use the chosen type when dimensionally compatible, otherwise the safe default.
        if destination == self.DEST_SAME_AXIS:
            return selected_axis_id
        if is_surface:
            allowed = {"Surface Plot", "Wireframe", "3D Scatter", "Contour", "Filled Contour"}
            chart_type = requested if requested in allowed else "Surface Plot"
            options: dict[str, Any] = {"grid": True, "projection": "3d"}
            if chart_type in {"Contour", "Filled Contour"}:
                # Contours are rendered from x/y/z but do not require a 3D projection.
                options.pop("projection", None)
        else:
            chart_type = "Scatter Plot"
            options = {"grid": True, "linestyle": "-", "marker": ""}
        return self.resolve_destination_axis(
            selected_axis_id,
            chart_type=chart_type,
            title=name,
            figure_name=name,
            options=options,
        )

    def selected_series(self) -> list[Any]: return []

    def result_series_spec(self, axis_id: int, table_name: str, result: FunctionResult) -> ResultSeriesSpec:
        del axis_id
        if result.source_sql:
            source = DataSource.function(result.function_name, result.source_sql).from_clause()
            return ResultSeriesSpec(name=result.result_name, sql_query=f'SELECT "x" AS "x", "y" AS "y" FROM {source}',
                                    roles={"x": "x", "y": "y"},
                                    style={**self.generated_style_filter, "function": result.function_name,
                                           "linestyle": "-", "linewidth": 1.6, "marker": ""})
        if result.z is not None:
            return ResultSeriesSpec(name=result.result_name, sql_query=f'SELECT x, y, z FROM "{table_name}"',
                                    roles={"x": "x", "y": "y", "z": "z"},
                                    style={**self.generated_style_filter, "function": result.function_name, "fit_mode": "2D"})
        return ResultSeriesSpec(name=result.result_name, sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
                                roles={"x": "x", "y": "y"},
                                style={**self.generated_style_filter, "function": result.function_name,
                                       "linestyle": "-", "linewidth": 1.6, "marker": ""})

    def result_table_name(self, axis_id: int, result: FunctionResult) -> str:
        return generated_table_name(f"Function_axis{axis_id}_{result.function_name}", fallback="Function_Result")

    def results_subtitle(self, results: Sequence[FunctionResult]) -> str: return ""

    def result_section(self, result: FunctionResult) -> str:
        values = result.z if result.z is not None else result.y; is_surface = result.z is not None
        finite = np.isfinite(values)
        rows: list[tuple[str, Any]] = [(_("Function"), result.function_name), (_("Points"), values.size),
                                      (_("Range"), result.metadata.get("range", "")),
                                      (_("Spacing"), result.metadata.get("spacing", ""))]
        if finite.any():
            rows.append((_("z range") if is_surface else _("y range"),
                         f"{report_html.format_number(float(np.min(values[finite])))} .. "
                         f"{report_html.format_number(float(np.max(values[finite])))}"))
        if "undefined" in result.metadata: rows.append((_("Undefined points"), result.metadata["undefined"]))
        blocks = [report_html.summary_table(rows)]
        if result.expression: blocks.append(report_html.raw_note(result.expression))
        if result.params:
            blocks.append(report_html.table((_("Parameter"), _("Value")),
                          [(n, report_html.format_number(v)) for n, v in result.params.items()],
                          align=("left", "right"), empty_message=_("This function takes no parameters.")))
        return report_html.section(result.function_name, *blocks)
