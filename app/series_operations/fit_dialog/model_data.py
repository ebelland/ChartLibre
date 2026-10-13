"""The chosen model, its parameters table, multi-peak models, and the data the fit reads."""
from __future__ import annotations

from html import escape as html_escape
import math
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QTableWidgetItem, QTreeWidgetItem
from app.analysis import fit as fit_engine

from app.data.data_source import row_value, parse_roles, resolve_role_column
from app.logs.logger import applogger

from app.utils.i18n import _


class ModelDataMixin:
    """The chosen model, its parameters table, multi-peak models, and the data the fit reads. A part of SeriesFitDialog (see dialog.py)."""

    def _catalog_data(self) -> dict[str, list[dict[str, Any]]]:
        """Return scanned fit functions grouped by category.

        The previous JSON catalog is intentionally removed.  All fit models are
        now function classes discovered from app/functions/functions.py and
        app/functions/user_functions.py through FunctionScanner, plus every
        z = f(x, y) surface function from app/functions/surface_functions.py
        and app/functions/user_surface_functions.py through
        SurfaceFunctionScanner. The two catalogs are merged by category - a
        surface function's own categories ("Surfaces", "User surfaces") show
        up as ordinary top-level groups in the same tree, no separate UI
        needed.
        """
        merged: dict[str, list[dict[str, Any]]] = {
            category: list(models) for category, models in self._function_scanner.catalog().items()
        }
        for category, models in self._surface_scanner.catalog().items():
            merged.setdefault(category, []).extend(models)
            merged[category].sort(key=lambda item: str(item.get("name", "")).lower())
        return dict(sorted(merged.items(), key=lambda item: item[0].lower()))

    def _build_model_catalog(self) -> None:
        self._models_tree.clear()
        for cat_name, models in self._catalog_data().items():
            # Translated for display only. The payload keeps the English name,
            # which is what a saved fit refers to and what the search below
            # falls back on - a catalogue whose identities changed with the
            # interface language would lose every fit saved in another one.
            cat = QTreeWidgetItem([_(str(cat_name))])
            self._models_tree.addTopLevelItem(cat)
            for payload in models:
                item = QTreeWidgetItem([_(str(payload["name"]))])
                item.setData(0, Qt.ItemDataRole.UserRole, payload)
                cat.addChild(item)
            cat.setExpanded(False)

    def _filter_model_catalog(self, text: str) -> None:
        needle = text.lower().strip()
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            any_visible = False
            if cat is None:
                continue
            for j in range(cat.childCount()):
                child = cat.child(j)
                match = needle in child.text(0).lower()
                child.setHidden(not match)
                any_visible = any_visible or match
            cat.setHidden(not any_visible)
            cat.setExpanded(bool(needle) and any_visible)

    def _select_first_model(self) -> None:
        """Select Linear by default, or the first model when there is none."""
        if self.select_model("Linear"):
            return
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            if cat is not None and cat.childCount():
                cat.setExpanded(True)
                self._models_tree.setCurrentItem(cat.child(0))
                return

    def select_model(self, name: str) -> bool:
        """Select the catalogue model called *name*; False when there is none."""
        wanted = name.strip().lower()
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            if cat is None:
                continue
            for j in range(cat.childCount()):
                child = cat.child(j)
                payload = child.data(0, Qt.ItemDataRole.UserRole)
                label = payload.get("name", child.text(0)) if isinstance(payload, dict) else child.text(0)
                if str(label).strip().lower() == wanted:
                    cat.setExpanded(True)
                    self._models_tree.setCurrentItem(child)
                    return True
        return False

    def _on_model_tree_selection(self) -> None:
        item = self._models_tree.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            self._apply_model_choice_from_payload(payload)

    def _hide_model_options(self) -> None:
        for w in (self._lbl_degree, self._spin_degree, self._lbl_knots, self._spin_knots, self._lbl_spacing, self._combo_knot_spacing):
            w.hide()    

    def _current_table(self) -> str:
        """Return a stable source table label for results/reporting."""
        return self._initial_table or self._source_name or "selected_series"

    def _is_2d_fit(self) -> bool:
        """True when the selected model is a surface function z = f(x, y).

        Driven entirely by the selected model's declared ``ndim`` (see
        ``FitFunctionSpec``/``SurfaceFunctionScanner``), not by anything
        about the selected series: a 2-variable model always needs a second
        independent variable, and a 1-variable one never uses one, whatever
        roles the series happens to have.
        """
        return int(self._selected_model.get("ndim", 1)) == 2

    def _selected_column_names(self) -> tuple[str, str | None, str]:
        """Return source X, optional X2, and target column names."""
        if self._is_2d_fit():
            return (
                self._source_x_col or "x",
                self._source_x2_col or "y",
                self._source_y_col or "z",
            )
        return self._source_x_col or "x", None, self._source_y_col or "target"

    def _apply_model_choice_from_payload(self, payload: dict[str, Any]) -> None:
        """Apply one scanned function class to the dialog.

        All selectable models now come from ``FunctionScanner``.  There are no
        expression, rational, orthopoly, spline, or JSON catalogue branches here.
        Function metadata is the single source of truth.
        """
        self._hide_model_options()
        self._selected_model = dict(payload)
        self._param_names = [str(name) for name in payload.get("params", [])]
        self._refresh_param_legend()

        p0 = [float(v) for v in payload.get("p0", [1.0, 1.0])]
        self._param_defaults = p0
        self._spin_params.setValue(len(p0))
        self._ensure_params_rows(len(p0))
        self._populate_params_defaults()
        # Then improve on them wherever the function can: see
        # _fill_initial_from_estimator.
        self._fill_initial_from_estimator()

        expression_html = str(payload.get("expression", "")).strip()
        description = _(str(payload.get("description", "")).strip())
        if description:
            description_html = f'<div style="color:#666; margin-top:6px;">{html_escape(description)}</div>'
            expression_html = f"{expression_html}{description_html}" if expression_html else description_html

        if expression_html:
            self._function_expression_html.setText(expression_html)
            self._function_expression_html.show()
        else:
            self._function_expression_html.clear()
            self._function_expression_html.hide()

        self._refresh_default_output_name()

    def _on_degree_changed(self, val: int) -> None:
        del val

    def _on_knots_changed(self, val: int) -> None:
        del val

    def _param_label(self, row: int) -> str:
        """Return the human-readable parameter name for the table/report."""
        global_label = self._global_label(row)
        if global_label is not None:
            return global_label
        name = self._param_names[row] if row < len(self._param_names) else ""
        return name if name else f"p[{row}]"

    def _refresh_param_legend(self) -> None:
        """Show the parameter meanings under the expression."""
        if not self._param_names:
            self._param_legend.clear()
            self._param_legend.hide()
            return

        self._param_legend.setText("   ".join(self._param_names))
        self._param_legend.show()

    def _ensure_params_rows(self, count: int) -> None:
        """Grow or shrink the parameter table, keeping the values already typed."""
        self._params_table.setRowCount(int(count))
        for row in range(int(count)):
            # Column 0 is rewritten every time rather than only when missing:
            # it carries the parameter's meaning, which changes with the model.
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            for col, text in ((1, "0.0"), (2, "-inf"), (3, "inf")):
                if self._params_table.item(row, col) is None:
                    self._params_table.setItem(row, col, QTableWidgetItem(text))
            if self._params_table.cellWidget(row, 4) is None:
                self._params_table.setCellWidget(row, 4, QCheckBox(self._params_table))
        # Not resizeColumnsToContents(): it widened the Stretch columns to
        # their longest value until the next resize, pushing Fix out of view.

    def _populate_params_defaults(self) -> None:
        # The model's own rows first; a global fit then lays them out per series.
        self._table_layout = None
        self._shared = []
        self._ensure_params_rows(len(self._param_defaults))
        for row, value in enumerate(self._param_defaults):
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._params_table.setItem(row, 1, QTableWidgetItem(str(float(value))))
            self._params_table.setItem(row, 2, QTableWidgetItem("-inf"))
            self._params_table.setItem(row, 3, QTableWidgetItem("inf"))
            cb = self._get_fix_checkbox(row)
            cb.setChecked(False)
        self._relayout_params_table()

    def _get_fix_checkbox(self, row: int) -> QCheckBox:
        widget = self._params_table.cellWidget(row, 4)
        if not isinstance(widget, QCheckBox):
            widget = QCheckBox(self._params_table)
            self._params_table.setCellWidget(row, 4, widget)
        return widget


    def on_multi_peak(self) -> None:
        """Create a configurable multi-peak model from the inline Model frame."""
        family = self._multi_family_combo.currentText()
        count = int(self._multi_count_spin.value())
        tie_width = bool(self._multi_tie_width.isChecked())
        tie_eta = bool(self._multi_tie_eta.isChecked())

        params: list[str] = []
        p0: list[float] = []
        for i in range(count):
            params.extend([f"amp {i + 1}", f"center {i + 1}"])
            p0.extend([1.0, float(i)])
            if not tie_width:
                params.append(f"width {i + 1}")
                p0.append(1.0)
            if family == "Pseudo-Voigt" and not tie_eta:
                params.append(f"eta {i + 1}")
                p0.append(0.5)
        if tie_width:
            params.append("shared width")
            p0.append(1.0)
        if family == "Pseudo-Voigt" and tie_eta:
            params.append("shared eta")
            p0.append(0.5)
        params.append("offset")
        p0.append(0.0)

        formula = {
            "Gaussian": "Σ Aᵢ exp(-(x-cᵢ)²/(2wᵢ²)) + C",
            "Lorentzian": "Σ Aᵢ (0.5wᵢ)² / ((x-cᵢ)² + (0.5wᵢ)²) + C",
            "Pseudo-Voigt": "Σ Aᵢ [ηᵢ Lᵢ(x) + (1-ηᵢ) Gᵢ(x)] + C",
        }.get(family, "multi-peak")

        self._selected_model = {
            "name": f"Multi {family} ({count})",
            "_multi_peak": True,
            "family": family,
            "count": count,
            "tie_width": tie_width,
            "tie_eta": tie_eta,
            "params": params,
            "p0": p0,
            "expression": f"<b>Multi {family}</b><br>{formula}",
            "description": "Configurable multi-peak model built from the inline controls.",
        }
        self._param_names = params
        self._refresh_param_legend()
        self._param_defaults = p0
        self._spin_params.setValue(len(p0))
        self._ensure_params_rows(len(p0))
        self._populate_params_defaults()
        self._function_expression_html.setText(str(self._selected_model["expression"]))
        self._function_expression_html.show()
        self._refresh_default_output_name(force=True)


    def _load_fit_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if self._is_2d_fit():
            return self._load_fit_data_2d()
        if self._global_rows():
            return self._load_global_fit_data()
        return self._load_fit_data_1d()

    def _load_fit_data_2d(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Read a series' x, y and z roles for a surface (2D) fit.

        A 2D fit's independent variables are not chosen through a combo: the
        series already names them through its own roles - x and y are the
        two independent variables, z is the target - which is what
        ``SeriesOperationDialogBase.series_xyz`` reads. No extra "X2" picker
        is needed for the common case this dialog supports (a series that
        already carries x/y/z), which is why none was added.
        """
        row = self._selected_series_row()
        roles = parse_roles(row_value(row, "roles", default={}))
        # A "name" role names the source; otherwise the series' own name.
        source_name = str(roles.get("name") or row_value(row, "name", "series_name", default="Series"))
        if not roles.get("z"):
            applogger.error(
                "A surface model needs a series with x, y and z roles; "
                "the selected series has no z role."
            )

        x_values, y_values, z_values = self.series_xyz(row, source_name)

        finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
        dropped = int(finite.size - int(np.count_nonzero(finite)))
        if dropped:
            applogger.warning(
                "%s: dropped %d row(s) with non-finite x/y/z before fitting.",
                source_name,
                dropped,
            )
        x_clean = x_values[finite]
        y_clean = y_values[finite]
        target_clean = z_values[finite]
        if target_clean.size < self.INPUT_MINIMUM_POINTS:
            applogger.error(
                f"{source_name}: not enough finite (x, y, z) points to fit a surface."
            )

        clean_frame = cast(
            pd.DataFrame,
            pd.DataFrame({"x": x_clean, "y": y_clean, "target": target_clean}),
        )

        self._source_name = source_name
        self._source_x_col = str(roles.get("x") or "x")
        self._source_x2_col = str(roles.get("y") or "y")
        self._source_y_col = str(roles.get("z") or "z")
        self._refresh_default_output_name(force=True)

        x_data = np.column_stack([x_clean, y_clean])
        target_data = target_clean
        return x_data, target_data, clean_frame

    def _load_fit_data_1d(self, row: Any = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        row = self._selected_series_row() if row is None else row
        source_name = str(
            parse_roles(row["roles"]).get("name") or row_value(row, "name", "series_name", default="Series")
        )
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            applogger.error("Selected series has no SQL query.")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            applogger.error("Selected series query returned no rows.")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = resolve_role_column(columns, roles, "x") or ""
        y_col = resolve_role_column(columns, roles, "y") or ""
        if x_col not in columns:
            x_col = numeric[0] if numeric else ""
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else ""

        if not x_col or not y_col:
            applogger.error("Selected series query must expose at least two numeric columns.")
        if x_col == y_col:
            applogger.error("Selected series X and Y columns must be different.")

        x_raw = self.numeric_x(frame[x_col], source_name)
        y_raw = self.numeric_y(frame[y_col])
        sigma = None
        if self._weights_key() == "errors":
            sigma = self._error_column(frame, columns, roles, source_name)
            x_raw, y_raw, sigma = _merge_points_with_errors(x_raw, y_raw, sigma, source_name)

        # Same three repairs as before - drop non-finite, sort by x, average
        # repeated x - but reported rather than silent. With errors they were
        # made above, carrying each point's error along.
        x_prepared, target_prepared = self.prepare_input_xy(x_raw, y_raw, label=source_name)

        clean_frame = cast(
            pd.DataFrame,
            pd.DataFrame({"x": x_prepared, "target": target_prepared}),
        )
        if sigma is not None:
            clean_frame["sigma"] = sigma

        self._source_name = source_name
        self._source_x_col = x_col
        self._source_y_col = y_col
        self._refresh_default_output_name(force=True)

        x_data = clean_frame["x"].to_numpy(dtype=float)
        target_data = clean_frame["target"].to_numpy(dtype=float)
        return x_data, target_data, clean_frame

    def _error_column(self, frame: pd.DataFrame, columns: list[str], roles: dict, source_name: str) -> np.ndarray:
        """Each point's y error: the yerr role, or the mean of yerr_low/yerr_high."""
        found = [resolve_role_column(columns, roles, role) for role in ("yerr", "yerr_low", "yerr_high")]
        if found[0] in columns:
            return self.numeric_y(frame[found[0]])
        halves = [self.numeric_y(frame[name]) for name in found[1:] if name in columns]
        if not halves:
            applogger.error(
                f"{source_name}: weighting by error bars needs a yerr column in the series' roles.",
                raise_error=True,
            )
        return np.mean(halves, axis=0)

    def _model_name(self) -> str:
        item = self._models_tree.currentItem()
        if item is not None and item.parent() is not None:
            return item.text(0)
        return str(self._selected_model.get("name", "Custom"))

    @staticmethod
    def _make_multi_peak_model(payload: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a callable for the inline multi-peak model."""
        return fit_engine.multi_peak_model(
            str(payload.get("family", "Gaussian")),
            int(payload.get("count", 1)),
            tie_width=bool(payload.get("tie_width", False)),
            tie_eta=bool(payload.get("tie_eta", False)),
        )

    def _build_model(self, x_data: np.ndarray, target_data: np.ndarray) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], np.ndarray] | None:
        """Build the selected scanned function model."""
        del x_data, target_data  # not needed for class-backed function models
        p0, _unused, _unused, _unused = self._collect_params_from_table()
        if self._selected_model.get("_multi_peak"):
            return self._make_multi_peak_model(self._selected_model), p0
        scanner = self._surface_scanner if self._is_2d_fit() else self._function_scanner
        try:
            return scanner.make_model(self._selected_model), p0
        except Exception:
            applogger.exception(
                "Failed to build scanned fit function: %s",
                self._selected_model.get("name", ""),
            )
            return None

    def _float_item(self, row: int, col: int, default: float) -> float:
        item = self._params_table.item(row, col)
        text = item.text().strip() if item is not None else ""
        if not text:
            return default
        low = text.lower()
        if low in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if low in ("-inf", "-infinity"):
            return -math.inf
        return float(text)

    def _collect_params_from_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self._params_table.rowCount()
        p0 = np.zeros(n, dtype=float)
        lb = np.full(n, -np.inf, dtype=float)
        ub = np.full(n, np.inf, dtype=float)
        fixed = np.zeros(n, dtype=bool)
        for row in range(n):
            p0[row] = self._float_item(row, 1, 0.0)
            lb[row] = self._float_item(row, 2, -np.inf)
            ub[row] = self._float_item(row, 3, np.inf)
            fixed[row] = self._get_fix_checkbox(row).isChecked()
            if lb[row] > ub[row]:
                applogger.error(f"Lower bound is greater than upper bound at p[{row}].")
            p0[row] = min(max(p0[row], lb[row]), ub[row])
        return p0, lb, ub, fixed

    def _set_initial_params(self, values: np.ndarray) -> None:
        for row, value in enumerate(self._expand_to_table(values)):
            if row < self._params_table.rowCount():
                self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))


def _merge_points_with_errors(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop points without a usable error, sort by x, merge repeated x.

    What prepare_input_xy does, done first so each point keeps its error:
    repeated x become their weighted mean, with the error of that mean.
    """
    x, y, sigma = (np.asarray(values, dtype=float) for values in (x, y, sigma))
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0.0)
    dropped = int(usable.size - np.count_nonzero(usable))
    if dropped:
        applogger.warning(f"{label}: {dropped} point(s) without a positive, finite error left out of the fit.")
    x, y, sigma = x[usable], y[usable], sigma[usable]
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size == x.size:
        order = np.argsort(x, kind="stable")
        return x[order], y[order], sigma[order]
    weight = 1.0 / np.square(sigma)
    total = np.bincount(inverse, weights=weight)
    merged_y = np.bincount(inverse, weights=weight * y) / total
    applogger.warning(f"{label}: {x.size - unique_x.size} repeated x merged into their weighted mean.")
    return unique_x, merged_y, 1.0 / np.sqrt(total)
