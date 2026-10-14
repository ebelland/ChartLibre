"""Global fit: one model over several series, some parameters shared (R-05).

Checking two or more series for a 1D model makes the fit global, and a
Shared row appears under the table - a menu of the parameters, since a
column of boxes did not fit the panel's width: a shared parameter is one row
of the table,
fitted to all the series together; an unshared one is a row per series.
The table stays what is drawn - every row is one entry of the global vector
- so Fit, Preview and OK run as for one series, on the series stacked with
an index (``fit.global_model``), and the outcome is split back into a result
per series, each with its own curve, report and output table.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QTableWidgetItem, QWidget

from app.analysis import fit as fit_engine
from app.data.data_source import parse_roles, row_value
from app.utils.i18n import _
from app.series_operations.dialog_base import generated_table_name

class GlobalFitMixin:
    """Global fit with shared parameters. A part of SeriesFitDialog (see dialog.py)."""

    # -- which layout ---------------------------------------------------

    def _global_rows(self) -> list[Any]:
        """The checked series when the fit is global (two or more, 1D), else []."""
        if self._is_2d_fit():
            return []
        rows = list(self.selected_series())
        return rows if len(rows) >= 2 else []

    def _fit_is_global(self) -> bool:
        """True when the checked series are fitted together (the default).

        False fits every series on its own: the layout then has no shared
        parameter, so each series gets a row per parameter and its own result.
        """
        return bool(getattr(self, "_global_fit_wanted", True))

    def _set_fit_mode(self, index: int) -> None:
        self._global_fit_wanted = index == 0
        self._relayout_params_table()
        self.mark_results_stale()

    @staticmethod
    def _series_label(row: Any) -> str:
        return str(parse_roles(row_value(row, "roles", default={})).get("name")
                   or row_value(row, "name", "series_name", default="Series"))

    def _wanted_layout(self) -> fit_engine.GlobalLayout | None:
        rows = self._global_rows()
        if not rows:
            return None
        n_params = len(self._param_defaults)
        shared = list(getattr(self, "_shared", []))[:n_params]
        shared += [False] * (n_params - len(shared))
        # What was picked is kept while the series are fitted one by one, so
        # switching back to a global fit finds the same shared parameters.
        self._shared = shared
        if not self._fit_is_global():
            shared = [False] * n_params
        self._global_names = [self._series_label(row) for row in rows]
        return fit_engine.GlobalLayout(tuple(shared), len(rows))

    def _global_label(self, row: int) -> str | None:
        """The table label of *row* in a global fit, None otherwise."""
        layout = getattr(self, "_table_layout", None)
        if layout is None:
            return None
        entries = layout.entries()
        if row >= len(entries):
            return None
        column, series = entries[row]
        name = self._param_names[column] if column < len(self._param_names) else f"p[{column}]"
        if series is None:
            return name
        names = getattr(self, "_global_names", [])
        return f"{name} · {names[series] if series < len(names) else series + 1}"

    # -- the table ------------------------------------------------------

    def _build_shared_row(self) -> QWidget:
        """'Shared: A, offset' with a menu of the parameters; global fits only."""
        self._shared_row = QWidget(self)
        row_layout = QHBoxLayout(self._shared_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._fit_mode_combo = QComboBox(self._shared_row)
        self._fit_mode_combo.addItem(_("Global fit"))
        self._fit_mode_combo.addItem(_("Each curve separately"))
        self._fit_mode_combo.setToolTip(
            _("Global: one fit of all the checked series, with the parameters picked in Shared "
              "having one value for all. Separately: every series gets its own parameters and result.")
        )
        self._fit_mode_combo.currentIndexChanged.connect(self._set_fit_mode)
        row_layout.addWidget(self._fit_mode_combo)
        row_layout.addWidget(QLabel(_("Shared:"), self._shared_row))
        self._shared_button = QPushButton(self._shared_row)
        self._shared_button.setToolTip(_("One value for every series, fitted to all of them together."))
        self._shared_menu = QMenu(self._shared_button)
        self._shared_button.setMenu(self._shared_menu)
        row_layout.addWidget(self._shared_button, 1)
        self._shared_row.hide()
        return self._shared_row

    def _refresh_shared_row(self, layout: fit_engine.GlobalLayout | None) -> None:
        self._shared_row.setVisible(layout is not None)
        if layout is None:
            return
        self._shared_button.setEnabled(self._fit_is_global())
        # The layout has no shared parameter when fitting separately, so the
        # menu shows what is kept for the global fit instead of the layout.
        shared_now = list(self._shared)
        shared_now += [False] * (len(layout.shared) - len(shared_now))
        names = [
            self._param_names[column] if column < len(self._param_names) else f"p[{column}]"
            for column in range(len(layout.shared))
        ]
        actions = self._shared_menu.actions()
        if [action.text() for action in actions] != names:
            # Rebuilt only for another model: never from inside one of its
            # own actions' toggled signal, which would delete the sender.
            self._shared_menu.clear()
            for column, name in enumerate(names):
                action = self._shared_menu.addAction(name)
                action.setCheckable(True)
                action.toggled.connect(lambda checked, column=column: self._set_shared(column, checked))
            actions = self._shared_menu.actions()
        for action, shared in zip(actions, shared_now):
            blocked = action.blockSignals(True)
            action.setChecked(shared)
            action.blockSignals(blocked)
        names = [action.text() for action in actions if action.isChecked()]
        self._shared_button.setText(", ".join(names) if names else _("none"))

    def _set_shared(self, column: int, shared: bool) -> None:
        self._shared[column] = bool(shared)
        self._relayout_params_table()

    def _relayout_params_table(self) -> None:
        """Lay the table out for the series checked now, keeping its values.

        Values move by (parameter, series): a parameter made shared keeps the
        first series' row, one made unshared copies its value to every series,
        and a series checked anew starts from the first series' values.
        """
        if not hasattr(self, "_param_defaults"):
            return
        new = self._wanted_layout()
        old = getattr(self, "_table_layout", None)
        self._refresh_shared_row(new)
        if new == old:
            return
        columns = self._collect_params_from_table()
        per_series = [values[None, :] if old is None else values[old.index] for values in columns]
        n_new = new.n_series if new is not None else 1
        moved = [
            np.vstack([values[min(series, values.shape[0] - 1)] for series in range(n_new)])
            for values in per_series
        ]
        rows = [new.expand(values) if new is not None else values[0] for values in moved]
        self._table_layout = new
        self._ensure_params_rows(len(rows[0]))
        p0, lower, upper, fixed = rows
        for row in range(len(p0)):
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(p0[row]):.12g}"))
            self._params_table.setItem(row, 2, QTableWidgetItem(_bound_text(lower[row])))
            self._params_table.setItem(row, 3, QTableWidgetItem(_bound_text(upper[row])))
            self._get_fix_checkbox(row).setChecked(bool(fixed[row]))
        self.mark_results_stale()

    def _expand_to_table(self, values: np.ndarray) -> np.ndarray:
        """One series' parameter values as the table's rows (the same for all)."""
        layout = getattr(self, "_table_layout", None)
        values = np.asarray(values, dtype=float)
        if layout is None or values.size != len(layout.shared):
            return values
        return layout.expand(np.tile(values, (layout.n_series, 1)))

    def _combine_estimates(self, estimates: list[np.ndarray | None]) -> np.ndarray | None:
        """Each series' own estimate as the table's rows; shared ones averaged."""
        layout = getattr(self, "_table_layout", None)
        current = self._collect_params_from_table()[0]
        per_series = current[layout.index].copy()
        found = [index for index, values in enumerate(estimates) if values is not None]
        if not found:
            return None
        for series in found:
            per_series[series] = estimates[series]
        mean = np.mean([estimates[series] for series in found], axis=0)
        theta = layout.expand(per_series)
        for row, (column, series) in enumerate(layout.entries()):
            if series is None:
                theta[row] = mean[column]
        return theta

    # -- the data -------------------------------------------------------

    def _load_global_fit_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Every checked series, stacked: x as (value, series), clean with a series column."""
        rows = self._global_rows()
        frames = []
        for number, row in enumerate(rows):
            _x, _y, clean = self._load_fit_data_1d(row)
            clean = clean.copy()
            clean["series"] = number
            clean["series_name"] = self._series_label(row)
            frames.append(clean)
        clean = pd.concat(frames, ignore_index=True)
        x_data = clean[["x", "series"]].to_numpy(dtype=float)
        return x_data, clean["target"].to_numpy(dtype=float), clean

    def _per_series_xy(self) -> list[tuple[np.ndarray, np.ndarray]]:
        _x, _y, clean = self._load_global_fit_data()
        return [
            (part["x"].to_numpy(float), part["target"].to_numpy(float))
            for _number, part in clean.groupby("series", sort=True)
        ]

    # -- the outcome, series by series ----------------------------------

    def _split_global(self, job: Any, outcome: fit_engine.CurveFit) -> list[tuple[Any, fit_engine.CurveFit]]:
        base_model, layout, names = job.global_parts
        shared_names = [self._param_names[column] for column, shared in enumerate(layout.shared) if shared]
        note = (
            _("Global fit of {count} series") if self._fit_is_global()
            else _("Each of {count} series fitted separately")
        ).format(count=layout.n_series)
        note += (" - " + _("shared: {names}").format(names=", ".join(shared_names))) if shared_names else ""
        series_of_point = np.asarray(job.x, dtype=float)[:, 1].astype(int)
        parts = []
        for number in range(layout.n_series):
            index = layout.index[number]
            mask = series_of_point == number
            target = job.target[mask]
            fit_values = outcome.fit_values[mask]
            if job.parts is not None:
                # Fitted on its own: its own metrics, errors and degrees of freedom.
                own = job.parts[number]
                metrics = dict(own.metrics)
            else:
                metrics = fit_engine.fit_metrics(target, fit_values, int(np.count_nonzero(~job.fixed[index])))
                # Bands use the fit's pooled residual variance, the one its errors come from.
                if "reduced_chi2" in outcome.metrics:
                    metrics["reduced_chi2"] = outcome.metrics["reduced_chi2"]
                for key in ("chi2", "chi2_reduced"):
                    if key in outcome.metrics:
                        metrics[f"global_{key}"] = outcome.metrics[key]
            square = np.ix_(index, index)
            part_job = replace(
                job,
                model=base_model,
                x=np.asarray(job.x, dtype=float)[mask, 0],
                target=target,
                clean=job.clean[mask].reset_index(drop=True),
                p0=job.p0[index],
                lower=job.lower[index],
                upper=job.upper[index],
                fixed=job.fixed[index],
                source_table=names[number],
                param_names=list(self._param_names),
                output_table=generated_table_name(f"{job.output_table}_{names[number]}", fallback="Fit_Result"),
                sigma=None if job.sigma is None else job.sigma[mask],
                global_parts=None,
            )
            if job.parts is not None:
                part = replace(job.parts[number], message=f"{note}. {job.parts[number].message}")
                parts.append((part_job, part))
                continue
            part = fit_engine.CurveFit(
                params=outcome.params[index],
                std=outcome.std[index],
                cov=outcome.cov[square],
                corr=outcome.corr[square],
                fit_values=fit_values,
                residual=outcome.residual[mask],
                metrics=metrics,
                success=outcome.success,
                message=f"{note}. {outcome.message}",
                dof=outcome.dof,
                tvalues=outcome.tvalues[index],
                pvalues=outcome.pvalues[index],
                ci_low=outcome.ci_low[index],
                ci_high=outcome.ci_high[index],
            )
            parts.append((part_job, part))
        return parts


def _bound_text(value: float) -> str:
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{float(value):.12g}"
