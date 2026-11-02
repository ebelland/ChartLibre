"""The SymPy functions demo: formulas stored with the project, drawn, differentiated, integrated and fitted.

Each function lives in the project's ``__sympy__`` table (Tables panel, marked F) and a chart of it is a
live query of the formula: its parameters are the series' own. Calculus differentiates and integrates a
function's series exactly - SymPy's formula, not the points' finite differences - and keeps the result as a
function of its own (Damped_d1, Logistic_int); Fit Curve offers the stored functions as models.

    python -m dev.demo.demo_functions
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.demos import DEMO_DIR, DEMO_PROJECTS, DemoProject
from app.data.repo.function_sql import GridRange, function_sql
from app.data.sqlite_repo import SqliteRepo
from app.data_operations import sympy_store
from app.logs.logger import applogger
from dev.demo.build_demos import demo_style
from dev.demo.demo_operations import (
    OperationDemo,
    _create_source_figure,
    _run_operation,
    _store_report,
)

#: The functions stored with the project: name, formula, parameters with their values, x range, description.
FUNCTIONS: tuple[tuple[str, str, dict[str, float], tuple[float, float], str], ...] = (
    ("Damped", "a*exp(-b*x)*cos(w*x)", {"a": 1.0, "b": 0.3, "w": 4.0}, (0.0, 10.0),
     "A damped oscillation: amplitude a, decay b, angular frequency w."),
    ("Logistic", "L/(1 + exp(-k*(x - x0)))", {"L": 100.0, "k": 1.2, "x0": 5.0}, (0.0, 10.0),
     "Logistic growth: ceiling L, rate k, midpoint x0."),
    ("Gaussian", "A*exp(-(x - mu)**2/(2*s**2))", {"A": 1.0, "mu": 0.0, "s": 1.0}, (-5.0, 5.0),
     "A bell curve: height A, centre mu, width s."),
)

_GRID = 201


def _store(repo: SqliteRepo) -> None:
    for name, code, values, (start, stop), description in FUNCTIONS:
        sympy_store.save(repo, {
            "function_name": name, "source_type": "expression", "symbols": ", ".join(values),
            "code_string": code, "from": str(start), "to": str(stop), "points": str(_GRID),
            "description": description, "values": values,
        })
    sympy_store.save(repo, {
        "function_name": "Saddle", "source_type": "expression", "symbols": "c", "code_string": "c*(x**2 - y**2)",
        "from": "-2", "to": "2", "points": "41", "y_from": "-2", "y_to": "2", "y_points": "41",
        "description": "A saddle: a function of x and y, drawn as a surface.", "values": {"c": 1.0},
    })


def _curve(name: str, title: str, y_label: str, dialog: str, setup) -> OperationDemo:
    _code, code_values, (start, stop) = next((c, v, r) for n, c, v, r, _d in FUNCTIONS if n == name)
    return OperationDemo(
        title, name, "", function_sql(name, code_values, GridRange(start, stop, _GRID)),
        {"x": "x", "y": "y"}, "x", y_label, dialog, chart_type="Scatter Plot",
        style={"linewidth": 1.6, "linestyle": "-", "marker": ""}, setup=setup)


def _calculus(model: str):
    def setup(dialog) -> None:
        dialog.model_combo.setCurrentText(model)
        dialog.destination_combo.setCurrentIndex(dialog.destination_combo.findData(dialog.DEST_NEW_AXIS))
    return setup


def _fit_stored_logistic(dialog) -> None:
    """The stored Logistic chosen, then Fit pressed: OK draws the table's parameters, the start until then."""
    if not dialog.select_model("Logistic"):
        raise RuntimeError("the stored Logistic is not in Fit Curve's catalogue")
    dialog.evaluate(dialog._after_fit, background=False, optimise=True)


def _family(repo: SqliteRepo) -> None:
    """One formula, three parameter sets: each series keeps its own values in its query."""
    figure_id = int(repo.create_figure_descriptor(
        name="Gaussian family - one formula, three parameter sets", nrows=1, ncols=1,
        options={"mpl_style": demo_style(), "layout_mode": "constrained"}))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot", title="Gaussian, three ways",
        x_label="x", y_label="A exp(-(x - mu)² / 2s²)", options={"title": "Gaussian, three ways", "legend": True}))
    for index, values in enumerate(({"A": 1.0, "mu": 0.0, "s": 1.0}, {"A": 0.6, "mu": -1.5, "s": 0.5},
                                    {"A": 0.8, "mu": 2.0, "s": 1.5})):
        label = ", ".join(f"{key} = {value:g}" for key, value in values.items())
        repo.create_series_descriptor(axis_id=axis_id, series_index=index, name=label, roles={"x": "x", "y": "y"},
                                      sql_query=function_sql("Gaussian", values, GridRange(-5.0, 5.0, _GRID)),
                                      style={"linewidth": 1.6, "linestyle": "-", "marker": ""})


def _saddle(repo: SqliteRepo) -> None:
    figure_id = int(repo.create_figure_descriptor(
        name="Saddle - a function of x and y", nrows=1, ncols=1,
        options={"mpl_style": demo_style(), "layout_mode": "constrained"}))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Surface Plot", title="c (x² - y²)", x_label="x",
        y_label="y", options={"projection": "3d", "cmap": "viridis", "title": "c (x² - y²)"}))
    grid = GridRange(-2.0, 2.0, 41)
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Saddle", roles={"x": "x", "y": "y", "z": "z"},
                                  sql_query=function_sql("Saddle", {"c": 1.0}, grid, grid), style={})


def _growth() -> pd.DataFrame:
    """Measurements of a logistic growth, with noise: what Fit Curve's stored Logistic is fitted to."""
    rng = np.random.default_rng(4)
    day = np.linspace(0.0, 10.0, 40)
    return pd.DataFrame({"day": day, "biomass": 80.0 / (1.0 + np.exp(-0.9 * (day - 4.5))) + rng.normal(0, 2.5, 40)})


#: Declared with the others in app.data.demos (the application lists them); found here by its builder.
FUNCTIONS_DEMO: DemoProject = next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_functions_demo"))


def build_functions_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the SymPy functions demo into *directory* and return its path."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.utils import config

    app = QApplication.instance() or QApplication([])
    real_exec = QMessageBox.exec
    QMessageBox.exec = lambda box, *a, **k: (  # type: ignore[method-assign]
        applogger.warning("Demo: message not shown: %s", box.text()) or QMessageBox.StandardButton.Yes
    )
    real_user_config = config.USER_CONFIG_PATH
    config.USER_CONFIG_PATH = Path(tempfile.mkdtemp(prefix="chartlibre-demo-")) / "user.json"
    try:
        path = Path(directory) / FUNCTIONS_DEMO.path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        _store(repo)
        repo.import_dataframe(_growth(), table_name="growth", normalize_columns=False)
        operations = (
            _curve("Damped", "Damped oscillation - its exact derivative", "a exp(-bx) cos(wx)",
                   "calculus_dialog:SeriesCalculusDialog", _calculus("Derivative (finite difference)")),
            _curve("Logistic", "Logistic growth - its exact integral", "L / (1 + exp(-k(x - x0)))",
                   "calculus_dialog:SeriesCalculusDialog", _calculus("Integral (cumulative)")),
            OperationDemo(
                "Noisy growth - the stored Logistic fitted", "Biomass by day", "growth",
                "SELECT day AS x, biomass AS y FROM growth", {"x": "x", "y": "y"}, "day", "biomass (g)",
                "fit_dialog:SeriesFitDialog", style={"marker": "o", "markersize": 4, "linestyle": ""},
                setup=_fit_stored_logistic),
        )
        for demo in operations:
            figure_id = _create_source_figure(repo, demo)
            try:
                _store_report(repo, figure_id, _run_operation(repo, figure_id, demo))
            except Exception:  # noqa: BLE001 - one figure failing leaves the others
                applogger.exception("Demo: %s failed.", demo.name)
        _family(repo)
        _saddle(repo)
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        QMessageBox.exec = real_exec  # type: ignore[method-assign]
        del app


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    print(build_functions_demo())
