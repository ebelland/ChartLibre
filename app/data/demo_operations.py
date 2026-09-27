"""Demo: the Series Operations, each run for real on a real dataset.

Every figure here starts as an ordinary chart of one demo table, and then
the operation's own dialog is driven exactly as a person would drive it -
built on the figure, left at its defaults (or given the one or two settings
the dataset needs), and Applied. What the file ends up holding is therefore
what the application itself produced: the result series drawn on the chart,
any result tables, and the HTML report in the chart's results pane.

Run ``python -m app.data.demo_operations`` to rebuild the file into
``demo/``. It needs a QApplication (the dialogs are widgets) and runs
offscreen; the dialogs' remembered settings are written to a throwaway
user.json, never to the real one.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.data.demo_project import DEMO_DIR, DEMO_PROJECTS, DEMO_STYLE, TABLE_SOURCES, DemoProject
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger

#: The figure options key the chart panel keeps its results pane in.
_VIEW_KEY = "view"


@dataclass(slots=True)
class OperationDemo:
    """One figure: a table drawn as a chart, and the operation run on it."""

    name: str
    title: str
    table: str
    sql: str
    roles: dict[str, str]
    x_label: str
    y_label: str
    dialog: str  # "module:Class" under app.series_operations
    chart_type: str = "Scatter Plot"
    style: dict[str, Any] = field(default_factory=lambda: {"marker": "o", "markersize": 2.5, "linestyle": ""})
    setup: Callable[[Any], None] | None = None


def _set_combo(dialog: Any, attribute: str, text: str) -> None:
    combo = getattr(dialog, attribute, None)
    if combo is not None:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)


OPERATIONS: tuple[OperationDemo, ...] = (
    OperationDemo(
        "01 · Smoothing - a moving average on a noisy price",
        "Daily close, smoothed", "stock_prices",
        "SELECT date AS x, close AS y FROM stock_prices WHERE stock = (SELECT MIN(stock) FROM stock_prices) ORDER BY date",
        {"x": "x", "y": "y"}, "date", "close", "smoothing_dialog:SeriesSmoothingDialog",
        style={"linewidth": 0.8, "marker": ""},
    ),
    OperationDemo(
        "02 · Spectral analysis - the tones inside a signal",
        "A noisy signal and its spectrum", "signal_time_domain",
        "SELECT t_s AS x, amplitude AS y FROM signal_time_domain ORDER BY t_s",
        {"x": "x", "y": "y"}, "time (s)", "amplitude", "spectral_dialog:SeriesSpectralDialog",
        style={"linewidth": 0.8, "marker": ""},
    ),
    OperationDemo(
        "03 · Filtering - a low-pass through the same signal",
        "Low-pass filtered signal", "signal_time_domain",
        "SELECT t_s AS x, amplitude AS y FROM signal_time_domain ORDER BY t_s",
        {"x": "x", "y": "y"}, "time (s)", "amplitude", "filter_dialog:SeriesFilterDialog",
        style={"linewidth": 0.8, "marker": ""},
    ),
    OperationDemo(
        "04 · Baseline correction - a drifting background removed",
        "Spectrum with a wandering baseline", "baseline_spectrum",
        "SELECT x, intensity AS y FROM baseline_spectrum ORDER BY x",
        {"x": "x", "y": "y"}, "x", "intensity", "baseline_dialog:SeriesBaselineDialog",
        style={"linewidth": 1.0, "marker": ""},
    ),
    OperationDemo(
        "05 · Peaks - the maxima of the same spectrum",
        "Peaks by prominence", "baseline_spectrum",
        "SELECT x, intensity AS y FROM baseline_spectrum ORDER BY x",
        {"x": "x", "y": "y"}, "x", "intensity", "peaks_dialog:SeriesPeaksDialog",
        style={"linewidth": 1.0, "marker": ""},
    ),
    OperationDemo(
        "06 · Roots - where a force curve changes sign",
        "DLVO force: repulsive, then attractive", "dlvo_curve",
        "SELECT separation_nm AS x, F_total_nN AS y FROM dlvo_curve ORDER BY separation_nm",
        {"x": "x", "y": "y"}, "separation (nm)", "force (nN)", "roots_dialog:SeriesRootsDialog",
        style={"linewidth": 1.2, "marker": ""},
    ),
    OperationDemo(
        "07 · Calculus - derivative and integral of the force",
        "Force, its derivative and its integral", "dlvo_curve",
        "SELECT separation_nm AS x, F_total_nN AS y FROM dlvo_curve ORDER BY separation_nm",
        {"x": "x", "y": "y"}, "separation (nm)", "force (nN)", "calculus_dialog:SeriesCalculusDialog",
        style={"linewidth": 1.2, "marker": ""},
    ),
    OperationDemo(
        "08 · Fit - a model through sparse calibration points",
        "Nine calibration points and a fitted curve", "sparse_calibration",
        "SELECT concentration AS x, response AS y FROM sparse_calibration ORDER BY concentration",
        {"x": "x", "y": "y"}, "concentration", "response", "fit_dialog:SeriesFitDialog",
        style={"marker": "o", "markersize": 6, "linestyle": ""},
    ),
    OperationDemo(
        "09 · Interpolation - a smooth curve through the same points",
        "Interpolated calibration curve", "sparse_calibration",
        "SELECT concentration AS x, response AS y FROM sparse_calibration ORDER BY concentration",
        {"x": "x", "y": "y"}, "concentration", "response", "interpolate_dialog:SeriesInterpolateDialog",
        style={"marker": "o", "markersize": 6, "linestyle": ""},
    ),
    OperationDemo(
        "10 · Statistics - four machines measuring one standard",
        "Machine A against machine C", "machine_measurements",
        "SELECT run AS x, value AS y FROM machine_measurements WHERE machine = 'Machine A' ORDER BY run",
        {"x": "x", "y": "y"}, "run", "reading (mm)", "statistics_dialog:SeriesStatisticsDialog",
    ),
    OperationDemo(
        "11 · Outliers - the salary nobody else earns",
        "Salary by age", "employee_compensation",
        "SELECT age AS x, salary_eur AS y FROM employee_compensation ORDER BY age",
        {"x": "x", "y": "y"}, "age", "salary (EUR)", "outlier_dialog:SeriesOutlierDialog",
        style={"marker": "o", "markersize": 4, "linestyle": ""},
    ),
    OperationDemo(
        "12 · Clustering - four thousand drivers in groups",
        "Distance against speeding", "driver_behaviour",
        "SELECT distance_feature AS x, speeding_feature AS y FROM driver_behaviour",
        {"x": "x", "y": "y"}, "distance", "speeding (%)", "cluster_dialog:SeriesClusterDialog",
        style={"marker": "o", "markersize": 2, "linestyle": ""},
    ),
    OperationDemo(
        "13 · Control chart - is the measuring process stable?",
        "Machine A, run by run", "machine_measurements",
        "SELECT run AS x, value AS y FROM machine_measurements WHERE machine = 'Machine A' ORDER BY run",
        {"x": "x", "y": "y"}, "run", "reading (mm)", "control_chart_dialog:SeriesControlChartDialog",
        style={"marker": "o", "markersize": 4, "linewidth": 0.8},
    ),
    OperationDemo(
        "14 · Regression - a robust trend through CO2",
        "Monthly CO2 at Mauna Loa", "co2_mauna_loa",
        "SELECT CAST(strftime('%Y', date) AS REAL) + (CAST(strftime('%m', date) AS REAL) - 0.5) / 12.0 AS x, "
        "co2_ppm AS y FROM co2_mauna_loa ORDER BY date",
        {"x": "x", "y": "y"}, "year", "CO2 (ppm)", "regression_dialog:SeriesRegressionDialog",
        style={"marker": "o", "markersize": 1.5, "linestyle": ""},
    ),
    OperationDemo(
        "15 · Transform - reshaping a skewed salary distribution",
        "Salaries, reshaped", "employee_compensation",
        "SELECT employee_id AS x, salary_eur AS y FROM employee_compensation ORDER BY employee_id",
        {"x": "x", "y": "y"}, "employee", "salary (EUR)", "transform_dialog:SeriesTransformDialog",
        style={"marker": "o", "markersize": 4, "linestyle": ""},
    ),
)

#: This file's entry in the demo set.
OPERATIONS_DEMO: DemoProject = next(
    demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_operations_demo")
)


def _dialog_class(path: str) -> type:
    module_name, class_name = path.split(":")
    module = importlib.import_module(f"app.series_operations.{module_name}")
    return getattr(module, class_name)


def _create_source_figure(repo: SqliteRepo, demo: OperationDemo) -> int:
    figure_id = int(
        repo.create_figure_descriptor(
            name=demo.name, nrows=1, ncols=1,
            options={"mpl_style": DEMO_STYLE, "layout_mode": "constrained"},
        )
    )
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type=demo.chart_type,
            title=demo.title, x_label=demo.x_label, y_label=demo.y_label,
            options={"title": demo.title},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name=demo.title, sql_query=demo.sql,
        roles=demo.roles, style=dict(demo.style),
    )
    return figure_id


def _run_operation(repo: SqliteRepo, figure_id: int, demo: OperationDemo) -> str:
    """Drive the operation's dialog to Apply, and return the report it published."""
    reports: list[str] = []
    dialog = _dialog_class(demo.dialog)(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.results_published.connect(reports.append)
        # What a person does first: tick the series to run on.
        selector = getattr(dialog, "series_selector", None)
        if selector is not None and hasattr(selector, "select_all_series"):
            selector.select_all_series()
        if demo.setup is not None:
            demo.setup(dialog)
        # OK, not Apply: it is what keeps the change and publishes the report
        # for every operation, including the ones whose Apply only previews.
        dialog.ok()
    finally:
        dialog.close()
    return "<hr>".join(reports)


def _store_report(repo: SqliteRepo, figure_id: int, html: str) -> None:
    if not html.strip():
        return
    options = repo.get_figure_options(figure_id)
    view = dict(options.get(_VIEW_KEY) or {})
    view.update({"resize_mode": "FIT", "notes_html": html, "notes_split": [520, 300]})
    options[_VIEW_KEY] = view
    repo.set_figure_options(figure_id, options)


def build_operations_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the Series Operations demo into *directory* and return its path."""
    from PySide6.QtWidgets import QApplication

    import app.utils.config as config

    from PySide6.QtWidgets import QMessageBox

    app = QApplication.instance() or QApplication([])
    # No one is there to click: a message box is logged instead of waited on.
    real_exec = QMessageBox.exec
    QMessageBox.exec = lambda box, *a, **k: (  # type: ignore[method-assign]
        applogger.warning("Demo: message not shown: %s", box.text()) or QMessageBox.StandardButton.Yes
    )
    real_user_config = config.USER_CONFIG_PATH
    scratch = Path(tempfile.mkdtemp(prefix="dhub-demo-")) / "user.json"
    config.USER_CONFIG_PATH = scratch  # dialogs remember their settings here, not in the real file
    try:
        path = Path(directory) / OPERATIONS_DEMO.path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        for table in sorted({demo.table for demo in OPERATIONS}):
            repo.import_dataframe(TABLE_SOURCES[table](), table_name=table, normalize_columns=False)
        for demo in OPERATIONS:
            figure_id = _create_source_figure(repo, demo)
            try:
                html = _run_operation(repo, figure_id, demo)
            except Exception:
                applogger.exception("Demo: %s failed.", demo.name)
                continue
            _store_report(repo, figure_id, html)
            applogger.info("Demo: %s - report %d characters.", demo.name, len(html))
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        QMessageBox.exec = real_exec  # type: ignore[method-assign]
        del app


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    print(build_operations_demo())
