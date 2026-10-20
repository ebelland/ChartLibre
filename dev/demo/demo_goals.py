"""Demos built around a goal, each asked as a person would (D-01).

The quick answers' demo (demo_questions) starts from three questions; these
start from the guided analysis's goals, with the data a person
would bring to them, and go the way Help me choose leads: the charts the
columns allow first, then the analyses that answer the questions.

* Relate - how the penguins' four measures move together: a scatter plot,
  Multivariate, Fit Y by X, Principal Components.
* Classify - telling the penguin species apart: a box plot, Discriminant,
  Partition, and Fit Y by X on species by island.
* Describe and compare - salaries across departments: a histogram,
  Distribution, Are there differences?, What has the most impact?
* Predict - which card transactions are fraud: What has the most impact?,
  Partition and a logistic Fit Model.
* Signal - the Keeling curve's season and trend, forecast five years by an
  ARIMA chosen by AIC; the sunspots' eleven-year cycle in their ACF.
* Anomalies - the card transactions that stand apart: their amounts'
  distribution, Explore Outliers on amount against distance from home, and
  whether distance tells fraud (Fit Y by X).
* Process - four gauges measuring one standard: Control Chart with the
  specification's capability for machines A and D, and Are there differences?
* Groups - the penguins clustered on their measures without their species,
  Principal Components coloured by species, Factor Analysis.

Each is a DemoProject whose builder is one of the functions below. Run
``python -m dev.demo.demo_goals`` to rebuild them into ``demo/``. They run
offscreen; the dialogs' remembered settings go to a throwaway user.json.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from app.data.demos import DEMO_DIR, DEMO_PROJECTS, DemoProject
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from dev.demo.demo_operations import OperationDemo, _create_source_figure, _run_operation, _store_report
from dev.demo.demo_questions import _ask

PENGUIN_MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TRANSACTION_FACTORS = ["transaction_amount", "transaction_frequency_7d", "account_age_days", "distance_from_home_km",
                       "time_since_last_transaction_hours", "failed_login_attempts_24h", "channel", "country"]

#: A step: ("chart", table, columns picked in Help me choose, which of its charts),
#: ("ask", "module:Dialog", table, role -> columns[, parameters[, column types]]) for a table
#: analysis, or ("operate", OperationDemo) for a data operation run on a chart of its own.
Step = tuple[Any, ...]

GOAL_DEMOS: dict[str, tuple[Step, ...]] = {
    "build_relate_demo": (
        ("chart", "penguins", ["flipper_length_mm", "body_mass_g"], 0),
        ("ask", "app.table_operations.multivariate_dialog:MultivariateDialog", "penguins",
         {"response": PENGUIN_MEASURES}),
        ("ask", "app.table_operations.fit_y_by_x_dialog:FitYByXDialog", "penguins",
         {"response": ["body_mass_g"], "factor": ["flipper_length_mm"]}),
        ("ask", "app.table_operations.principal_components_dialog:PrincipalComponentsDialog", "penguins",
         {"response": PENGUIN_MEASURES}),
    ),
    "build_classify_demo": (
        ("chart", "penguins", ["bill_depth_mm", "species"], 0),
        ("ask", "app.table_operations.discriminant_dialog:DiscriminantDialog", "penguins",
         {"response": ["species"], "covariate": PENGUIN_MEASURES}),
        ("ask", "app.table_operations.partition_dialog:PartitionDialog", "penguins",
         {"response": ["species"], "factor": [*PENGUIN_MEASURES, "island", "sex"]}),
        ("ask", "app.table_operations.fit_y_by_x_dialog:FitYByXDialog", "penguins",
         {"response": ["species"], "factor": ["island"]}),
    ),
    "build_describe_compare_demo": (
        ("chart", "employee_compensation", ["salary_eur"], 0),
        ("ask", "app.table_operations.distribution_dialog:DistributionDialog", "employee_compensation",
         {"response": ["salary_eur"]}),
        ("ask", "app.table_operations.differences_dialog:DifferencesDialog", "employee_compensation",
         {"response": ["salary_eur"], "group": ["department"]}),
        ("ask", "app.table_operations.impact_dialog:ImpactDialog", "employee_compensation",
         {"response": ["salary_eur"], "factor": ["department", "age", "bonus_eur"]}),
    ),
    "build_predict_demo": (
        ("chart", "transactions", ["distance_from_home_km", "fraud"], 0),
        ("ask", "app.table_operations.impact_dialog:ImpactDialog", "transactions",
         {"response": ["fraud"], "factor": TRANSACTION_FACTORS}, None, {"fraud": "nominal"}),
        ("ask", "app.table_operations.partition_dialog:PartitionDialog", "transactions",
         {"response": ["fraud"], "factor": TRANSACTION_FACTORS}, None, {"fraud": "nominal"}),
        ("ask", "app.table_operations.fit_model_dialog:FitModelDialog", "transactions",
         {"response": ["fraud"], "effects": ["distance_from_home_km", "failed_login_attempts_24h",
                                             "transaction_amount"]}, None, {"fraud": "nominal"}),
    ),
    "build_signal_demo": (
        ("chart", "co2_mauna_loa", ["date", "co2_ppm"], 0),
        ("ask", "app.table_operations.time_series_dialog:TimeSeriesDialog", "co2_mauna_loa",
         {"response": ["co2_ppm"], "time": ["date"]}, {"model": "auto", "horizon": 60, "lags": 48}),
        ("ask", "app.table_operations.time_series_dialog:TimeSeriesDialog", "sunspots_monthly",
         {"response": ["sunspot_number"], "time": ["date"]}, {"model": "none", "season": 1, "lags": 300}),
    ),
    "build_anomalies_demo": (
        ("chart", "transactions", ["transaction_amount"], 0),
        ("ask", "app.table_operations.distribution_dialog:DistributionDialog", "transactions",
         {"response": ["transaction_amount"]}),
        ("operate", OperationDemo(
            "Explore Outliers - the transactions that stand apart", "Amount against distance from home",
            "transactions",
            "SELECT distance_from_home_km AS x, transaction_amount AS y FROM transactions ORDER BY x",
            {"x": "x", "y": "y"}, "distance from home (km)", "amount", "outlier_dialog:SeriesOutlierDialog",
            style={"marker": "o", "markersize": 3, "linestyle": ""},
            # Coloured, not hidden: the next question is whether they are the frauds.
            setup=lambda dialog: dialog._parameter_form_spec.set_values({"action": "colour"}))),
        ("ask", "app.table_operations.fit_y_by_x_dialog:FitYByXDialog", "transactions",
         {"response": ["fraud"], "factor": ["distance_from_home_km"]}, None, {"fraud": "nominal"}),
    ),
    "build_process_demo": (
        ("chart", "machine_measurements", ["value", "machine"], 0),
        ("operate", OperationDemo(
            "Control Chart - machine A against its specification", "Machine A, run by run",
            "machine_measurements",
            "SELECT run AS x, value AS y FROM machine_measurements WHERE machine = 'Machine A' ORDER BY run",
            {"x": "x", "y": "y"}, "run", "reading (mm)", "control_chart_dialog:SeriesControlChartDialog",
            style={"marker": "o", "markersize": 4, "linewidth": 0.8}, setup=lambda dialog: _spec(dialog))),
        ("operate", OperationDemo(
            "Control Chart - machine D, the one that wanders", "Machine D, run by run",
            "machine_measurements",
            "SELECT run AS x, value AS y FROM machine_measurements WHERE machine = 'Machine D' ORDER BY run",
            {"x": "x", "y": "y"}, "run", "reading (mm)", "control_chart_dialog:SeriesControlChartDialog",
            style={"marker": "o", "markersize": 4, "linewidth": 0.8}, setup=lambda dialog: _spec(dialog))),
        ("ask", "app.table_operations.differences_dialog:DifferencesDialog", "machine_measurements",
         {"response": ["value"], "group": ["machine"]}),
    ),
    "build_groups_demo": (
        ("chart", "penguins", ["bill_length_mm", "bill_depth_mm"], 0),
        ("ask", "app.table_operations.clustering_dialog:ClusteringDialog", "penguins",
         {"response": PENGUIN_MEASURES}),
        ("ask", "app.table_operations.principal_components_dialog:PrincipalComponentsDialog", "penguins",
         {"response": PENGUIN_MEASURES, "by": ["species"]}),
        ("ask", "app.table_operations.factor_analysis_dialog:FactorAnalysisDialog", "penguins",
         {"response": PENGUIN_MEASURES}, {"factors": 2}),
    ),
}


#: The gauge standard is 25 mm; the specification allows 0.02 mm either way.
SPEC_LIMITS = {"lsl": "24.98", "usl": "25.02"}


def _spec(dialog: Any) -> None:
    """The specification limits, so the report gives the capability (Cp, Cpk...)."""
    dialog._parameter_form_spec.set_values(SPEC_LIMITS)


def _demo(builder: str) -> DemoProject:
    return next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(f":{builder}"))


def _chart(repo: SqliteRepo, table: str, columns: list[str], index: int) -> None:
    """The chart Help me choose offers *index*-th for *columns*, made as Create chart makes it."""
    from app.plugin_api.table_operations import guided_plan, table_profile

    frame = repo.table_frame(table).drop(columns=["Hide", "Selected"], errors="ignore")
    declared = {str(row[1]): str(row[2] or "") for row in repo.table_info(table)}
    idea = guided_plan.plan(table_profile.profile(table, frame, declared), columns).charts[index]
    figure_id = guided_plan.make_chart(repo, table, idea)
    applogger.info("Demo: %s of %s - figure %d.", idea.chart_type, table, figure_id)


def _build(builder: str, directory: Path) -> Path:
    from PySide6.QtWidgets import QApplication

    import app.utils.config as config
    from dev.demo.build_demos import TABLE_SOURCES

    from PySide6.QtWidgets import QMessageBox

    app = QApplication.instance() or QApplication([])
    # No one is there to click: a message box is logged instead of waited on.
    real_exec = QMessageBox.exec
    QMessageBox.exec = lambda box, *a, **k: (  # type: ignore[method-assign]
        applogger.warning("Demo: message not shown: %s", box.text()) or QMessageBox.StandardButton.Yes
    )
    real_user_config = config.USER_CONFIG_PATH
    config.USER_CONFIG_PATH = Path(tempfile.mkdtemp(prefix="dhub-demo-")) / "user.json"
    steps = GOAL_DEMOS[builder]
    try:
        path = Path(directory) / _demo(builder).path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        tables = [step[1] if step[0] == "chart" else step[1].table if step[0] == "operate" else step[2]
                  for step in steps]
        for table in dict.fromkeys(tables):
            repo.import_dataframe(TABLE_SOURCES[table](), table_name=table, normalize_columns=False)
        for step in steps:
            if step[0] == "chart":
                _chart(repo, *step[1:])
            elif step[0] == "operate":
                figure_id = _create_source_figure(repo, step[1])
                _store_report(repo, figure_id, _run_operation(repo, figure_id, step[1]))
            else:
                _ask(app, repo, *step[1:])
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        QMessageBox.exec = real_exec  # type: ignore[method-assign]
        del app


def build_relate_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_relate_demo", directory)


def build_classify_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_classify_demo", directory)


def build_describe_compare_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_describe_compare_demo", directory)


def build_predict_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_predict_demo", directory)


def build_signal_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_signal_demo", directory)


def build_anomalies_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_anomalies_demo", directory)


def build_process_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_process_demo", directory)


def build_groups_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_groups_demo", directory)


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dev.demo.build_demos import describe_demo

    for name in GOAL_DEMOS:
        built = globals()[name]()
        describe_demo(_demo(name), built)
        print(built)
