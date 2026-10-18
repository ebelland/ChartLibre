"""Demos built around a goal, each asked as a person would (D-01).

The quick answers' demo (demo_questions) starts from three questions; these
start from three of the guided analysis's goals, with the data a person
would bring to them, and go the way Help me choose leads: the charts the
columns allow first, then the analyses that answer the questions.

* Relate - how the penguins' four measures move together: a scatter plot,
  Multivariate, Fit Y by X, Principal Components.
* Classify - telling the penguin species apart: a box plot, Discriminant,
  Partition, and Fit Y by X on species by island.
* Describe and compare - salaries across departments: a histogram,
  Distribution, Are there differences?, What has the most impact?

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
from dev.demo.demo_questions import _ask

PENGUIN_MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

#: A step: ("chart", table, columns picked in Help me choose, which of its charts)
#: or ("ask", "module:Dialog", table, role -> columns).
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
}


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

    app = QApplication.instance() or QApplication([])
    real_user_config = config.USER_CONFIG_PATH
    config.USER_CONFIG_PATH = Path(tempfile.mkdtemp(prefix="dhub-demo-")) / "user.json"
    steps = GOAL_DEMOS[builder]
    try:
        path = Path(directory) / _demo(builder).path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        for table in dict.fromkeys(step[1] if step[0] == "chart" else step[2] for step in steps):
            repo.import_dataframe(TABLE_SOURCES[table](), table_name=table, normalize_columns=False)
        for step in steps:
            if step[0] == "chart":
                _chart(repo, *step[1:])
            else:
                _ask(app, repo, *step[1:])
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        del app


def build_relate_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_relate_demo", directory)


def build_classify_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_classify_demo", directory)


def build_describe_compare_demo(directory: Path = DEMO_DIR) -> Path:
    return _build("build_describe_compare_demo", directory)


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dev.demo.build_demos import describe_demo

    for name in GOAL_DEMOS:
        built = globals()[name]()
        describe_demo(_demo(name), built)
        print(built)
