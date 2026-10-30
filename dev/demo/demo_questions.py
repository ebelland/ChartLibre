"""Demo: the quick answers, each asked of real data (D-01).

The demo set was organised by chart type and by operation; someone who
starts from a question - "is it different?", "what matters?", "has it
changed?" - had no project that began there. This one does: each quick
answer asked of a real dataset, by the real dialog, as a person would ask it,
so the file holds what the application answered - the table, the chart and
the answer in words in the chart's notes.

* Are there differences? - the Palmer penguins' body mass by species.
* What has the most impact? - the same body mass against everything else
  measured about each penguin.
* Has it changed over time? - NASA's annual global temperature anomaly.

Run ``python -m dev.demo.demo_questions`` to rebuild the file into
``demo/``. It runs offscreen; the dialogs' remembered settings go to a
throwaway user.json, never to the real one.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from app.data.demos import DEMO_DIR, DEMO_PROJECTS, DemoProject
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger

#: The table Has it changed over time? is asked of: a figure of its years and anomalies.
CHANGE_TABLE = "global_temperature"

#: (dialog module:class, table, roles) - one per quick answer, in the guided analysis's order.
QUESTIONS: tuple[tuple[str, str, dict[str, list[str]]], ...] = (
    ("app.table_operations.differences_dialog:DifferencesDialog", "penguins",
     {"response": ["body_mass_g"], "group": ["species"]}),
    ("app.table_operations.impact_dialog:ImpactDialog", "penguins",
     {"response": ["body_mass_g"],
      "factor": ["species", "sex", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm"]}),
)


def _demo() -> DemoProject:
    return next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_questions_demo"))


def _ask(app, repo: SqliteRepo, dialog_path: str, table: str, roles: dict[str, list[str]],
         params: dict[str, Any] | None = None, kinds: dict[str, str] | None = None) -> None:
    """Open the analysis on *table*, cast the columns (typed by *kinds*), set *params*, press OK and wait."""
    import importlib

    module_name, class_name = dialog_path.split(":")
    dialog = getattr(importlib.import_module(module_name), class_name)(repo=repo, table=table)
    try:
        roles = dict(roles)
        effects = roles.pop("effects", None)  # Fit Model's: its X are effects, not a role
        dialog.roles_widget.set_casting(roles, kinds)
        if effects:
            dialog.cast_explanatory(list(effects))
        if params:
            dialog._parameter_form.set_values(params)
        problems = dialog.problems()
        if problems:
            raise RuntimeError("; ".join(problems))
        dialog.ok()
        deadline = time.monotonic() + 120
        while dialog.job_running and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if dialog.job_running or not dialog.created_figure_ids:
            raise RuntimeError(f"{class_name} did not finish")
        applogger.info("Demo: %s on %s - figure %s.", class_name, table, dialog.created_figure_ids)
    finally:
        dialog.close()


def _ask_figure(app, repo: SqliteRepo, dialog_path: str, table: str, *, x: str, y: str) -> None:
    """Open a data operation on a chart of *table*'s *x* and *y*, press OK and wait."""
    import importlib

    module_name, class_name = dialog_path.split(":")
    figure_id = int(repo.create_figure_descriptor(name=table))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title=table, x_label=x, y_label=y, options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=y,
                                  sql_query=f'SELECT "{x}" AS x, "{y}" AS y FROM "{table}"',
                                  roles={"x": "x", "y": "y"}, style={})
    dialog = getattr(importlib.import_module(module_name), class_name)(repo=repo, figure_id=figure_id)
    try:
        dialog.ok()
        deadline = time.monotonic() + 120
        while dialog.job_running and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if dialog.job_running or not dialog.created_figure_ids:
            raise RuntimeError(f"{class_name} did not finish")
        applogger.info("Demo: %s on %s - figure %s.", class_name, table, dialog.created_figure_ids)
    finally:
        dialog.close()


def build_questions_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the quick-answers demo into *directory* and return its path."""
    from PySide6.QtWidgets import QApplication

    import app.utils.config as config
    from dev.demo.build_demos import TABLE_SOURCES

    app = QApplication.instance() or QApplication([])
    real_user_config = config.USER_CONFIG_PATH
    config.USER_CONFIG_PATH = Path(tempfile.mkdtemp(prefix="dhub-demo-")) / "user.json"
    try:
        path = Path(directory) / _demo().path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        for table in dict.fromkeys([*(table for _dialog, table, _roles in QUESTIONS), CHANGE_TABLE]):
            repo.import_dataframe(TABLE_SOURCES[table](), table_name=table, normalize_columns=False)
        for dialog_path, table, roles in QUESTIONS:
            _ask(app, repo, dialog_path, table, roles)
        _ask_figure(app, repo, "app.data_operations.change_dialog:ChangeDialog", CHANGE_TABLE,
                    x="year", y="anomaly_c")
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        del app


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dev.demo.build_demos import describe_demo

    built = build_questions_demo()
    describe_demo(_demo(), built)
    print(built)
