"""Demo: a designed experiment and its analysis - Design of Experiments, then Fit Model.

A Box-Behnken design for a reaction (temperature, time, catalyst), written
the way the Design of Experiments dialog writes it - the design kept with the
table - with the responses a laboratory would have measured filled in: the
yield, curved, with an optimum inside the region; the purity, falling with
temperature; and a pass/fail grade.

Fit Model is then driven as a person would drive it: the yield as a response
surface (Standard Least Squares, Effect Screening), and the grade by Nominal
Logistic. What the file holds is what the application produced: the result
tables, the figures and the reports in their notes.

Run ``python -m dev.demo.demo_fit_model`` to rebuild the file into
``demo/``. It runs offscreen; the dialog's remembered settings go to a
throwaway user.json, never to the real one.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np

from app.data.demos import DEMO_DIR, DEMO_PROJECTS, DemoProject
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger

TABLE = "Reaction_DOE"


def _demo() -> DemoProject:
    return next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_fit_model_demo"))


def _experiment():
    """The design, as the DOE dialog makes it, and its responses as measured."""
    from app.dialogs.doe_dialog import DOEExperimentDialog, DOERequest, Factor

    request = DOERequest(
        model="box_behnken", table_name=TABLE,
        factors=(Factor("Temperature", 150, 190), Factor("Time", 30, 90), Factor("Catalyst", 0.5, 1.5)),
        responses=("Yield", "Purity", "Grade"), levels=3, center_points=3, samples=20,
        replicates=2, randomize=True, seed=11, lhs_algorithm="random",
    )
    frame = DOEExperimentDialog.generate_matrix(request)
    rng = np.random.default_rng(2024)
    # Coded -1..+1, the way the model sees the factors.
    t = (frame["Temperature"] - 170) / 20
    m = (frame["Time"] - 60) / 30
    c = (frame["Catalyst"] - 1.0) / 0.5
    # Optimum a little hotter than the centre, longer, with more catalyst;
    # temperature and time interact (a hot short run is as good as a cool
    # long one).
    frame["Yield"] = (
        78 + 6 * t + 4 * m + 2.5 * c - 3 * t * m - 5 * t**2 - 3 * m**2 - 1.5 * c**2
        + rng.normal(0, 0.8, len(frame))
    ).round(1)
    frame["Purity"] = (97 - 2.2 * t - 0.6 * m + 0.4 * c - 0.9 * t**2 + rng.normal(0, 0.35, len(frame))).round(2)
    # A grade the laboratory records: pass when the yield is good and the
    # product clean enough - a categorical answer for the logistic fit.
    score = 0.35 * (frame["Yield"] - 75) + 1.5 * (frame["Purity"] - 96) + rng.normal(0, 1.2, len(frame))
    frame["Grade"] = np.where(score > 0, "Pass", "Fail")
    return frame, DOEExperimentDialog.design_record(request)


def _wait(app, dialog) -> None:
    deadline = time.monotonic() + 120
    while dialog._task is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    if dialog._task is not None:
        raise RuntimeError("Fit Model did not finish")


def _run_fit(app, repo: SqliteRepo, *, responses: list[str], personality: str, emphasis: str,
             terms: str) -> None:
    """Open Fit Model on the table, cast, choose, and press OK."""
    from PySide6.QtCore import Qt

    from app.table_operations.fit_model_dialog import FitModelDialog

    dialog = FitModelDialog(repo=repo, table=TABLE)
    try:
        dialog.roles_widget.set_casting({"response": responses})
        dialog._personality.setCurrentIndex(dialog._personality.findData(personality))
        dialog._emphasis.setCurrentIndex(dialog._emphasis.findData(emphasis))
        columns = dialog.roles_widget._column_list
        for row in range(columns.count()):
            item = columns.item(row)
            item.setSelected(item.data(Qt.ItemDataRole.UserRole) in ("Temperature", "Time", "Catalyst"))
        dialog._apply_macro(terms, edited=True)
        problems = dialog.problems()
        if problems:
            raise RuntimeError("; ".join(problems))
        dialog.ok()
        _wait(app, dialog)
        applogger.info("Demo: Fit Model (%s) on %s - figures %s.", personality, responses, dialog.created_figure_ids)
    finally:
        dialog.close()


def build_fit_model_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the designed-experiment demo into *directory* and return its path."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    import app.utils.config as config

    app = QApplication.instance() or QApplication([])
    real_exec = QMessageBox.exec
    QMessageBox.exec = lambda box, *a, **k: (  # type: ignore[method-assign]
        applogger.warning("Demo: message not shown: %s", box.text()) or QMessageBox.StandardButton.Yes
    )
    real_user_config = config.USER_CONFIG_PATH
    config.USER_CONFIG_PATH = Path(tempfile.mkdtemp(prefix="dhub-demo-")) / "user.json"
    try:
        path = Path(directory) / _demo().path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        frame, design = _experiment()
        repo.import_dataframe(frame, table_name=TABLE, normalize_columns=False)
        repo.set_table_info(TABLE, "doe", design)
        _run_fit(app, repo, responses=["Yield", "Purity"], personality="standard",
                 emphasis="screening", terms="surface")
        _run_fit(app, repo, responses=["Grade"], personality="nominal", emphasis="screening", terms="main")
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        QMessageBox.exec = real_exec  # type: ignore[method-assign]
        del app


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dev.demo.build_demos import describe_demo

    built = build_fit_model_demo()
    describe_demo(_demo(), built)
    print(built)
