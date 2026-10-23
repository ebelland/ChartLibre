"""Demo: a measuring system and a skewed process, judged the way a quality engineer would (AN-06).

Simulated, with the variation built in known, so the answers can be checked:

* Gage R&R - a gauge on 10 parts, 3 operators, 3 repeats each; the parts vary
  much more than the gauge, one operator reads a little high. Drawn on the
  Multi-Vari chart.
* A surface roughness Ra (um), 120 parts in production order, lognormal as
  roughness is: Is it in control? (yes), Is it normal? (no - skewed to the
  right), its Distribution with the best fit by AIC (lognormal), and the
  Control Chart with its capability judged by the percentile method on a
  lognormal, against USL 3.2 um.
* A vortex sampled on a grid - Vector Field: its divergence (none) and curl.

Run ``python -m dev.demo.demo_quality`` to rebuild the file into ``demo/``.
It runs offscreen; the dialogs' remembered settings go to a throwaway
user.json, never to the real one.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.data.demos import DEMO_DIR, DEMO_PROJECTS, DemoProject
from app.data.sqlite_repo import SqliteRepo
from dev.demo.demo_questions import _ask


def _demo() -> DemoProject:
    return next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_quality_demo"))


def gauge_study() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    parts = rng.normal(0.0, 0.05, 10)
    operators = {"Anna": 0.0, "Bruno": 0.012, "Carla": -0.004}
    rows = [(f"P{p + 1:02d}", name, round(20.0 + parts[p] + bias + rng.normal(0.0, 0.008), 4))
            for p in range(10) for name, bias in operators.items() for _repeat in range(3)]
    return pd.DataFrame(rows, columns=["part", "operator", "diameter_mm"])


def roughness() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    values = rng.lognormal(mean=np.log(1.2), sigma=0.35, size=120)
    days = pd.date_range("2026-03-02", periods=120, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"day": days, "ra_um": np.round(values, 3)})


def vortex() -> pd.DataFrame:
    g = np.linspace(-2.0, 2.0, 17)
    xs, ys = np.meshgrid(g, g, indexing="ij")
    r2 = xs ** 2 + ys ** 2 + 0.3
    return pd.DataFrame({"x": xs.ravel(), "y": ys.ravel(), "u": (-ys / r2).ravel(), "v": (xs / r2).ravel()})


def _control_chart_with_capability(app, repo: SqliteRepo) -> None:
    """Control Chart on the roughness: an individuals chart, capability on a lognormal against USL 3.2."""
    from app.data_operations.control_chart_dialog import SeriesControlChartDialog

    figure = repo.create_figure_descriptor(name="Roughness Ra - control chart")
    axis = repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Time Series", title="Ra",
                                       x_label="day", y_label="Ra (um)", options={})
    repo.create_series_descriptor(axis_id=axis, series_index=0, name="Ra", sql_query='SELECT day AS x, ra_um AS y '
                                  'FROM "roughness"', roles={"x": "x", "y": "y"}, style={"marker": "o"})
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure, parent=None)
    try:
        dialog.series_selector.select_all_series()
        dialog._parameter_form_spec.set_values({"usl": "3.2", "distribution": "lognormal"})
        done: list[bool] = []
        dialog.apply(then=done.append)
        deadline = time.monotonic() + 120
        while not done and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        if not done or not done[0]:
            raise RuntimeError("the control chart was not applied")
    finally:
        dialog.close()


def build_quality_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the quality demo into *directory* and return its path."""
    from PySide6.QtWidgets import QApplication

    import app.utils.config as config

    app = QApplication.instance() or QApplication([])
    real_user_config = config.USER_CONFIG_PATH
    config.USER_CONFIG_PATH = Path(tempfile.mkdtemp(prefix="dhub-demo-")) / "user.json"
    try:
        path = Path(directory) / _demo().path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        for table, frame in (("gauge_study", gauge_study()), ("roughness", roughness()), ("vortex", vortex())):
            repo.import_dataframe(frame, table_name=table, normalize_columns=False)
        _ask(app, repo, "app.table_operations.gage_dialog:GageDialog", "gauge_study",
             {"response": ["diameter_mm"], "part": ["part"], "operator": ["operator"]},
             params={"tolerance": "0.4"}, kinds={"part": "nominal", "operator": "nominal"})
        _ask(app, repo, "app.table_operations.in_control_dialog:InControlDialog", "roughness",
             {"response": ["ra_um"], "time": ["day"]})
        _ask(app, repo, "app.table_operations.normality_dialog:NormalityDialog", "roughness",
             {"response": ["ra_um"]})
        _ask(app, repo, "app.table_operations.distribution_dialog:DistributionDialog", "roughness",
             {"response": ["ra_um"]}, params={"distribution": "best"})
        _control_chart_with_capability(app, repo)
        _ask(app, repo, "app.table_operations.vector_field_dialog:VectorFieldDialog", "vortex",
             {"x": ["x"], "y": ["y"], "u": ["u"], "v": ["v"]})
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        del app


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dev.demo.build_demos import describe_demo

    built = build_quality_demo()
    describe_demo(_demo(), built)
    print(built)
