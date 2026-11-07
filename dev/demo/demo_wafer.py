"""The Wafer Map demo: four wafers, each measured or inspected, each run through Wafer Map for real.

- A 300 mm wafer's film thickness at 49 sites (a centre and three rings - the usual pattern): a bowl and a
  tilt, binned against the limits - the contour, the die means, the die bins and the yield.
- A 150 mm wafer with its flat, its sheet resistance read with the flat on the right: rotated -90° so the flat
  is at the bottom where the map draws it.
- A 200 mm wafer's particle scan: a scratch, a cluster and a scatter of loose particles - the defects per die,
  the density and the Poisson yield.
- A 100 mm wafer on an 11 x 11 site grid, a site on each 9 mm die: a hot spot binned high.

    python -m dev.demo.demo_wafer
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.demos import DEMO_DIR, DEMO_PROJECTS, DemoProject
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from dev.demo.demo_operations import (
    OperationDemo,
    _create_source_figure,
    _run_operation,
    _store_report,
)

#: Declared with the others in app.data.demos (the application lists them); found here by its builder.
WAFER_DEMO: DemoProject = next(demo for demo in DEMO_PROJECTS if demo.builder.endswith(":build_wafer_demo"))

_SITES = {"x": "x", "y": "y", "z": "z"}


def _thickness() -> pd.DataFrame:
    """49 sites: the centre, then rings of 8, 16 and 24 at a third, two thirds and nearly the edge."""
    rng = np.random.default_rng(11)
    x, y = [0.0], [0.0]
    for count, radius in ((8, 49.0), (16, 98.0), (24, 145.0)):
        angle = np.linspace(0.0, 2 * np.pi, count, endpoint=False) + np.pi / count
        x += list(radius * np.cos(angle))
        y += list(radius * np.sin(angle))
    x, y = np.array(x), np.array(y)
    nm = 1000.0 + 0.0022 * (x ** 2 + y ** 2) + 0.12 * x + rng.normal(0.0, 1.5, x.size)  # a bowl and a tilt
    return pd.DataFrame({"x_mm": x.round(2), "y_mm": y.round(2), "thickness_nm": nm.round(1)})


def _resistance() -> pd.DataFrame:
    """A 150 mm wafer read with its flat on the right (+x): high towards the flat, as an implant drifts."""
    rng = np.random.default_rng(12)
    axis = np.arange(-60.0, 61.0, 15.0)
    gx, gy = np.meshgrid(axis, axis)
    keep = np.hypot(gx, gy) <= 66
    x, y = gx[keep], gy[keep]
    ohms = 52.0 + 0.045 * x + 0.0006 * (x ** 2 + y ** 2) + rng.normal(0.0, 0.25, x.size)
    return pd.DataFrame({"x_mm": x, "y_mm": y, "rs_ohm_sq": ohms.round(2)})


def _particles() -> pd.DataFrame:
    """A scratch from the centre out, a cluster near the edge, and loose particles all over."""
    rng = np.random.default_rng(13)
    t = rng.uniform(0.0, 1.0, 70)
    scratch = np.column_stack((-20 + 75 * t + rng.normal(0, 0.6, 70), 10 + 40 * t + rng.normal(0, 0.6, 70)))
    cluster = rng.normal((-55.0, -60.0), 4.0, size=(45, 2))
    radius, angle = 98 * np.sqrt(rng.uniform(0, 1, 120)), rng.uniform(0, 2 * np.pi, 120)
    loose = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    points = np.vstack((scratch, cluster, loose))
    return pd.DataFrame({"x_mm": points[:, 0].round(2), "y_mm": points[:, 1].round(2)})


def _dense() -> pd.DataFrame:
    """11 x 11 sites on a 100 mm wafer: flat but for a hot spot off centre."""
    rng = np.random.default_rng(14)
    axis = np.linspace(-45.0, 45.0, 11)
    gx, gy = np.meshgrid(axis, axis)
    keep = np.hypot(gx, gy) <= 47
    x, y = gx[keep], gy[keep]
    value = 20.0 + 6.0 * np.exp(-((x - 15) ** 2 + (y - 10) ** 2) / (2 * 9.0 ** 2)) + rng.normal(0, 0.3, x.size)
    return pd.DataFrame({"x_mm": x, "y_mm": y, "line_width_nm": value.round(2)})


#: Every wafer sets all of these: the dialog remembers the last run's, and a rotation left from the one before
#: would turn the next wafer too.
_DEFAULTS: dict[str, Any] = {"rotation": 0.0, "lower": "", "upper": "", "die_offset_x": 0.0, "die_offset_y": 0.0}


def _setup(values: dict[str, Any]):
    def setup(dialog: Any) -> None:
        dialog._parameter_form_spec.set_values({**_DEFAULTS, **values})
    return setup


TABLES = {"film_thickness": _thickness, "sheet_resistance": _resistance, "particle_scan": _particles,
          "line_width": _dense}

OPERATIONS: tuple[OperationDemo, ...] = (
    OperationDemo(
        "Film thickness - a 300 mm wafer, 49 sites", "Film thickness (nm)", "film_thickness",
        "SELECT x_mm AS x, y_mm AS y, thickness_nm AS z FROM film_thickness", _SITES, "x (mm)", "y (mm)",
        "wafer_map_dialog:WaferMapDialog",
        style={"marker": "o", "markersize": 5, "linestyle": ""},
        setup=_setup({"diameter": "300", "mark": "notch", "mark_angle": "270", "edge_exclusion": 3.0,
                      "die_width": 26.0, "die_height": 33.0, "lower": "1005", "upper": "1040"})),
    OperationDemo(
        "Sheet resistance - a 150 mm wafer with its flat, rotated", "Sheet resistance (ohm/sq)", "sheet_resistance",
        "SELECT x_mm AS x, y_mm AS y, rs_ohm_sq AS z FROM sheet_resistance", _SITES, "x (mm)", "y (mm)",
        "wafer_map_dialog:WaferMapDialog",
        style={"marker": "s", "markersize": 5, "linestyle": ""},
        # Read with the flat on the right: a quarter turn clockwise brings it to the bottom.
        setup=_setup({"diameter": "150", "mark": "flat", "mark_angle": "270", "rotation": -90.0,
                      "edge_exclusion": 5.0, "die_width": 15.0, "die_height": 15.0, "lower": "49", "upper": "56"})),
    OperationDemo(
        "Particle scan - a 200 mm wafer's defects", "Particles", "particle_scan",
        "SELECT x_mm AS x, y_mm AS y FROM particle_scan", {"x": "x", "y": "y"}, "x (mm)", "y (mm)",
        "wafer_map_dialog:WaferMapDialog",
        style={"marker": "o", "markersize": 3, "linestyle": ""},
        setup=_setup({"diameter": "200", "mark": "notch", "mark_angle": "270", "edge_exclusion": 2.0,
                      "die_width": 12.0, "die_height": 12.0})),
    OperationDemo(
        "Line width - a 100 mm wafer, a site per die", "Line width (nm)", "line_width",
        "SELECT x_mm AS x, y_mm AS y, line_width_nm AS z FROM line_width", _SITES, "x (mm)", "y (mm)",
        "wafer_map_dialog:WaferMapDialog",
        style={"marker": "o", "markersize": 4, "linestyle": ""},
        setup=_setup({"diameter": "100", "mark": "flat", "mark_angle": "270", "edge_exclusion": 3.0,
                      "die_width": 9.0, "die_height": 9.0, "upper": "23"})),
)


def build_wafer_demo(directory: Path = DEMO_DIR) -> Path:
    """Write the Wafer Map demo into *directory* and return its path."""
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
        path = Path(directory) / WAFER_DEMO.path_name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", ".undo.db"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repo = SqliteRepo(db_path=path)
        for table, make in TABLES.items():
            repo.import_dataframe(make(), table_name=table, normalize_columns=False)
        for demo in OPERATIONS:
            figure_id = _create_source_figure(repo, demo)
            try:
                _store_report(repo, figure_id, _run_operation(repo, figure_id, demo))
            except Exception:
                applogger.exception("Demo: %s failed.", demo.name)
        repo.close()
        Path(f"{path}.undo.db").unlink(missing_ok=True)
        return path
    finally:
        config.USER_CONFIG_PATH = real_user_config
        QMessageBox.exec = real_exec  # type: ignore[method-assign]
        del app


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    print(build_wafer_demo())
