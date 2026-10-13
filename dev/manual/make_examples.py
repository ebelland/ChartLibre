"""Regenerate the user manual's example pictures: one per chart type, data operation and analysis.

Run it from the project root::

    python dev/manual/make_examples.py

and it rewrites ``docs/manual/examples/*.png``. The pictures are the
application's own drawings - the demo projects' figures, rendered by the
same code the chart panel uses, in each figure's own style - and, for the
analyses no demo carries, the analysis windows run on real data (the Palmer
penguins) exactly as a person would press OK in them. Nothing is drawn by
hand, so a picture cannot drift from what the application shows.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "docs" / "manual" / "examples"
DPI = 110
SINGLE = (6.0, 3.4)
MULTI = (9.0, 5.6)

#: Chart type -> the demo figure that shows it best (by part of its name); otherwise the first single-axis one.
PREFERRED = {
    "Scatter Plot": "Penguin bill dimensions",
    "Histogram": "Penguin body mass",
    "Box Plot": "Employee compensation",
    "Time Series": "Stock prices",
    "Bar Chart": "Transactions by channel",
}
#: Data operation picture -> part of its figure's name in the "Series operations" demo.
OPERATIONS = {
    "op_smoothing": "01 · Smoothing", "op_spectral": "02 · Spectral", "op_filtering": "03 · Filtering",
    "op_baseline": "04 · Baseline", "op_peaks": "05 · Peaks", "op_roots": "06 · Roots",
    "op_calculus": "07 · Calculus", "op_fit": "08 · Fit", "op_interpolation": "09 · Interpolation",
    "op_statistics": "10 · Statistics", "op_outliers": "11 · Outliers", "op_clustering": "12 · Clustering",
    "op_control_chart": "13 · Control chart", "op_regression": "14 · Regression", "op_transform": "15 · Transform",
}


def slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_").replace("__", "_")


def _save(repo, figure_id: int, name: str, size: tuple[float, float]) -> None:
    from app.charts.render_figure import render_figure_from_descriptor

    figure = Figure(figsize=size, dpi=DPI)
    render_figure_from_descriptor(figure=figure, descriptor=repo.load_figure_descriptor(int(figure_id)), repo=repo)
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"  wrote examples/{name}.png")


def _figures(repo) -> list[tuple[int, str, list[str]]]:
    return [(int(fid), str(name), [a.name for a in repo.load_figure_descriptor(int(fid)).axes])
            for fid, name, *_rest in repo.load_figures_from_db()]


def charts(demos: list[Path]) -> None:
    from app.data.sqlite_repo import SqliteRepo

    chosen: dict[str, tuple[Path, int]] = {}
    for path in demos:
        repo = SqliteRepo(db_path=path)
        for figure_id, name, types in _figures(repo):
            if len(types) != 1:
                continue
            kind = types[0]
            preferred = PREFERRED.get(kind)
            if kind not in chosen or (preferred and preferred in name and preferred not in chosen[kind][2]):
                chosen[kind] = (path, figure_id, name)
        repo.close()
    for kind, (path, figure_id, _name) in sorted(chosen.items()):
        repo = SqliteRepo(db_path=path)
        _save(repo, figure_id, f"chart_{slug(kind)}", SINGLE)
        repo.close()


def demo_figures(demos: list[Path], demo_part: str, pictures: dict[str, str], size) -> None:
    from app.data.sqlite_repo import SqliteRepo

    path = next(p for p in demos if demo_part in p.name)
    repo = SqliteRepo(db_path=path)
    figures = _figures(repo)
    for picture, part in pictures.items():
        figure_id = next(fid for fid, name, _types in figures if part in name)
        _save(repo, figure_id, picture, size)
    repo.close()


def _run(app, repo, dialog_class, table: str, roles: dict, kinds: dict | None = None, setup=None) -> int:
    dialog = dialog_class(repo=repo, table=table)
    try:
        dialog.roles_widget.set_casting(roles, kinds)
        if setup is not None:
            setup(dialog)
        dialog.ok()
        deadline = time.monotonic() + 120
        while dialog._task is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        return int(dialog.created_figure_ids[0])
    finally:
        dialog.close()


def analyses(scratch: Path) -> None:
    """The analyses no demo carries, run on the Palmer penguins (and a spectrum set for PLS)."""
    from PySide6.QtWidgets import QApplication

    from app.data.sqlite_repo import SqliteRepo
    from app.table_operations.discriminant_dialog import DiscriminantDialog
    from app.table_operations.partition_dialog import PartitionDialog
    from app.table_operations.pls_dialog import PLSDialog
    from dev.demo.build_demos import TABLE_SOURCES

    app = QApplication.instance() or QApplication([])
    repo = SqliteRepo(db_path=scratch / "analyses.dhub")
    penguins = TABLE_SOURCES["penguins"]()
    repo.import_dataframe(penguins, table_name="penguins", normalize_columns=False)
    measures = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

    def tree_charts(dialog) -> None:
        dialog._parameter_form.set_values({"splits": 4})
        for key, check in dialog._chart_checks.items():
            check.setChecked(key in ("partition", "importance"))

    figure_id = _run(app, repo, PartitionDialog, "penguins",
                     {"response": ["body_mass_g"], "factor": ["species", "sex", "flipper_length_mm", "bill_depth_mm"]},
                     {"species": "nominal", "sex": "nominal", "flipper_length_mm": "continuous",
                      "bill_depth_mm": "continuous", "body_mass_g": "continuous"}, tree_charts)
    _save(repo, figure_id, "analysis_partition", MULTI)
    figure_id = _run(app, repo, DiscriminantDialog, "penguins", {"response": ["species"], "covariate": measures},
                     {"species": "nominal", **{name: "continuous" for name in measures}})
    _save(repo, figure_id, "analysis_discriminant", SINGLE)

    # PLS needs a wide table: near-infrared-like spectra of two absorbers, the first one's
    # concentration the response - synthetic, so the answer is known.
    rng = np.random.default_rng(0)
    wavelengths = np.linspace(900, 1700, 120)
    conc, other = rng.uniform(0, 1, 60), rng.uniform(0, 1, 60)
    band = lambda centre, width: np.exp(-((wavelengths - centre) / width) ** 2)  # noqa: E731
    spectra = np.outer(conc, band(1200, 30)) + np.outer(other, band(1450, 60)) + 0.3 + rng.normal(0, 0.01, (60, 120))
    frame = pd.DataFrame(spectra, columns=[f"nm{int(w)}" for w in wavelengths])
    frame["concentration"] = conc
    repo.import_dataframe(frame, table_name="spectra", normalize_columns=False)
    figure_id = _run(app, repo, PLSDialog, "spectra",
                     {"response": ["concentration"], "predictor": [c for c in frame.columns if c != "concentration"]})
    _save(repo, figure_id, "analysis_pls", MULTI)
    repo.close()


def main() -> int:
    import app.utils.config as config
    from app.data.demos import DEMO_DIR

    scratch = Path(tempfile.mkdtemp(prefix="chartlibre-examples-"))
    config.USER_CONFIG_PATH = scratch / "user.json"
    # Copies: opening a project writes beside it, and the shipped demos must stay as they are.
    demos = []
    for source in sorted(DEMO_DIR.glob("*.dhub")):
        target = scratch / source.name
        shutil.copy2(source, target)
        demos.append(target)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Writing examples:")
    charts(demos)
    demo_figures(demos, "Series operations", OPERATIONS, SINGLE)
    demo_figures(demos, "3D Series Operations", {"op_surface": "Series Operations on 3D data"}, MULTI)
    demo_figures(demos, "Designed experiment", {"analysis_fit_model": "Fit Model · Yield"}, (9.0, 3.4))
    demo_figures(demos, "Quick answers", {"answer_differences": "Are there differences?",
                                          "answer_impact": "What has the most impact?",
                                          "answer_change": "Has it changed over time?"}, SINGLE)
    analyses(scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
