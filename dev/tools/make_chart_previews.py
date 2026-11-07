"""Draw the small example of every chart type that New plot shows: app/charts/previews/<chart>.png.

    python dev/tools/make_chart_previews.py

Run again after adding a chart type or changing how one looks. The data are the tests' showcase figures.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from app.data.sqlite_repo import SqliteRepo
from app.plugin_api.charts.render_figure import render_figure_from_descriptor
from app.widgets.renderer_picker import preview_path
from dev.tests._figure_factory import create_renderer_showcase_db


def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        database = Path(folder) / "previews.dhub"
        figure_ids = create_renderer_showcase_db(database, n_points=400)
        repo = SqliteRepo(db_path=database)
        for chart_type, figure_id in figure_ids.items():
            descriptor = repo.load_figure_descriptor(figure_id=figure_id)
            figure = Figure(figsize=(6.4, 4.0))
            render_figure_from_descriptor(figure=figure, descriptor=descriptor, repo=repo)
            target = preview_path(chart_type)
            target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(target, dpi=70)
        repo.close()


if __name__ == "__main__":
    main()
