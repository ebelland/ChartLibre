"""S-01: a click on the chart names what it landed on - outside the plotting area too."""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel
from dev.tests._figure_factory import create_renderer_showcase_db


@pytest.fixture
def panel(qapp, tmp_path: Path) -> Iterator[ChartPanel]:
    ids = create_renderer_showcase_db(tmp_path / "showcase.dhub", n_points=50)
    repo = SqliteRepo(db_path=tmp_path / "showcase.dhub")
    widget = ChartPanel(repo, ids["Scatter Plot"])
    widget.resize(800, 600)
    widget.show()
    end = time.monotonic() + 0.3
    while time.monotonic() < end:
        QApplication.processEvents()
    yield widget
    widget.close()
    repo.close()


def test_a_click_on_an_axis_ticks_or_title_names_that_part(panel: ChartPanel) -> None:
    """S-01: outside the plotting area is not always the figure - the X or Y ticks, or the title."""
    from types import SimpleNamespace

    panel.ensure_rendered()
    panel._canvas.draw()
    axes = next(a for a in panel._canvas.figure.axes if getattr(a, "_dhub_axis_id", None) is not None)
    renderer = panel._canvas.get_renderer()

    def at(box) -> SimpleNamespace:
        return SimpleNamespace(x=(box.x0 + box.x1) / 2, y=(box.y0 + box.y1) / 2)

    # The middle of the room each axis' ticks and label take (a tick label
    # in the list may be off the view, and not drawn).
    assert panel._axis_part_at(at(axes.xaxis.get_tightbbox(renderer))) == (axes, "x")
    assert panel._axis_part_at(at(axes.yaxis.get_tightbbox(renderer))) == (axes, "y")
    if axes.title.get_text():
        assert panel._axis_part_at(at(axes.title.get_window_extent(renderer))) == (axes, "title")
    assert panel._axis_part_at(SimpleNamespace(x=1, y=1)) == (None, None)
