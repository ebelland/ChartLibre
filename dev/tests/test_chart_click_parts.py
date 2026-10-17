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


def test_flash_highlights_an_axis_or_a_series_for_a_moment(panel: ChartPanel) -> None:
    """S-01, the other way: what Chart properties selects is shown on the chart, then gone."""
    import app.widgets.chart_panel.picking as picking

    panel.ensure_rendered()
    axes = next(a for a in panel._canvas.figure.axes if getattr(a, "_dhub_axis_id", None) is not None)
    before = len(axes.patches) + len(axes.lines) + len(axes.collections)
    name = next(str(a.get_label()) for a in (*axes.collections, *axes.lines) if not str(a.get_label()).startswith("_"))

    assert panel.flash(axes._dhub_axis_id, name)
    assert len(axes.patches) + len(axes.lines) + len(axes.collections) > before
    end = time.monotonic() + picking.FLASH_MS / 1000 + 0.5
    while time.monotonic() < end:
        QApplication.processEvents()
    assert len(axes.patches) + len(axes.lines) + len(axes.collections) == before

    assert panel.flash(axes._dhub_axis_id)  # no series: a frame round the axis
    assert any(str(p.get_label()) == "_flash" for p in axes.patches)
    assert not panel.flash(-1)
    # Let it clear here: its timer would otherwise redraw this chart during the next test.
    end = time.monotonic() + picking.FLASH_MS / 1000 + 0.5
    while time.monotonic() < end:
        QApplication.processEvents()
    assert not any(str(p.get_label()) == "_flash" for p in axes.patches)


def test_hovering_an_axis_says_what_a_click_will_edit(panel: ChartPanel) -> None:
    from types import SimpleNamespace

    from PySide6.QtCore import Qt

    panel.ensure_rendered()
    panel._canvas.draw()
    axes = next(a for a in panel._canvas.figure.axes if getattr(a, "_dhub_axis_id", None) is not None)
    box = axes.xaxis.get_tightbbox(panel._canvas.get_renderer())
    panel._show_axis_part_hint(SimpleNamespace(x=(box.x0 + box.x1) / 2, y=(box.y0 + box.y1) / 2))
    assert panel._canvas.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert "X axis" in panel._canvas.toolTip()
    panel._show_axis_part_hint(None)
    assert panel._canvas.toolTip() == ""


def test_escape_on_the_chart_goes_back_to_the_figure(panel: ChartPanel) -> None:
    """S-01: Esc says "the figure", as a click outside every axis does - not while a tool owns the mouse."""
    from types import SimpleNamespace

    hits: list[dict] = []
    panel.chart_object_clicked.connect(hits.append)
    panel._on_escape_to_figure(SimpleNamespace(key="a"))
    panel._on_escape_to_figure(SimpleNamespace(key="escape"))
    assert hits == [{"axis_id": None, "series": None, "part": None}]


def test_a_series_style_edit_shows_on_its_lines_without_a_render(panel: ChartPanel) -> None:
    """S-01: colour, line and marker edits change what is drawn; anything else asks for a render."""
    from matplotlib.colors import to_hex

    panel.ensure_rendered()
    axes = next(a for a in panel._canvas.figure.axes if getattr(a, "_dhub_axis_id", None) is not None)
    axis_id = axes._dhub_axis_id
    # The showcase's points are coloured and sized by columns: only a render redoes those.
    cloud = next(a for a in axes.collections if getattr(a, "_dhub_series", None))
    assert not panel.preview_series_style(axis_id, cloud._dhub_series, {"color": "#ff0000"})
    assert not panel.preview_series_style(axis_id, cloud._dhub_series, {"markersize": 9})
    assert panel.preview_series_style(axis_id, cloud._dhub_series, {"marker": "s"})

    # A series drawn as a line and points, as the Scatter Plot renderer tags them.
    line, = axes.plot([1, 2, 3], [1, 2, 3], linestyle="-", marker="None")
    points = axes.scatter([1, 2, 3], [1, 2, 3], s=36)
    line._dhub_series = points._dhub_series = "manual"
    assert panel.preview_series_style(axis_id, "manual", {"color": "#ff0000", "markersize": 9, "linestyle": "--"})
    assert to_hex(line.get_color()) == to_hex(points.get_facecolor()[0]) == "#ff0000"
    assert list(points.get_sizes()) == [81.0] and line.get_linestyle() == "--"
    assert not panel.preview_series_style(axis_id, "manual", {"label": "other"})
    assert not panel.preview_series_style(axis_id, "manual", {"marker": ""})  # the points would go
    assert not panel.preview_series_style(axis_id, "manual", {"linestyle": "None"})  # the line would go
    assert not panel.preview_series_style(axis_id, "manual", {"color": "not a colour"})
    assert not panel.preview_series_style(axis_id, "no such series", {"color": "#00ff00"})


def test_a_time_series_edit_restyles_its_rolling_line_only_where_it_follows(qapp, tmp_path: Path) -> None:
    """S-01: the rolling mean takes the series' colour, not its line style."""
    from matplotlib.colors import to_hex

    ids = create_renderer_showcase_db(tmp_path / "ts.dhub", n_points=20)
    repo = SqliteRepo(db_path=tmp_path / "ts.dhub")
    widget = ChartPanel(repo, ids["Time Series"])
    try:
        widget.ensure_rendered()
        axes = next(a for a in widget._canvas.figure.axes if getattr(a, "_dhub_axis_id", None) is not None)
        raw, rolling = (l for l in axes.lines if getattr(l, "_dhub_series", None) == "second")
        assert widget.preview_series_style(axes._dhub_axis_id, "second", {"color": "#ff0000", "linestyle": ":"})
        assert to_hex(raw.get_color()) == to_hex(rolling.get_color()) == "#ff0000"
        assert raw.get_linestyle() == ":" and rolling.get_linestyle() == "--"
    finally:
        widget.close()
        repo.close()
