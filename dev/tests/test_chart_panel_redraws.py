"""A chart is drawn when what it shows changes, and not otherwise.

Each full draw re-runs every renderer and the layout engine - 50 to 150 ms
for an ordinary figure - so a stray one per mouse move made hovering lag,
worst on Windows. These pin the cases that used to draw for nothing.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel
from dev.tests._figure_factory import create_renderer_showcase_db


@pytest.fixture(scope="module")
def showcase(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[SqliteRepo, dict[str, int]]]:
    path = tmp_path_factory.mktemp("redraws") / "showcase.dhub"
    ids = create_renderer_showcase_db(path, n_points=200)
    repo = SqliteRepo(db_path=path)
    yield repo, ids
    repo.close()


@pytest.fixture
def draws(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counted: list[int] = []
    original = FigureCanvasAgg.draw

    def draw(self, *args, **kwargs):
        counted.append(id(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FigureCanvasAgg, "draw", draw)
    return counted


def _settle(seconds: float = 0.3) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture
def panel(qapp, showcase) -> Iterator[ChartPanel]:
    repo, ids = showcase
    widget = ChartPanel(repo, ids["Scatter Plot"])
    widget.set_resize_mode("FIT", persist=False)
    widget.resize(800, 600)
    widget.show()
    _settle()
    yield widget
    widget.close()


def test_hovering_never_redraws_the_figure(panel: ChartPanel, draws: list[int]) -> None:
    canvas = panel._canvas
    for step in range(30):
        point = QPointF(canvas.width() * (0.2 + 0.02 * step), canvas.height() * 0.5)
        event = QMouseEvent(QEvent.Type.MouseMove, point, point, Qt.MouseButton.NoButton,
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(canvas, event)
        _settle(0.02)
    assert draws == []


def test_a_layout_request_checks_but_does_not_redraw(panel: ChartPanel, draws: list[int]) -> None:
    QCoreApplication.postEvent(panel, QEvent(QEvent.Type.LayoutRequest))
    _settle(0.2)
    assert draws == []


def test_a_real_resize_redraws_once(panel: ChartPanel, draws: list[int]) -> None:
    panel.resize(900, 650)
    _settle()
    assert draws == [id(panel._canvas)]
    width, height = panel._figure.get_size_inches() * panel._figure.dpi
    ratio = panel._canvas.device_pixel_ratio
    assert (round(width), round(height)) == (round(panel._canvas.width() * ratio), round(panel._canvas.height() * ratio))


def test_a_reload_redraws_and_a_hidden_panel_waits(qapp, showcase, draws: list[int]) -> None:
    repo, ids = showcase
    hidden = ChartPanel(repo, ids["Bar Chart"])
    _settle()
    assert id(hidden._canvas) not in draws
    hidden.resize(700, 500)
    hidden.show()
    _settle()
    assert draws.count(id(hidden._canvas)) == 1
    hidden.reload()
    _settle()
    assert draws.count(id(hidden._canvas)) == 2
    hidden.hide()
    hidden.show()
    _settle()
    # Shown again unchanged: the canvas still holds the drawing.
    assert draws.count(id(hidden._canvas)) == 2
    hidden.close()


def test_the_first_hover_after_a_redraw_needs_no_draw_of_its_own(panel: ChartPanel, draws: list[int]) -> None:
    assert panel._hover_background is not None
    panel.reload()
    _settle()
    assert panel._hover_background is not None
    assert draws == [id(panel._canvas)]
