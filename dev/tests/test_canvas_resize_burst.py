"""A burst of resizes draws the chart once, at the final size (W-04)."""
from __future__ import annotations

import time

from matplotlib.figure import Figure

from app.widgets.chart_panel import canvas as canvas_module
from app.widgets.chart_panel.canvas import ChartCanvas


def _settle(qapp, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()


def test_a_burst_of_resizes_draws_once_when_it_settles(qapp, monkeypatch) -> None:
    figure = Figure(dpi=100)
    figure.add_subplot().plot([0, 1], [1, 0])
    canvas = ChartCanvas(figure)
    canvas.resize(400, 300)
    canvas.show()
    canvas.draw()
    _settle(qapp, 0.3)
    draws: list[tuple[int, int]] = []
    original = ChartCanvas.draw
    monkeypatch.setattr(ChartCanvas, "draw", lambda self: (draws.append(self.get_width_height()), original(self))[1])
    try:
        canvas.resize(410, 300)  # the first of the burst goes through
        for step in range(1, 10):
            canvas.resize(410 + 10 * step, 300)
            canvas.repaint()  # each step paints - the stretched image, not a draw
            qapp.processEvents()
        assert canvas.resize_deferred
        drawn_mid_burst = len(draws)
        _settle(qapp, (canvas_module.SETTLE_MS + 150) / 1000.0)
        assert not canvas.resize_deferred
        assert drawn_mid_burst <= 1
        assert len(draws) - drawn_mid_burst == 1
        assert draws[-1][0] == 500
    finally:
        canvas.close()
