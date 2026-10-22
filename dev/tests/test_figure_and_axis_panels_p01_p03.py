"""Figure and Axis properties as todo P-01..P-03 asked: View on its own above Style, no Default
style button, the Grid card folds, Border draws a line round the figure, the axis title takes two lines."""
from __future__ import annotations

from types import SimpleNamespace

from matplotlib.figure import Figure

from app.plugin_api.charts.render_figure import _apply_figure_options
from app.styles.style.widgets import TitledCard
from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget
from app.widgets.chart_properties.figure_properties import FigurePropertiesWidget


def test_view_comes_before_style_and_the_grid_folds(qapp) -> None:
    panel = FigurePropertiesWidget(None)
    titles = [card.title_label.text().lower() for card in panel.findChildren(TitledCard)]
    assert titles.index("view") < titles.index("style")
    assert not hasattr(panel, "_btn_default")
    grid = next(card for card in panel.findChildren(TitledCard) if card.title_label.text().lower() == "grid")
    assert grid.disclosure is not None


def test_border_draws_a_line_round_the_figure() -> None:
    for border in (True, False):
        figure = Figure()
        figure.patch.set_linewidth(0.0)
        _apply_figure_options(figure, SimpleNamespace(options={"border": border}))
        assert (figure.patch.get_linewidth() > 0) == border


def test_a_two_line_title_is_stored_with_a_typed_line_break(qapp) -> None:
    panel = AxisPropertiesWidget(None)
    sent: list[dict] = []
    panel.axis_options_requested.connect(sent.append)
    panel._current_axis_id = 1
    panel._axis_label_edit.setPlainText("Time\n(s)")
    panel._emit_axis_options_requested()
    assert sent and sent[-1]["label"] == "Time\\n(s)"
