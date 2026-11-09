"""Marker every, custom dashes and custom markers, drawn through the real renderers."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.plugin_api.charts.render_figure import render_figure_from_descriptor


@pytest.mark.parametrize("chart", ["Scatter Plot", "Time Series"])
def test_drawn(qapp, repo, chart):
    repo.import_dataframe(pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) ** 0.5}), table_name="d",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type=chart, title="t",
                                              x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query='SELECT a AS x, b AS y FROM d',
                                  roles={"x": "x", "y": "y"},
                                  style={"marker": "★", "linestyle": "--.", "markevery": 10})
    figure = Figure()
    render_figure_from_descriptor(figure=figure, descriptor=repo.load_figure_descriptor(figure_id), repo=repo)
    ax = figure.axes[0]
    drawn = [len(c.get_offsets()) for c in ax.collections] + [
        len(line.get_xdata()[::line.get_markevery()]) for line in ax.lines
        if line.get_marker() not in ("None", "") and isinstance(line.get_markevery(), int)]
    assert 4 in drawn, (drawn, [(ln.get_marker(), ln.get_markevery()) for ln in ax.lines])


def test_max_points_is_the_axis_own_else_the_figures():
    from types import SimpleNamespace

    from app.plugin_api.charts.render_figure import _axis_downsample_threshold

    assert _axis_downsample_threshold(SimpleNamespace(options={}), 1000) == 1000
    assert _axis_downsample_threshold(SimpleNamespace(options={"downsample_threshold": 100}), 1000) == 100
    assert _axis_downsample_threshold(SimpleNamespace(options={"downsample_threshold": 0}), 1000) == 0
