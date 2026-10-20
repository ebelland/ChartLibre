"""An emptied Title field means no title - not the title the axis was created with."""
from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from app.plugin_api.charts.axis_options import axis_title
from app.plugin_api.charts.render_figure import render_figure_from_descriptor


def test_the_options_title_wins_even_empty() -> None:
    assert axis_title({"title": ""}, "Created") == ""
    assert axis_title({"title": " Mine "}, "Created") == "Mine"
    assert axis_title({}, "Created") == "Created"  # never edited: the descriptor's own
    assert axis_title({"label": "Old style"}, "Created") == "Old style"


def test_an_emptied_title_draws_none(qapp, repo) -> None:
    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0]}), table_name="t", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="f")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="Created", x_label="", y_label="", options={"title": ""})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query="SELECT a AS x, a AS y FROM t",
                                  roles={"x": "x", "y": "y"}, style={})
    figure = Figure()
    render_figure_from_descriptor(figure=figure, descriptor=repo.load_figure_descriptor(figure_id), repo=repo)
    assert figure.axes[0].get_title() == ""
