"""Titles, labels, legend entries and annotations over several lines: "\\n" typed, and how the lines align."""
from __future__ import annotations

from matplotlib.figure import Figure

from app.plugin_api.charts.text_lines import display_text


def test_a_typed_backslash_n_is_a_new_line_except_in_mathtext() -> None:
    assert display_text(r"Time\n(s)") == "Time\n(s)"
    assert display_text(r"$\nu$ rate\n(Hz)") == "$\\nu$ rate\n(Hz)"
    assert display_text(r"keep \\n") == "keep \\n"
    assert display_text("plain") == "plain"


def test_the_figure_draws_them_over_several_lines(qapp, repo) -> None:
    import pandas as pd

    from app.plugin_api.charts.render_figure import render_figure_from_descriptor

    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), table_name="t", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="f")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot", title=r"First\nSecond",
        x_label=r"Time\n(s)", y_label="y",
        options={"label_multialignment": "left", "legend": True, "show_legend": True,
                 "annotations": [{"type": "text", "x": 1, "y": 3, "text": r"one\ntwo"}]})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=r"series\nname",
                                  sql_query="SELECT a AS x, b AS y FROM t", roles={"x": "x", "y": "y"}, style={})
    figure = Figure()
    render_figure_from_descriptor(figure=figure, descriptor=repo.load_figure_descriptor(figure_id), repo=repo)
    ax = figure.axes[0]
    assert ax.get_title() == "First\nSecond" and ax.get_xlabel() == "Time\n(s)"
    assert ax.xaxis.label._multialignment == "left"  # Matplotlib has no getter for it
    assert any(text.get_text() == "one\ntwo" for text in ax.texts)
    legend = ax.get_legend()
    if legend is not None:
        assert [text.get_text() for text in legend.get_texts()] == ["series\nname"]
