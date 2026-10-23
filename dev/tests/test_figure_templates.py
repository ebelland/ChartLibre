"""Figure templates (G-18): a figure's layout and settings saved and applied, no data, no style."""
from __future__ import annotations

from pathlib import Path

from app.data import figure_templates as ft
from app.data.sqlite_repo import SqliteRepo


def _source(repo: SqliteRepo) -> int:
    """Two axes overlapping (the second a twin of the first), letters, a size, a style, a series."""
    figure = repo.create_figure_descriptor(name="source", nrows=1, ncols=1, options={
        "panel_labels": "a", "figure_width_cm": 8.9, "mpl_style": "lines.linewidth: 3", "border": True})
    first = repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Time Series", title="T",
                                        x_label="Time", y_label="Temp", options={"y_scale": "log"})
    second = repo.create_axis_descriptor(figure_id=figure, axis_index=1, chart_type="Scatter Plot", title="",
                                         x_label="", y_label="Flow", options={"twin_of": first, "y_break": "1, 5"})
    repo.create_series_descriptor(axis_id=first, series_index=0, name="s", sql_query="SELECT 1 AS x, 2 AS y",
                                  roles={"x": "x", "y": "y"}, style={"color": "red"})
    del second
    return figure


def test_a_template_keeps_layout_types_and_settings_but_no_data_and_no_style(repo: SqliteRepo) -> None:
    template = ft.to_template(repo.load_figure_descriptor(_source(repo)), "Paper")
    assert template["figure"] == {"panel_labels": "a", "figure_width_cm": 8.9, "border": True}
    assert [axis["chart_type"] for axis in template["axes"]] == ["Time Series", "Scatter Plot"]
    assert template["axes"][1]["options"] == {"twin_of": 0, "y_break": "1, 5"}  # a position, not an id
    text = str(template)
    assert "SELECT" not in text and "red" not in text and "linewidth" not in text


def test_applied_to_another_figure_by_position(repo: SqliteRepo, tmp_path: Path) -> None:
    template = ft.load(ft.save(ft.to_template(repo.load_figure_descriptor(_source(repo)), "Paper"), tmp_path))
    assert [name for name, _path in ft.available(tmp_path)] == ["Paper"]
    target = repo.create_figure_descriptor(name="target", options={"mpl_style": "mine"})
    only = repo.create_axis_descriptor(figure_id=target, axis_index=0, chart_type="Scatter Plot", title="",
                                       x_label="", y_label="", options={"x_scale": "log"})
    repo.create_series_descriptor(axis_id=only, series_index=0, name="d", sql_query="SELECT 1 AS x, 2 AS y",
                                  roles={"x": "x", "y": "y"})
    notes = ft.apply_template(repo, target, template)
    figure = repo.load_figure_descriptor(target)
    axes = sorted(figure.axes, key=lambda axis: axis.index)
    assert [axis.name for axis in axes] == ["Time Series", "Scatter Plot"]  # the second one made
    assert axes[0].series and axes[0].options.get("y_scale") == "log" and "x_scale" not in axes[0].options
    assert axes[1].options["twin_of"] == axes[0].id and not axes[1].series
    assert figure.options["mpl_style"] == "mine" and figure.options["panel_labels"] == "a"
    assert notes == []


def test_a_chart_type_the_series_cannot_take_is_kept_with_a_note(repo: SqliteRepo) -> None:
    template = {"version": 1, "nrows": 1, "ncols": 1, "figure": {},
                "axes": [{"position": 0, "chart_type": "Heatmap", "options": {}}]}
    target = repo.create_figure_descriptor(name="t")
    axis = repo.create_axis_descriptor(figure_id=target, axis_index=0, chart_type="Scatter Plot", title="",
                                       x_label="", y_label="", options={})
    repo.create_series_descriptor(axis_id=axis, series_index=0, name="d", sql_query="SELECT 1 AS x, 2 AS y",
                                  roles={"x": "x", "y": "y"})
    notes = ft.apply_template(repo, target, template)
    assert repo.load_figure_descriptor(target).axes[0].name == "Scatter Plot" and notes
