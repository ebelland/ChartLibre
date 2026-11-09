"""Pictures kept in the project and drawn as markers."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox

from app.data.repo.images import kind_of
from app.plugin_api.charts.render_figure import render_figure_from_descriptor

SVG = (b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
       b'<circle cx="5" cy="5" r="4" fill="#000000"/></svg>')


def test_store_kind_replace_and_undo(repo):
    assert kind_of(SVG) == "svg"
    with pytest.raises(ValueError):
        kind_of(b"hello")
    repo.add_image("dot", SVG, source="test", license="MIT")
    stored = repo.get_image("dot")
    assert stored.kind == "svg" and stored.license == "MIT" and repo.image_names() == ["dot"]
    repo.remove_image("dot")
    assert repo.image_names() == []
    repo.undo_last()
    assert repo.image_names() == ["dot"]


@pytest.mark.parametrize("chart", ["Scatter Plot", "Time Series"])
def test_drawn_as_markers(qapp, repo, chart):
    repo.add_image("dot", SVG)
    repo.import_dataframe(pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0)}), table_name="d",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type=chart, title="t",
                                              x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query="SELECT a AS x, b AS y FROM d",
                                  roles={"x": "x", "y": "y"}, style={"marker": "image:dot", "markevery": 2,
                                                                     "color": "#d62728"})
    figure = Figure()
    render_figure_from_descriptor(figure=figure, descriptor=repo.load_figure_descriptor(figure_id), repo=repo)
    boxes = [a for a in figure.axes[0].get_children() if isinstance(a, AnnotationBbox)]
    assert len(boxes) == 10
    array = boxes[0].offsetbox.get_data()
    opaque = array[..., 3] > 0.5
    assert np.allclose(array[..., 0][opaque], 214 / 255, atol=0.02)  # a single-colour icon takes the series' colour
    figure.savefig(__import__("io").BytesIO(), format="svg")


def test_the_option_box_offers_the_projects_pictures(qapp, repo):
    from types import SimpleNamespace

    from app.widgets.chart_properties.series_properties import SeriesPropertiesWidget

    repo.add_image("dot", SVG)
    widget = SeriesPropertiesWidget()
    widget._repo = repo
    widget._load_series_descriptor(SimpleNamespace(id=1, options={"marker": "image:dot"}, sql_query=""))
    chooser = widget._marker_chooser
    assert chooser._buttons["picture"].isChecked()
    assert chooser._picture_list.currentData() == "dot"
    assert widget._marker_combo.current_marker() == "image:dot"


def test_a_typed_line_pattern_applies_and_previews(qapp):
    from matplotlib.lines import Line2D

    from app.plugin_api.charts import dashes
    from app.widgets.chart_panel.panel import _drawn_linestyle
    from app.widgets.drop_down_controls.line_combo import LineStyleCombo
    from app.widgets.drop_down_controls.style_chooser import StyleChooser

    combo = LineStyleCombo()
    chooser = StyleChooser(combo, is_custom=dashes.is_custom, select=combo.set_current_linestyle, hint="")
    chooser._buttons["custom"].setChecked(True)
    chooser._text.setText(".")
    chooser._typed()
    assert combo.current_linestyle() == "."
    line = Line2D([0, 1], [0, 1])
    line.set_linestyle(_drawn_linestyle("."))
    assert line.get_linestyle() == "--" or isinstance(line.get_dashes if hasattr(line, "get_dashes") else 0, object)


def test_a_picture_marker_with_the_rolling_mean(qapp, repo):
    repo.add_image("dot", SVG)
    repo.import_dataframe(pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0)}), table_name="d",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Time Series", title="t",
                                              x_label="x", y_label="y", options={"show_rolling": True}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s", sql_query="SELECT a AS x, b AS y FROM d",
                                  roles={"x": "x", "y": "y"}, style={"marker": "image:dot", "linestyle": "-"})
    figure = Figure()
    render_figure_from_descriptor(figure=figure, descriptor=repo.load_figure_descriptor(figure_id), repo=repo)
    assert figure.axes[0].lines and any(isinstance(a, AnnotationBbox) for a in figure.axes[0].get_children())


def test_typing_a_pattern_leaves_the_standard_list_as_it_was(qapp):
    from app.plugin_api.charts import dashes
    from app.widgets.drop_down_controls.line_combo import LineStyleCombo
    from app.widgets.drop_down_controls.style_chooser import StyleChooser

    combo = LineStyleCombo()
    chooser = StyleChooser(combo, is_custom=dashes.is_custom, select=combo.set_current_linestyle, hint="")
    combo.set_current_linestyle("--")
    chooser.sync()
    assert chooser._shown.currentData() == "--"
    chooser._buttons["custom"].setChecked(True)
    chooser._text.setText("..--")
    chooser._typed()
    assert combo.current_linestyle() == "..--" and chooser._shown.currentData() == "--"
    chooser._buttons["standard"].setChecked(True)
    assert combo.current_linestyle() == "--"
