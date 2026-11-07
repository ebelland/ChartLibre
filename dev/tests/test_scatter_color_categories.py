"""A colour column of text that is not colour names is a set of categories."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from app.charts.scatter import ScatterAxisRenderer
from app.data.series_frame import SeriesFrame
from app.plugin_api.charts.base import SeriesData


def _draw(colors: list[str]):
    ax = Figure().add_subplot()
    frame = SeriesFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "color": colors})
    ScatterAxisRenderer().render_axis(ax, [SeriesData(name="s", df=frame, style={})], {})
    return ax.collections[0].get_facecolors()


def test_text_labels_become_one_colour_per_category() -> None:
    faces = _draw(["major", "minor", "major"])
    assert (faces[0] == faces[2]).all() and not (faces[0] == faces[1]).all()


def test_real_colour_names_are_still_used_as_given() -> None:
    faces = _draw(["red", "red", "blue"])
    assert tuple(faces[0][:3]) == (1.0, 0.0, 0.0)
