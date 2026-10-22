"""Image Plot: a table as a matrix, or a file (G-16)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from app.charts.image_plot import ImagePlotAxisRenderer
from app.plugin_api.charts.base import SeriesData


def _draw(frame: pd.DataFrame, options: dict | None = None) -> Figure:
    figure = Figure()
    ax = figure.add_subplot()
    ImagePlotAxisRenderer().render_axis(ax, [SeriesData(name="m", df=frame, style={})], options or {})
    return figure


def test_each_numeric_column_is_a_column_and_the_label_names_the_rows() -> None:
    frame = pd.DataFrame({"label": ["r1", "r2", "r3"], "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    ax = _draw(frame).axes[0]
    (image,) = ax.images
    assert np.array(image.get_array()).shape == (3, 2)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    # first row at the top: the last y position is the first row
    assert [t.get_text() for t in ax.get_yticklabels()] == ["r3", "r2", "r1"]
    assert np.array(image.get_array())[-1].tolist() == [1.0, 4.0]
    assert len(_draw(frame).axes) == 2  # the colorbar


def test_values_written_into_the_cells() -> None:
    ax = _draw(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), {"annotate": True, "annotate_format": ".1f"}).axes[0]
    assert sorted(t.get_text() for t in ax.texts) == ["1.0", "2.0", "3.0", "4.0"]


def test_an_image_file_is_shown_with_square_pixels_and_no_colorbar(tmp_path: Path) -> None:
    import matplotlib.image as mpimg

    path = tmp_path / "picture.png"
    mpimg.imsave(path, np.random.default_rng(0).random((6, 9, 3)))
    figure = _draw(pd.DataFrame({"a": [1.0]}), {"image_file": str(path)})
    assert len(figure.axes) == 1
    assert figure.axes[0].images[0].get_array().shape[:2] == (6, 9)
    assert figure.axes[0].get_aspect() == 1.0


def test_a_missing_file_draws_nothing_and_does_not_raise() -> None:
    figure = _draw(pd.DataFrame({"a": [1.0]}), {"image_file": "/no/such/file.png"})
    assert not figure.axes[0].images
