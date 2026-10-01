"""Outlier "Colour", drawn the way the chart panel draws it.

test_outlier_colour.py feeds the renderer a hand-built frame; the chart
reads the column through SqliteRepo.series_frame, where it is a pandas
string array, and the scatter renderer failed on it - the whole axis came
out empty. This goes through render_figure_from_descriptor.
"""
from __future__ import annotations

from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import app.series_operations.outlier_dialog as outlier_module  # noqa: E402
from app.analysis.outliers import OUTLIER_ZSCORE  # noqa: E402
from app.charts.render_figure import render_figure_from_descriptor  # noqa: E402
from app.data.sqlite_repo import SqliteRepo  # noqa: E402
from app.series_operations.outlier_dialog import ACTION_COLOUR, SeriesOutlierDialog  # noqa: E402
from app.utils.dialog_state import clear_state  # noqa: E402

RED = "#d62728"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    clear_state("SeriesOutlierDialog")
    monkeypatch.setattr(outlier_module, "show_message", lambda *_a, **_k: None)
    yield
    clear_state("SeriesOutlierDialog")


def _coloured_figure(repo: SqliteRepo, chart_type: str) -> Figure:
    values = np.random.default_rng(1).normal(10.0, 0.5, 60)
    values[[10, 40]] = [60.0, -40.0]
    repo.import_dataframe(pd.DataFrame({"n": np.arange(60.0), "v": values}), table_name="t", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type=chart_type, title="s", x_label="n", y_label="v", options={},
    ))
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig", sql_query="SELECT n AS x, v AS y FROM t",
        roles={"x": "x", "y": "y"}, style={},
    )
    dialog = SeriesOutlierDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    dialog.model_combo.setCurrentText(OUTLIER_ZSCORE)
    dialog._parameter_form_spec.set_values({"action": ACTION_COLOUR, "colour": RED})
    dialog.ok()
    dialog.close()

    figure = Figure()
    descriptor = repo.load_figure_descriptor(figure_id)
    assert descriptor is not None
    render_figure_from_descriptor(figure=figure, descriptor=descriptor, repo=repo)
    return figure


def _point_colours(figure: Figure) -> list[str]:
    return [to_hex(colour) for collection in figure.axes[0].collections for colour in collection.get_facecolors()]


def test_a_scatter_draws_the_outliers_in_the_colour_and_the_rest_as_before(qapp, repo: SqliteRepo) -> None:
    colours = _point_colours(_coloured_figure(repo, "Scatter Plot"))
    assert len(colours) == 60
    assert colours.count(RED) == 2
    assert len(set(colours)) == 2  # the series' own colour for the other 58


def test_a_time_series_keeps_its_line_and_marks_the_outliers(qapp, repo: SqliteRepo) -> None:
    figure = _coloured_figure(repo, "Time Series")
    ax = figure.axes[0]
    assert ax.lines and to_hex(ax.lines[0].get_color()) != RED
    assert _point_colours(figure) == [RED, RED]
