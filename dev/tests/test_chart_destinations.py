"""Where a report's charts go: Same axis draws the first chart on the axis of its kind, Same figure adds
every chart as new axes to the current figure, New figure makes a figure of them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data_operations.distribution_dialog import DistributionDialog
from app.plugin_api.operation_shell import OperationShell


@pytest.fixture
def histogram_figure(repo: SqliteRepo) -> int:
    """A figure whose one axis is a histogram, the kind the Histogram chart draws on."""
    repo.import_dataframe(pd.DataFrame({"x": np.arange(50.0), "y": np.random.default_rng(4).normal(size=50)}),
                          table_name="d", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Data"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Histogram",
                                              title="d", x_label="", y_label="", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Before", sql_query='SELECT y AS value FROM d',
                                  roles={"value": "value"}, style={})
    return figure_id


def _draw(qapp, repo: SqliteRepo, figure_id: int, destination: str) -> DistributionDialog:
    dialog = DistributionDialog(repo=repo, figure_id=figure_id)
    dialog.draw_on = lambda: destination  # the choice the Charts tab would make
    dialog.current_figure_id = figure_id
    dialog.parameter_values = lambda: {"distribution": "norm"}
    dialog.ok()
    return dialog


def test_same_axis_adds_the_first_chart_to_its_axis_and_draws_no_other(qapp, repo: SqliteRepo,
                                                                    histogram_figure: int) -> None:
    dialog = _draw(qapp, repo, histogram_figure, OperationShell.DEST_SAME_AXIS)
    try:
        figure = repo.load_figure_descriptor(histogram_figure)
        assert [axis.name for axis in figure.axes] == ["Histogram"]  # no Q-Q plot beside it
        assert len(figure.axes[0].series) == 2
        assert dialog.created_figure_ids == [histogram_figure]
    finally:
        dialog.close()


def test_same_axis_without_an_axis_of_that_kind_adds_only_the_first_chart(qapp, repo: SqliteRepo) -> None:
    repo.import_dataframe(pd.DataFrame({"x": np.arange(50.0), "y": np.random.default_rng(5).normal(size=50)}),
                          table_name="e", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Scatter"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="e", x_label="", y_label="", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Points",
                                  sql_query='SELECT x, y FROM e', roles={"x": "x", "y": "y"}, style={})
    dialog = DistributionDialog(repo=repo, figure_id=figure_id)
    dialog.draw_on = lambda: OperationShell.DEST_SAME_AXIS
    dialog.current_figure_id = figure_id
    dialog.parameter_values = lambda: {"distribution": "norm"}
    try:
        dialog.ok()
        axes = [axis.name for axis in repo.load_figure_descriptor(figure_id).axes]
        assert axes == ["Scatter Plot", "Histogram"]  # the histogram is new; the Q-Q plot is not drawn
    finally:
        dialog.close()


def test_same_figure_adds_every_chart_as_new_axes(qapp, repo: SqliteRepo, histogram_figure: int) -> None:
    dialog = _draw(qapp, repo, histogram_figure, OperationShell.DEST_NEW_AXIS)
    try:
        figure = repo.load_figure_descriptor(histogram_figure)
        assert [axis.name for axis in figure.axes] == ["Histogram", "Histogram", "Q-Q Plot"]
        assert dialog.created_figure_ids == [histogram_figure]
    finally:
        dialog.close()


def test_new_figure_makes_a_figure_of_every_chart(qapp, repo: SqliteRepo, histogram_figure: int) -> None:
    dialog = _draw(qapp, repo, histogram_figure, OperationShell.DEST_NEW_FIGURE)
    try:
        assert len(dialog.created_figure_ids) == 1 and dialog.created_figure_ids[0] != histogram_figure
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Histogram", "Q-Q Plot"]
        assert len(repo.load_figure_descriptor(histogram_figure).axes) == 1
    finally:
        dialog.close()


def test_same_axis_hides_every_chart_but_the_first_in_the_charts_tab() -> None:
    from app.plugin_api.data_operations.parameter_spec import BoolParam
    from app.plugin_api.data_operations.roles_base import _hidden_by_same_axis

    charts = (BoolParam("Histogram", "Histogram", default_value=True, tab="charts"),
              BoolParam("Q-Q Plot", "Q-Q Plot", default_value=True, tab="charts"))
    first, second = (_hidden_by_same_axis(spec, "Histogram", OperationShell.DEST_SAME_AXIS) for spec in charts)
    same_axis = {"_destination": OperationShell.DEST_SAME_AXIS}
    new_figure = {"_destination": OperationShell.DEST_NEW_FIGURE}
    assert first.is_visible(same_axis) and not second.is_visible(same_axis)
    assert first.is_visible(new_figure) and second.is_visible(new_figure)
