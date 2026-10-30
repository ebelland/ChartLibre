"""Clustering's report and its figure: the scores, a row per cluster, the k tried; the clusters on two principal
components, the silhouette and the means by cluster - and a preview's figure gone with Close."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data_operations.cluster_dialog import SeriesClusterDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _fresh_dialog_state() -> Iterator[None]:
    clear_state("SeriesClusterDialog")
    yield
    clear_state("SeriesClusterDialog")


@pytest.fixture
def blobs(repo: SqliteRepo) -> int:
    rng = np.random.default_rng(7)
    points = np.vstack([rng.normal(centre, 0.2, size=(40, 2)) for centre in ((0, 0), (5, 5), (0, 6))])
    repo.import_dataframe(pd.DataFrame({"a": points[:, 0], "b": points[:, 1]}), table_name="blobs",
                          normalize_columns=False)
    figure = int(repo.create_figure_descriptor(name="Blobs"))
    axis = int(repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Scatter Plot", title="blobs",
                                           x_label="a", y_label="b", options={}))
    repo.create_series_descriptor(axis_id=axis, series_index=0, name="Blobs", sql_query="SELECT a AS x, b AS y FROM blobs",
                                  roles={"x": "x", "y": "y"}, style={})
    return figure


def _dialog(repo: SqliteRepo, figure: int, **values: object) -> SeriesClusterDialog:
    dialog = SeriesClusterDialog(repo=repo, figure_id=figure, parent=None)
    dialog.series_selector.reload(select_all_series=True)
    dialog._parameter_form_spec.set_values({"clusters": 3, **values})
    return dialog


def _figures(repo: SqliteRepo) -> set[int]:
    return {int(figure_id) for figure_id, _name in repo.load_figures_from_db()}


def test_the_report_gives_the_scores_the_clusters_and_what_next(qapp, repo: SqliteRepo, blobs: int) -> None:
    dialog = _dialog(repo, blobs)
    try:
        np.random.seed(0)
        (result,) = dialog.compute_results()
        html = dialog.format_results([result])
        for words in ("Summary", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "Clusters", "Mean silhouette",
                      "How to read this", "What next?", "Do the clusters differ in a?"):
            assert words in html, words
        assert "Choosing k" not in html  # one k asked for: nothing was chosen
    finally:
        dialog.close()


def test_ok_draws_the_report_figure_and_its_charts_render(qapp, repo: SqliteRepo, blobs: int) -> None:
    from matplotlib.figure import Figure

    from app.plugin_api.charts.render_figure import render_figure_from_descriptor

    dialog = _dialog(repo, blobs, automatic_k=True, k_max=5)
    try:
        np.random.seed(0)
        dialog.ok()
        (figure_id,) = dialog.created_figure_ids
        figure = repo.load_figure_descriptor(figure_id)
        assert [axis.title for axis in figure.axes] == [
            "Clusters on two principal components", "Silhouette by cluster", "Cluster means", "Choosing k"]
        assert [series.name for series in figure.axes[2].series] == ["a", "b"]  # the columns, not "x" and "y"
        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert len(drawn.axes) >= 4
        projection = repo.query_df(figure.axes[0].series[0].sql_query)
        assert len(projection) == 120 and set(projection["color"]) == {1, 2, 3}
    finally:
        dialog.close()


def test_a_chart_cleared_in_the_charts_tab_is_not_drawn(qapp, repo: SqliteRepo, blobs: int) -> None:
    dialog = _dialog(repo, blobs)
    dialog.chart_form.set_values({"chart_projection": False, "chart_means": False})
    try:
        np.random.seed(0)
        dialog.ok()
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.title for axis in figure.axes] == ["Silhouette by cluster"]
    finally:
        dialog.close()


def test_a_previews_figure_goes_with_close(qapp, repo: SqliteRepo, blobs: int) -> None:
    before = _figures(repo)
    dialog = _dialog(repo, blobs)
    np.random.seed(0)
    assert dialog.preview()
    assert len(_figures(repo)) == len(before) + 1
    dialog.close()
    assert _figures(repo) == before
    assert "ClusterId" not in repo.get_columns("blobs")


def test_the_results_pane_explains_each_series_once_and_logs_no_error(qapp, repo: SqliteRepo, blobs: int) -> None:
    import logging

    from app.logs.logger import applogger

    class Errors(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.ERROR)
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    errors = Errors()
    applogger.addHandler(errors)
    dialog = _dialog(repo, blobs)
    try:
        np.random.seed(0)
        (result,) = dialog.compute_results()
        assert dialog.results_report([result]).count("How to read this") == 1
        assert errors.messages == []
    finally:
        applogger.removeHandler(errors)
        dialog.close()
