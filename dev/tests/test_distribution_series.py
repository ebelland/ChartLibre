"""Distribution on the data engine: one report and one figure per series, and no tables written."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data_operations.distribution_dialog import DistributionDialog


@pytest.fixture
def lognormal_figure(repo: SqliteRepo) -> int:
    values = np.random.default_rng(1).lognormal(1.0, 0.6, 400)
    repo.import_dataframe(pd.DataFrame({"x": np.arange(400.0), "y": values}), table_name="ln",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Lognormal"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="ln", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Sample", sql_query='SELECT x, y FROM "ln"',
                                  roles={"x": "x", "y": "y"}, style={})
    return figure_id


def test_each_series_gets_its_report_and_no_table_is_written(qapp, repo: SqliteRepo, lognormal_figure: int) -> None:
    dialog = DistributionDialog(repo=repo, figure_id=lognormal_figure)
    try:
        results = dialog.compute_results()
        assert [result.source_name for result in results] == ["Sample"]
        assert results[0].report.n == 400
        assert len(results[0].to_df()) == len(results[0].report.quantiles)
        html = dialog.format_results(results)
        assert "Quantiles" in html and "Moments" in html and "Normality" in html
        assert set(repo.list_user_tables()["Table"]) == {"ln"}
    finally:
        dialog.close()


def test_best_fit_picks_the_family_of_the_sample(qapp, repo: SqliteRepo, lognormal_figure: int) -> None:
    dialog = DistributionDialog(repo=repo, figure_id=lognormal_figure)
    try:
        dialog.parameter_values = lambda: {"distribution": "best"}
        (result,) = dialog.compute_results()
        assert result.fit is not None and result.fit.name == "lognorm"
    finally:
        dialog.close()


def test_ok_draws_a_histogram_beside_a_q_q_plot(qapp, repo: SqliteRepo, lognormal_figure: int) -> None:
    dialog = DistributionDialog(repo=repo, figure_id=lognormal_figure)
    try:
        dialog.parameter_values = lambda: {"distribution": "norm"}
        dialog.ok()
        assert dialog.created_figure_ids
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Histogram", "Q-Q Plot"]
    finally:
        dialog.close()
