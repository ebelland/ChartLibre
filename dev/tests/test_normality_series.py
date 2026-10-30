"""Is it normal? on the data engine: one answer and one Q-Q figure per series, its "What next?" links on the figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data_operations.normality_dialog import NormalityDialog, source_table
from app.plugin_api.table_operations import next_steps


@pytest.fixture
def skewed_figure(repo: SqliteRepo) -> int:
    values = np.random.default_rng(3).lognormal(1.0, 0.8, 120)
    repo.import_dataframe(pd.DataFrame({"x": np.arange(120.0), "y": values}), table_name="skew",
                          normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Skew"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="skew", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Mass", sql_query='SELECT x, y FROM "skew"',
                                  roles={"x": "x", "y": "y"}, style={})
    return figure_id


def test_the_answer_is_a_report_with_its_links_and_no_table_is_written(qapp, repo: SqliteRepo,
                                                                      skewed_figure: int) -> None:
    dialog = NormalityDialog(repo=repo, figure_id=skewed_figure)
    try:
        (result,) = dialog.compute_results()
        assert result.source_name == "Mass" and result.answer.verdict == "no"
        html = dialog.format_results([result])
        assert "Mass (N = 120)" in html and "What next?" in html
        assert "Distribution of Mass" in html
        assert set(repo.list_user_tables()["Table"]) == {"skew"}
    finally:
        dialog.close()


def test_ok_draws_a_normal_q_q_plot_in_its_own_figure(qapp, repo: SqliteRepo, skewed_figure: int) -> None:
    dialog = NormalityDialog(repo=repo, figure_id=skewed_figure)
    try:
        dialog.ok()
        assert dialog.created_figure_ids
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Q-Q Plot"]
    finally:
        dialog.close()


def test_the_distribution_link_opens_on_the_same_figure(qapp, repo: SqliteRepo, skewed_figure: int) -> None:
    dialog = NormalityDialog(repo=repo, figure_id=skewed_figure)
    try:
        (result,) = dialog.compute_results()
        steps = dialog.next_steps(result)
        assert steps[0].operation == "Distribution" and steps[0].figure == skewed_figure
        parsed = next_steps.parse_step(next_steps.step_url(steps[0]))
        assert parsed is not None and parsed.figure == skewed_figure and parsed.table == ""
    finally:
        dialog.close()


@pytest.mark.parametrize(("query", "table"), [
    ('SELECT x, y FROM "skew"', "skew"),
    ('SELECT t AS x, a AS y FROM t ORDER BY x', "t"),
    ('SELECT x, y FROM "my ""odd"" table"', 'my "odd" table'),
    ('SELECT x, y FROM a JOIN b ON a.id = b.id WHERE a.x > 0 AND b.y IN (SELECT 1)', ""),
    ("", ""),
])
def test_source_table_names_the_table_a_series_reads(query: str, table: str) -> None:
    assert source_table(query) == table
