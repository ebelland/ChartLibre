"""A Geometry result over a Hide-filtered series stays valid SQL, and its report says what it did.

Geometry writes "WITH source AS (<the series' SQL>) SELECT ... FROM source".
When the series' SQL carried the Hide filter, the filter's own check saw a
WHERE inside the CTE, took it for the outer query's and appended
' AND "Hide" = 0' after "FROM source": not SQL, so the new series failed
and was never drawn, and the report showed only the query.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app.charts.render_figure import render_figure_from_descriptor  # noqa: E402
from app.data.sqlite_repo import SqliteRepo  # noqa: E402
from app.series_operations.geometry_dialog import ROTATE, SeriesGeometryDialog  # noqa: E402
from app.utils.dialog_state import clear_state  # noqa: E402

SOURCE_SQL = 'SELECT "salary_eur" AS "x", "bonus_eur" AS "y" FROM "pay" WHERE "Hide" = 0'


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    clear_state("SeriesGeometryDialog")
    yield
    clear_state("SeriesGeometryDialog")


def _figure(repo: SqliteRepo) -> int:
    rng = np.random.default_rng(0)
    repo.import_dataframe(
        pd.DataFrame({"salary_eur": rng.normal(55000, 8000, 50), "bonus_eur": rng.normal(5000, 1200, 50)}),
        table_name="pay", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot", title="t", x_label="x", y_label="y", options={},
    ))
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="pay", sql_query=SOURCE_SQL, roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id


def test_a_rotated_hide_filtered_series_is_written_as_valid_sql_and_drawn(qapp, repo: SqliteRepo) -> None:
    figure_id = _figure(repo)
    dialog = SeriesGeometryDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    dialog.model_combo.setCurrentText(ROTATE)
    dialog._parameter_form_spec.set_values({"angle": 30.0})
    reports: list[str] = []
    dialog.results_published.connect(reports.append)
    assert dialog.apply()
    dialog.close()

    sqls = list(repo.query_df("SELECT sql_query FROM __series_descriptors__ ORDER BY id")["sql_query"])
    rotated = str(sqls[1])
    assert rotated.rstrip().endswith("FROM source")
    assert len(repo.query_df(rotated)) == 50

    figure = Figure()
    descriptor = repo.load_figure_descriptor(figure_id)
    assert descriptor is not None
    render_figure_from_descriptor(figure=figure, descriptor=descriptor, repo=repo)
    assert [len(collection.get_offsets()) for collection in figure.axes[0].collections] == [50, 50]

    [report] = reports
    assert "data:image/png;base64," in report
    assert "30°" in report and "Centre" in report
    assert "WITH source AS" in report


def test_the_hide_filter_reads_only_the_outer_query(repo: SqliteRepo) -> None:
    cte = f"WITH source AS ({SOURCE_SQL}) SELECT x + 1 AS x, y FROM source"
    assert not repo.is_table_backed_sql(cte)
    assert repo.sql_with_hide_filter(cte) == cte
    assert not repo.is_table_backed_sql("SELECT * FROM (SELECT 1 AS x)")
    running = "SELECT a.m AS x, (SELECT COUNT(*) FROM e b WHERE b.m <= a.m) AS y FROM e a ORDER BY a.m"
    assert repo.sql_with_hide_filter(running) == (
        'SELECT a.m AS x, (SELECT COUNT(*) FROM e b WHERE b.m <= a.m) AS y FROM e a WHERE "Hide" = 0 ORDER BY a.m'
    )


def test_a_project_broken_the_old_way_is_repaired_on_open(tmp_path: Path) -> None:
    tmp_db_path = tmp_path / "broken.dhub"
    repo = SqliteRepo(db_path=tmp_db_path)
    _figure(repo)
    broken = f"WITH source AS ({SOURCE_SQL}) SELECT x, y FROM source AND \"Hide\" = 0"
    assert repo._con is not None
    repo._con.execute("UPDATE __series_descriptors__ SET sql_query = ?", (broken,))
    repo._con.commit()
    repo.close()

    reopened = SqliteRepo(db_path=tmp_db_path)
    [sql] = list(reopened.query_df("SELECT sql_query FROM __series_descriptors__")["sql_query"])
    assert sql == f"WITH source AS ({SOURCE_SQL}) SELECT x, y FROM source"
    assert len(reopened.query_df(sql)) == 50
    reopened.close()
