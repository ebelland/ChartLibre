"""A cluster preview is rolled back on Close and never reaches the undo history.

Restoring ClusterId after a preview used to drop the column through
``delete_table_column``, which records an undo snapshot - inside the preview
SAVEPOINT, where attaching the undo database is refused, so every Close and
every re-run logged an ERROR.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.cluster_dialog import SeriesClusterDialog
from app.utils.dialog_state import clear_state


class _Errors(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def errors() -> Iterator[_Errors]:
    # The app logger does not propagate, so caplog would see nothing.
    handler = _Errors()
    applogger.addHandler(handler)
    yield handler
    applogger.removeHandler(handler)


@pytest.fixture(autouse=True)
def _fresh_dialog_state() -> Iterator[None]:
    clear_state("SeriesClusterDialog")
    yield
    clear_state("SeriesClusterDialog")


def _figure(repo: SqliteRepo, *, cluster_id: bool) -> int:
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 12.0, 160)
    frame = pd.DataFrame({
        "t": t,
        "a": np.sin(t) + rng.normal(0.0, 0.05, t.size),
        "b": np.cos(0.7 * t) + rng.normal(0.0, 0.05, t.size),
    })
    if cluster_id:
        frame["ClusterId"] = 7
    repo.import_dataframe(frame, table_name="sig", normalize_columns=False)
    figure = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure, axis_index=0, chart_type="Scatter Plot",
        title="t", x_label="t", y_label="y", options={},
    ))
    for index, column in enumerate(("a", "b")):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=column,
            sql_query=f"SELECT t AS x, {column} AS y FROM sig",
            roles={"x": "x", "y": "y"}, style={},
        )
    return figure


def _preview(dialog: SeriesClusterDialog) -> None:
    np.random.seed(0)  # k-means starts from random centroids
    assert dialog.preview()
    assert not dialog.evaluating


@pytest.mark.parametrize("cluster_id", [False, True], ids=["new-column", "existing-column"])
def test_preview_rerun_and_close_log_no_error_and_record_no_undo(
    qapp, repo: SqliteRepo, errors: _Errors, cluster_id: bool
) -> None:
    figure_id = _figure(repo, cluster_id=cluster_id)
    columns_before = list(repo.get_columns("sig"))
    entries_before = len(repo.undo_entries())

    dialog = SeriesClusterDialog(repo=repo, figure_id=figure_id, parent=None)
    dialog.series_selector.reload(select_all_series=True)
    _preview(dialog)
    assert "ClusterId" in repo.get_columns("sig")
    _preview(dialog)  # re-running cancels the first preview
    dialog.close()

    assert errors.messages == []
    assert len(repo.undo_entries()) == entries_before
    assert list(repo.get_columns("sig")) == columns_before
    if cluster_id:
        assert set(repo.query_df('SELECT "ClusterId" FROM sig')["ClusterId"]) == {7}
