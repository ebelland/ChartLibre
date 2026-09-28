"""Tests for applying a multi-result series operation.

A spectral analysis of five series produced one series on the chart.  The cause
was not in the loop that writes the results - that was always correct - but in
what the loop ended with: ``optimize_db()``, whose VACUUM cannot run inside a
transaction.  The OperationalError escaped through ``_run_operation``, which
rolled the savepoint back, so everything after the first flush to disk was
undone.

These tests pin the two halves of the fix without needing a dialog: the
repository refuses to VACUUM inside a transaction instead of raising, and a
batch of results all survive a transaction that also touches the tables they
are written to.
"""
from __future__ import annotations

from collections.abc import Iterator

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> Iterator[SqliteRepo]:
    repo = SqliteRepo(db_path=tmp_db_path)
    yield repo
    repo.close()


def _frame(size: int = 32) -> pd.DataFrame:
    x = np.arange(size, dtype=float)
    return pd.DataFrame({"frequency": x, "power": np.sin(x)})


# ----------------------------------------------------------------------
# optimize_db inside a transaction
# ----------------------------------------------------------------------






# ----------------------------------------------------------------------
# The batch survives
# ----------------------------------------------------------------------


def test_every_result_of_a_batch_is_written(repo: SqliteRepo) -> None:
    """Five results in, five tables out - the spectral bug in miniature.

    The loop ends with the optimize call that used to raise; if it raised
    again, the exception would reach here.
    """
    for index in range(5):
        repo.import_dataframe(
            _frame(), table_name=f"result_{index}", normalize_columns=False
        )
    repo.optimize_db()

    tables = set(repo.list_table_names())
    assert {f"result_{index}" for index in range(5)} <= tables






# ----------------------------------------------------------------------
# The spectral results get their own chart
# ----------------------------------------------------------------------












# ----------------------------------------------------------------------
# Applying an operation is one undoable step (todo.txt P2-11)
# ----------------------------------------------------------------------
def test_applying_an_operation_can_be_taken_back(qapp, repo, tmp_db_path) -> None:
    """The result table and the series descriptor that draws it go back
    together: the snapshot is opened before the target axis is resolved -
    which may create it - and completed with the result table's name once
    that is known."""
    import numpy as np
    import pandas as pd

    from app.series_operations.roots_dialog import SeriesRootsDialog

    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(60.0), "y": np.sin(np.arange(60) / 5.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="w", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )
    tables_before = sorted(repo.list_user_tables()["Table"])
    series_before = len(repo.get_series(axis_id))

    dialog = SeriesRootsDialog(repo=repo, figure_id=figure_id)
    dialog._run_operation(commit=True)

    assert len(repo.get_series(axis_id)) == series_before + 1
    assert [entry.label for entry in repo.undo_entries()] == ["Apply Roots"]

    repo.undo_last()

    assert sorted(repo.list_user_tables()["Table"]) == tables_before
    assert len(repo.get_series(axis_id)) == series_before
    repo.undo_store.discard_file()


def test_previewing_an_operation_records_nothing(qapp, repo, tmp_db_path) -> None:
    """A preview is rolled back by its own savepoint. Recording it would
    fill the undo stack with steps that never happened."""
    import numpy as np
    import pandas as pd

    from app.series_operations.roots_dialog import SeriesRootsDialog

    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(60.0), "y": np.sin(np.arange(60) / 5.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="w", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )

    dialog = SeriesRootsDialog(repo=repo, figure_id=figure_id)
    dialog._run_operation(commit=False)

    assert repo.undo_entries() == []
    dialog.cancel_operation_changes(refresh=False)
    repo.undo_store.discard_file()
