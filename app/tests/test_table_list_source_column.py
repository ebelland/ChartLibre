"""The table list's Source column (formerly "File") and its "Edit…" menu
entry for a saved query.

A table, a database-linked table, a file/web-linked table and a saved
query each show something different there - the file/web case already
worked before this, so the new ground covered is the database link's
"connection -> table" split and the query's own SQL - and only a saved
query offers "Edit…", which opens it in the Query Builder rather than
promising to edit a table's rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets import table_list as table_list_module
from app.widgets.table_list import TableListPanel


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    repo = SqliteRepo(db_path=tmp_db_path)
    repo.import_dataframe(
        pd.DataFrame({"x": [1.0, 2.0]}), table_name="plain", normalize_columns=False
    )
    repo.import_dataframe(
        pd.DataFrame({"x": [1.0, 2.0]}), table_name="from_file", normalize_columns=False
    )
    repo.upsert_link(
        table_name="from_file",
        source_path="/Users/someone/Desktop/measurements.csv",
        settings={"source": {"kind": "file", "path": "/Users/someone/Desktop/measurements.csv"}},
    )
    repo.import_dataframe(
        pd.DataFrame({"x": [1.0, 2.0]}), table_name="from_db", normalize_columns=False
    )
    repo.upsert_link(
        table_name="from_db",
        source_path="MyPostgres#customers",
        settings={"source": {"kind": "postgres", "table": "customers"}},
    )
    repo.save_query(
        "long_query",
        "SELECT x FROM plain WHERE x > 0 ORDER BY x DESC LIMIT 10 -- padding to push this past sixty characters",
    )
    yield repo
    repo.close()


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    built = TableListPanel(repo=repo, parent=None)
    built.reload()
    return built


# ----------------------------------------------------------------------
# _source_display_text - pure, no widgets
# ----------------------------------------------------------------------
def test_a_plain_table_shows_nothing() -> None:
    assert TableListPanel._source_display_text(False, False, "") == ""


def test_a_file_link_shows_the_filename() -> None:
    text = TableListPanel._source_display_text(
        False, True, "/Users/someone/Desktop/measurements.csv"
    )
    assert text == "measurements.csv"


def test_a_database_link_shows_connection_arrow_table() -> None:
    text = TableListPanel._source_display_text(False, True, "MyPostgres#customers")
    assert text == "MyPostgres → customers"


def test_a_file_path_that_happens_to_contain_a_hash_is_not_mistaken_for_a_db_link() -> None:
    text = TableListPanel._source_display_text(False, True, "/data/run#1/results.csv")
    assert text == "results.csv"


def test_a_query_shows_its_own_sql_collapsed_and_truncated() -> None:
    sql = "SELECT x\nFROM plain\nWHERE   x > 0"
    text = TableListPanel._source_display_text(True, False, sql)
    assert text == "SELECT x FROM plain WHERE x > 0"


def test_a_long_query_is_truncated_with_an_ellipsis() -> None:
    sql = "SELECT " + ", ".join(f"col{i}" for i in range(30))
    text = TableListPanel._source_display_text(True, False, sql)
    assert len(text) == TableListPanel._SOURCE_TEXT_MAX
    assert text.endswith("…")


# ----------------------------------------------------------------------
# The panel: header label, cell content, tooltip, and the Edit… action
# ----------------------------------------------------------------------
def test_the_source_header_replaces_file(panel: TableListPanel) -> None:
    assert panel._model.horizontalHeaderItem(panel.COL_FILE).text() == "Source"


def test_the_query_row_shows_truncated_sql_with_the_full_text_as_tooltip(
    panel: TableListPanel,
) -> None:
    row = _row_for(panel, "long_query")
    item = panel._model.item(row, panel.COL_FILE)
    assert item.text().endswith("…")
    assert "padding to push this past sixty characters" in item.toolTip()


def test_the_database_linked_row_shows_connection_and_table(panel: TableListPanel) -> None:
    row = _row_for(panel, "from_db")
    item = panel._model.item(row, panel.COL_FILE)
    assert item.text() == "MyPostgres → customers"


def test_edit_opens_the_query_builder_on_that_query(
    panel: TableListPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[tuple[object, str | None]] = []

    class _FakeQueryBuilderDialog:
        def __init__(self, repo, *, query_name=None, parent=None) -> None:
            opened.append((repo, query_name))

        def exec(self) -> None:
            return None

    monkeypatch.setattr(table_list_module, "QueryBuilderDialog", _FakeQueryBuilderDialog)

    row = _row_for(panel, "long_query")
    panel._view.selectionModel().clearSelection()
    panel._view.selectRow(row)

    panel._edit_selected_query()

    assert opened == [(panel._repo, "long_query")]


def test_edit_does_nothing_for_a_plain_table(
    panel: TableListPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[object] = []
    monkeypatch.setattr(
        table_list_module, "QueryBuilderDialog",
        lambda *a, **k: opened.append(1),
    )

    row = _row_for(panel, "plain")
    panel._view.selectionModel().clearSelection()
    panel._view.selectRow(row)

    panel._edit_selected_query()  # must not raise, must not open anything

    assert opened == []


def _row_for(panel: TableListPanel, table_name: str) -> int:
    for row in range(panel._model.rowCount()):
        item = panel._model.item(row, panel.COL_TABLE)
        if item is not None and item.data(panel.ROLE_TABLE_NAME) == table_name:
            return row
    raise AssertionError(f"{table_name!r} not found in the table list")
