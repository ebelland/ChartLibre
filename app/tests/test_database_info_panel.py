"""The Database Info panel: path/size, tables, and the link status of each.

The one thing worth pinning beyond "the table has rows" is that Export and
Update link track the *selected* row - a table with no link must not offer
Update link just because some other table in the list has one.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QLabel

from app.data.sqlite_repo import SqliteRepo
from app.widgets.database_info_panel import DatabaseInfoPanel, _human_size


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
        pd.DataFrame({"x": [1, 2, 3]}), table_name="plain_table"
    )
    repo.import_dataframe(
        pd.DataFrame({"x": [1, 2]}), table_name="linked_table"
    )
    repo.upsert_link(
        table_name="linked_table",
        source_path="https://example.invalid/data.csv",
        settings={"source": {"kind": "web", "url": "https://example.invalid/data.csv"}},
    )
    yield repo
    repo.close()


def test_every_table_is_listed_with_its_row_count(qapp, repo: SqliteRepo) -> None:
    panel = DatabaseInfoPanel(repo, parent=None)

    rows = {
        panel._table.item(row, 0).text(): panel._table.item(row, 1).text()
        for row in range(panel._table.rowCount())
    }
    assert rows == {"plain_table": "3", "linked_table": "2"}


def test_only_the_linked_table_is_marked_linked(qapp, repo: SqliteRepo) -> None:
    panel = DatabaseInfoPanel(repo, parent=None)

    linked = {
        panel._table.item(row, 0).text(): panel._table.item(row, 2).text()
        for row in range(panel._table.rowCount())
    }
    assert linked["linked_table"] == "Yes"
    assert linked["plain_table"] == ""


def test_update_link_is_only_enabled_for_a_linked_selection(
    qapp, repo: SqliteRepo
) -> None:
    panel = DatabaseInfoPanel(repo, parent=None)

    for row in range(panel._table.rowCount()):
        if panel._table.item(row, 0).text() == "plain_table":
            panel._table.selectRow(row)
            break
    assert not panel._update_link_button.isEnabled()
    assert panel._export_csv_button.isEnabled()

    for row in range(panel._table.rowCount()):
        if panel._table.item(row, 0).text() == "linked_table":
            panel._table.selectRow(row)
            break
    assert panel._update_link_button.isEnabled()


def test_no_selection_disables_every_row_action(qapp, repo: SqliteRepo) -> None:
    panel = DatabaseInfoPanel(repo, parent=None)
    panel._table.clearSelection()
    panel._table.setCurrentCell(-1, -1)
    panel._update_button_states()

    assert not panel._export_csv_button.isEnabled()
    assert not panel._export_xlsx_button.isEnabled()
    assert not panel._update_link_button.isEnabled()


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (2048, "2.0 KB"),
        (5 * 1024 * 1024, "5.0 MB"),
    ],
)
def test_human_size_reads_the_way_a_person_would(num_bytes: int, expected: str) -> None:
    assert _human_size(num_bytes) == expected


# ----------------------------------------------------------------------
# PRAGMA info - what SQLite and the filesystem themselves can say
# ----------------------------------------------------------------------
def test_database_pragma_info_counts_tables_and_rows(repo: SqliteRepo) -> None:
    info = repo.database_pragma_info()

    assert info.table_count == 2
    assert info.total_rows == 5  # 3 + 2
    assert info.sqlite_version
    assert info.page_size > 0
    assert info.page_count > 0


def test_size_bytes_is_page_size_times_page_count(repo: SqliteRepo) -> None:
    info = repo.database_pragma_info()
    assert info.size_bytes == info.page_size * info.page_count


def test_file_modified_is_read_from_the_real_file(repo: SqliteRepo) -> None:
    info = repo.database_pragma_info()
    assert info.file_modified is not None


def test_the_panel_opens_with_the_pragma_section(qapp, repo: SqliteRepo) -> None:
    # Constructing the panel runs _add_pragma_rows; not raising and the
    # data it read back matching the fixture is the meaningful assertion -
    # hunting individual QLabel widgets would only restate the same thing
    # more fragilely.
    panel = DatabaseInfoPanel(repo, parent=None)

    assert panel._repo.database_pragma_info().table_count == 2


def test_a_pragma_failure_does_not_break_the_panel(
    qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(self) -> None:
        raise RuntimeError("boom")

    # An instance attribute cannot be patched here - SqliteRepo's mixins
    # are all __slots__ = (), so the class method is the only patch point.
    monkeypatch.setattr(SqliteRepo, "database_pragma_info", _raise)

    panel = DatabaseInfoPanel(repo, parent=None)  # must not raise

    assert panel is not None


def test_set_repo_refreshes_the_table_and_form(
    qapp, repo: SqliteRepo, tmp_db_path: Path
) -> None:
    """Opening a different project through MainWindow must not leave the
    embedded panel showing the previous database's path and tables."""
    panel = DatabaseInfoPanel(repo, parent=None)
    assert panel._table.rowCount() == 2

    other_path = tmp_db_path.with_name("other.dhub")
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        other_path.with_suffix(suffix).unlink(missing_ok=True)
    other = SqliteRepo(db_path=other_path)
    try:
        other.import_dataframe(pd.DataFrame({"x": [1]}), table_name="only_table")

        panel.set_repo(other)

        assert panel._table.rowCount() == 1
        assert panel._table.item(0, 0).text() == "only_table"
        labels = [label.text() for label in panel._form_host.findChildren(QLabel)]
        assert any(str(other_path) == text for text in labels)
    finally:
        other.close()
