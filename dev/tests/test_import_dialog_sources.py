"""The import dialog's two sources, and the rule that only one is current.

Opening a file and pasting are alternatives, not layers. The dialog kept the
file's path, title and sheet list when data was pasted over it, and then - on
the last line of the paste handler - called the preview refresh, which read
``self.file_name`` and nothing else. So pasting after opening a file put the
file's rows back under the clipboard's table name, and pasting again changed
nothing at all.

These tests are mostly about which source is current after each move, because
that is the state the bug lived in.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.import_data_dialog import ImportDataDialog


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text("region,units\nnorth,10\nsouth,20\n", encoding="utf-8")
    return path


@pytest.fixture
def dialog(qapp, tmp_db_path: Path):
    repo = SqliteRepo(db_path=tmp_db_path)
    built = ImportDataDialog(repo)
    yield built
    repo.close()


def _paste(dialog, text: str) -> None:
    QApplication.clipboard().setText(text)
    dialog._on_load_clipboard()


def _columns(dialog) -> list[str]:
    assert dialog._df is not None
    return [str(column) for column in dialog._df.columns]


# ----------------------------------------------------------------------
# Replacing one source with the other
# ----------------------------------------------------------------------
def test_pasting_over_an_opened_file_replaces_it(dialog, csv_file: Path) -> None:
    """The bug, in one test: the file's rows came back after the paste."""
    dialog.load_file(csv_file)
    assert _columns(dialog) == ["region", "units"]

    _paste(dialog, "alpha\tbeta\n1\t2\n")

    assert dialog._source_mode == "clipboard"
    assert _columns(dialog) == ["alpha", "beta"]


# ----------------------------------------------------------------------
# The options describe how to parse, not where it came from
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Refusing a paste leaves what is loaded alone
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Another database joins the same source-switching contract
# ----------------------------------------------------------------------
@pytest.fixture
def other_db(tmp_path: Path) -> Path:
    """Another .dhub-shaped file: one real table, and one that looks like
    this application's own bookkeeping - the picker must skip the second."""
    import sqlite3

    path = tmp_path / "other.dhub"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE readings (t REAL, v REAL)")
        conn.execute("INSERT INTO readings VALUES (1.0, 2.0), (2.0, 4.0)")
        conn.execute("CREATE TABLE __figure_descriptors__ (id INTEGER)")
        conn.commit()
    finally:
        conn.close()
    return path


def _pick_database(
    dialog, path: Path, monkeypatch: pytest.MonkeyPatch, *, table: str = "readings"
) -> None:
    """Stand in for the connect dialog, the way LoadDemoDialog.exec is
    stood in for elsewhere: picking a database is its own dialog now, not a
    bare file picker, since PostgreSQL and MySQL need host/user/password
    fields a file dialog has no room for."""
    from app.dialogs.connect_database_dialog import ConnectDatabaseDialog
    from app.utils.data_sources import DatabaseConnection

    def fake_exec(self: ConnectDatabaseDialog) -> bool:
        self.connection = DatabaseConnection(kind="sqlite", path=str(path))
        self.table = table
        return True

    monkeypatch.setattr(ConnectDatabaseDialog, "exec", fake_exec)
    dialog._on_import_database()


def test_picking_a_database_loads_its_first_table(
    dialog, other_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pick_database(dialog, other_db, monkeypatch)

    assert dialog._source_mode == "database"
    assert _columns(dialog) == ["t", "v"]
    assert dialog._table.text() == "readings"
    assert dialog._db_connection is not None


# ----------------------------------------------------------------------
# The web source: fetched into the same preview/import path as a file
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The left panel reads as sections, the same convention as the properties
# panels (Figure/Axis/Series) and the connect-database dialog.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Web sources: Add source / Delete source
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_user_web_sources(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A user_web_sources catalogue of its own, not the developer's own.

    Without this, add_user_web_source/remove_user_web_source read and write
    the real user.json - exactly the "test run rewrote a settings file"
    problem app.utils.config's own docstring warns about.
    """
    import app.utils.config as config

    stored: list[dict] = []
    monkeypatch.setattr(config, "get_user_web_sources", lambda: list(stored))

    def _set(entries: list[dict]) -> None:
        stored.clear()
        stored.extend(entries)

    monkeypatch.setattr(config, "set_user_web_sources", _set)
    return {"entries": stored}


