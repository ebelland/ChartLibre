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




# ----------------------------------------------------------------------
# Clipboard transposed
# ----------------------------------------------------------------------
def test_a_table_copied_one_series_per_row_is_read_transposed(dialog) -> None:
    QApplication.clipboard().setText("t\t0\t1\t2\nspeed\t5\t7\t9\n")
    dialog._on_load_clipboard(transposed=True)
    frame = dialog._df
    assert frame is not None
    assert list(frame.columns) == ["t", "speed"]
    assert frame["t"].tolist() == [0, 1, 2]
    assert frame["speed"].tolist() == [5, 7, 9]  # numbers again, not text


def test_changing_an_option_after_a_transposed_paste_keeps_it_transposed(dialog) -> None:
    QApplication.clipboard().setText("a,1,2\nb,3,4\n")
    dialog._on_load_clipboard(transposed=True)
    dialog._has_header.setChecked(False)
    dialog._refresh_preview()
    frame = dialog._df
    assert frame is not None and frame.shape == (3, 2)  # a/b is now data, three rows


def test_an_ordinary_paste_after_a_transposed_one_is_not_transposed(dialog) -> None:
    QApplication.clipboard().setText("x\t1\t2\ny\t3\t4\n")
    dialog._on_load_clipboard(transposed=True)
    QApplication.clipboard().setText("x\ty\n1\t3\n2\t4\n")
    dialog._on_load_clipboard()
    frame = dialog._df
    assert frame is not None and list(frame.columns) == ["x", "y"] and frame.shape == (2, 2)


def test_transposing_text_keeps_quoted_fields_and_pads_short_rows() -> None:
    from app.utils.data_sources import transpose_delimited_text

    text = 'name,"Smith, J",Lee\nage,40\n'
    assert transpose_delimited_text(text, ",") == "name\tage\nSmith, J\t40\nLee\t\n"
    assert transpose_delimited_text("   ") == ""


def test_web_sources_json_sits_beside_config_and_is_read_afresh(tmp_path, monkeypatch) -> None:
    """The user edits it by hand: an edit shows at once, a broken file breaks nothing."""
    import json

    from app.utils import data_sources

    root = data_sources.WEB_SOURCES_PATH.parent
    assert data_sources.WEB_SOURCES_PATH.name == "web_sources.json" and (root / "config.json").is_file()
    assert data_sources._bundled_web_data_sources(), "the shipped catalogue is empty"

    path = tmp_path / "web_sources.json"
    monkeypatch.setattr(data_sources, "WEB_SOURCES_PATH", path)
    path.write_text(json.dumps([{"name": "Mine", "url": "https://example.org/a.csv"}]), encoding="utf-8")
    first = data_sources._bundled_web_data_sources()
    assert [(s.name, s.category) for s in first] == [("Mine", "Other")]
    path.write_text(json.dumps([{"name": "Mine", "url": "https://example.org/a.csv"},
                                {"name": "Second", "url": "https://example.org/b.csv", "category": "Data"},
                                {"name": "no url"}]), encoding="utf-8")
    assert [s.name for s in data_sources._bundled_web_data_sources()] == ["Mine", "Second"]
    path.write_text('[{"name": "Mine",}]', encoding="utf-8")  # a comma too many
    assert data_sources._bundled_web_data_sources() == ()
