"""Tests that the repository creates its descriptor schema and hides system tables."""
from __future__ import annotations

import sqlite3

from app.data.sqlite_repo import SqliteRepo


def test_repo_creates_descriptor_schema(tmp_db_path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)

    # Trigger schema creation
    repo.query_df("SELECT 1")

    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {str(r[0]) for r in cur.fetchall()}

    # Verify expected columns (new persisted options_json)
    cols_fig = [r[1] for r in con.execute('PRAGMA table_info(__figure_descriptors__)').fetchall()]
    assert 'options_json' in cols_fig
    cols_ax = [r[1] for r in con.execute('PRAGMA table_info(__axis_descriptors__)').fetchall()]
    assert 'options_json' in cols_ax
    cols_series = [r[1] for r in con.execute('PRAGMA table_info(__series_descriptors__)').fetchall()]
    assert 'style_json' in cols_series

    assert '__figure_descriptors__' in names
    assert '__axis_descriptors__' in names
    assert '__series_descriptors__' in names
    assert '__import_links__' in names


def test_list_user_tables_excludes_metadata(tmp_db_path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)

    # Create a user table and a descriptor-like table directly via sqlite3.
    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        con.execute('CREATE TABLE IF NOT EXISTS data_table (x INTEGER)')
        con.execute('CREATE TABLE IF NOT EXISTS __foo_descriptors__ (x INTEGER)')

    names = set(repo.list_user_tables()["Table"].tolist())
    assert 'data_table' in names
    assert '__import_links__' not in names
    assert '__foo_descriptors__' not in names

def test_create_empty_writes_the_file_and_its_system_tables(tmp_db_path) -> None:
    """Startup uses this when the remembered database has gone: the point is
    that the failure surfaces here, not on the first query."""
    missing = tmp_db_path.parent / "gone_and_recreated.dhub"
    if missing.exists():
        missing.unlink()

    created = SqliteRepo.create_empty(missing)

    assert created.exists()
    with sqlite3.connect(str(created)) as con:
        names = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert '__figure_descriptors__' in names
    assert '__import_links__' in names






def test_a_table_with_a_space_in_its_name_has_its_hide_preview_undone(tmp_path) -> None:
    """Outliers' Preview then Cancel: Hide back as it was, the helper column gone.

    table_info did not quote the name, so for "CO2 per capita" it read no
    columns, and the restore and the clean-up both quietly did nothing.
    """
    import pandas as pd

    from app.data.sqlite_repo import SqliteRepo

    repo = SqliteRepo(db_path=tmp_path / "space.dhub")
    repo.import_dataframe(pd.DataFrame({"x": [1, 2, 3]}), table_name="CO2 per capita", normalize_columns=False)
    assert repo.get_columns("CO2 per capita") == ["x"]
    repo.ensure_preview_state_columns("CO2 per capita")
    repo.mark_hide_rowids(table_name="CO2 per capita", rowids=[1, 3])
    assert repo.count_hidden_rows("CO2 per capita") == 2
    repo.restore_preview_state_columns("CO2 per capita")
    repo.drop_preview_state_columns("CO2 per capita")
    assert repo.count_hidden_rows("CO2 per capita") == 0
    assert repo.get_columns("CO2 per capita") == ["x", "Hide"]
    repo.close()


def test_a_figure_with_a_damaged_options_field_still_loads(tmp_path) -> None:
    """One axis's options that are not JSON leave that axis with none, not the figure unopened."""
    from app.data.sqlite_repo import SqliteRepo

    repo = SqliteRepo(db_path=tmp_path / "damaged.dhub")
    figure_id = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Line Plot",
                                              title="", x_label="", y_label="", options={}))
    repo.query_df("UPDATE __axis_descriptors__ SET options_json = '{not json' WHERE id = ?", (axis_id,))
    figure = repo.load_figure_descriptor(figure_id)
    assert figure is not None and figure.axes and figure.axes[0].options == {}
    repo.close()
