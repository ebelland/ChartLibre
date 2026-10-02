"""The demo set: real datasets, each named for what it shows.

A demo is the first thing a new installation opens, so its failures are the
ones nobody reports - a chart with no data behind it, a table the figures do
not read, a file whose name says nothing. These tests build the whole set and
open every file, because that is the only way to know the demo demonstrates
anything.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from app.data.demos import DEMO_PROJECTS
from dev.demo.build_demos import build_demo_project, build_demo_projects, _figure_specs, _multi_axis_figure_specs
from app.data.sqlite_repo import SqliteRepo


def _all_specs():
    """Every figure spec, single- and multi-axis alike."""
    return [*_figure_specs(), *_multi_axis_figure_specs()]


@pytest.fixture(scope="module")
def demo_set(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """The whole set, built once from the real sample data."""
    return build_demo_projects(tmp_path_factory.mktemp("demo_set"))


# ----------------------------------------------------------------------
# What the set is
# ----------------------------------------------------------------------


def test_every_demo_names_figures_that_exist() -> None:
    """A typo in a figure key would give a demo file with no charts at all."""
    known = {spec.key for spec in _all_specs()}

    for demo in DEMO_PROJECTS:
        unknown = sorted(set(demo.figures) - known)
        assert unknown == [], f"{demo.file_name}: {unknown}"


# ----------------------------------------------------------------------
# What is in each file
# ----------------------------------------------------------------------
def _open(path: Path) -> tuple[list[tuple[int, str]], list[str], list[str]]:
    repo = SqliteRepo(db_path=path)
    try:
        return (
            list(repo.get_figures()),
            list(repo.list_table_names()),
            [saved.name for saved in repo.list_queries()],
        )
    finally:
        repo.close()


def test_every_file_opens_and_has_charts(demo_set: list[Path]) -> None:
    for path in demo_set:
        figures, tables, _queries = _open(path)
        assert figures, f"{path.name} has no figures"
        assert tables, f"{path.name} has no tables"


def test_every_series_in_every_file_returns_rows(demo_set: list[Path]) -> None:
    """The failure this catches is the quiet one: a chart drawn from a query
    that matches nothing looks like an empty axis and reads as a broken app."""
    empty: list[str] = []

    for path in demo_set:
        repo = SqliteRepo(db_path=path)
        try:
            for figure_id, _name in repo.get_figures():
                descriptor = repo.load_figure_descriptor(figure_id=int(figure_id))
                assert descriptor is not None
                for axis in descriptor.axes:
                    for series in axis.series:
                        frame = repo.query_df(series.sql_query)
                        if frame.empty:
                            empty.append(f"{path.stem}: {series.name}")
        finally:
            repo.close()

    assert empty == []


def test_building_twice_gives_the_same_data(tmp_path: Path) -> None:
    """The source files are static, so nothing here should vary between
    builds - unlike the old synthetic generator, there is no seed to pin."""
    first = build_demo_project(tmp_path / "one.dhub", ("dlvo_force",))
    second = build_demo_project(tmp_path / "two.dhub", ("dlvo_force",))

    def forces(path: Path) -> list[float]:
        repo = SqliteRepo(db_path=path)
        try:
            return repo.query_df(
                "SELECT F_total_nN FROM dlvo_curve ORDER BY separation_nm"
            )["F_total_nN"].tolist()
        finally:
            repo.close()

    assert forces(first) == forces(second)


# ----------------------------------------------------------------------
# The datasets shaped for a specific operation
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Multi-axis figures: one demo per layout preset
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# What the running application does with a pre-built file
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# Loading a demo again
# ----------------------------------------------------------------------
def test_loading_a_demo_again_gives_the_pristine_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last copy's -wal and undo history go with it.

    A -wal left beside the old copy - by a connection still open, or a crash
    - was replayed by SQLite onto the fresh file, so a demo loaded again
    came back with the edits made to it last time.
    """
    import sqlite3

    import app.data.demos as demos

    source_dir = tmp_path / "built"
    source_dir.mkdir()
    demo = demos.DemoProject("Tiny", "One table.", ())
    with sqlite3.connect(source_dir / demo.path_name) as con:
        con.execute("CREATE TABLE t (v INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
    monkeypatch.setattr(demos, "DEMO_DIR", source_dir)

    target = demos.copy_demo_project(demo, tmp_path / "projects" / demo.path_name)
    con = sqlite3.connect(target)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("DELETE FROM t")
    con.commit()
    wal = target.with_name(target.name + "-wal")
    pending = wal.read_bytes()
    con.close()
    wal.write_bytes(pending)  # the edit, still waiting in the log
    undo = target.with_name(target.name + ".undo.db")
    undo.write_bytes(b"old history")

    demos.copy_demo_project(demo, target)

    with sqlite3.connect(target) as con:
        assert con.execute("SELECT v FROM t").fetchall() == [(1,)]
    assert not undo.exists()


# ----------------------------------------------------------------------
# The pictures Load demo shows
# ----------------------------------------------------------------------
def test_every_demo_has_a_preview_and_a_named_one_exists(tmp_path: Path) -> None:
    """Rebuilt with python3 -m dev.demo.demo_previews after a demo changes."""
    import shutil

    from PIL import Image

    for demo in DEMO_PROJECTS:
        assert demo.preview_path.exists(), f"no preview for {demo.file_name}"
        with Image.open(demo.preview_path) as picture:
            assert picture.size == (640, 400)
        if demo.preview:
            # A copy: even reading a WAL database leaves -shm/-wal files beside
            # it, and the shipped project must not be touched by a test.
            copy = Path(shutil.copy(demo.source_path, tmp_path / demo.path_name))
            with closing(sqlite3.connect(str(copy))) as con:
                names = [str(name) for (name,) in con.execute("SELECT name FROM __figure_descriptors__")]
            assert any(demo.preview in name for name in names), f"{demo.file_name}: no figure named {demo.preview!r}"


def test_a_preview_is_drawn_from_the_built_project(demo_set: list[Path], tmp_path: Path) -> None:
    import shutil

    from dev.demo.demo_previews import build_preview

    demo = next(demo for demo in DEMO_PROJECTS if demo.preview)
    shutil.copy(Path(demo_set[0]).parent / demo.path_name, tmp_path / demo.path_name)
    written = build_preview(demo, tmp_path)
    assert written is not None and written == tmp_path / "previews" / f"{demo.file_name}.png"
    assert written.stat().st_size > 5_000


def test_load_demo_shows_the_selected_demos_picture(qapp) -> None:
    from app.dialogs.load_demo_dialog import LoadDemoDialog

    dialog = LoadDemoDialog()
    for row in (0, len(DEMO_PROJECTS) - 1):
        dialog._list.setCurrentRow(row)
        picture = dialog._preview.pixmap()
        assert not picture.isNull() and picture.width() / picture.devicePixelRatio() <= dialog.PREVIEW_SIZE.width()
    dialog.close()
