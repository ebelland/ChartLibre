"""The demo set: real datasets, each named for what it shows.

A demo is the first thing a new installation opens, so its failures are the
ones nobody reports - a chart with no data behind it, a table the figures do
not read, a file whose name says nothing. These tests build the whole set and
open every file, because that is the only way to know the demo demonstrates
anything.
"""
from __future__ import annotations

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


