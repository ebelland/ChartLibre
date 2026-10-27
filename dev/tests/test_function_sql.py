"""A stored function as SQL: its grid and its values computed by SQLite at each read (function_sql)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.data.repo.function_sql import GridRange, function_sql, grid_point
from app.data.sqlite_repo import SqliteRepo
from app.data_operations import sympy_store


@pytest.fixture
def repo(tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "f.dhub")
    yield repo
    repo.close()


def _save(repo: SqliteRepo, name: str, code: str, symbols: str) -> None:
    sympy_store.save(repo, {"function_name": name, "source_type": "expression", "symbols": symbols,
                            "code_string": code})


def _rows(repo: SqliteRepo, sql: str) -> np.ndarray:
    return np.array([tuple(row) for row in repo._con.execute(sql).fetchall()], dtype=float)


def test_the_grid_is_numpys_linear_or_log_range() -> None:
    np.testing.assert_allclose([grid_point(n, -2, 3, 11, "linear") for n in range(11)], np.linspace(-2, 3, 11))
    np.testing.assert_allclose([grid_point(n, 0.1, 1000, 9, "log") for n in range(9)], np.logspace(-1, 3, 9))
    assert grid_point(3, 0, 10, 11, "linear") == pytest.approx(3.0) and grid_point(10, 0, 0.3, 11, "linear") == 0.3
    assert grid_point(0, -1, 10, 5, "log") is None


def test_a_function_is_drawn_from_its_formula_with_each_charts_parameters(repo: SqliteRepo) -> None:
    _save(repo, "Line", "a*x + b", "a, b")
    sql = function_sql("Line", {"a": 2.0, "b": 1.0}, GridRange(0.0, 4.0, 5))
    np.testing.assert_allclose(_rows(repo, sql), [[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]])
    # Another chart, other parameters - a missing one at its default.
    other = function_sql("Line", {"a": -1.0}, GridRange(0.0, 4.0, 5))
    np.testing.assert_allclose(_rows(repo, other)[:, 1], [1, 0, -1, -2, -3])
    # Read as a saved query is: a subquery.
    assert repo._con.execute(f'SELECT COUNT(*), MAX("y") FROM ({sql}) AS "_src"').fetchone()[:] == (5, 9.0)

    # Saved again with a new formula: the same query draws the new one.
    _save(repo, "Line", "a*x**2 + b", "a, b")
    np.testing.assert_allclose(_rows(repo, sql)[:, 1], [1, 3, 9, 19, 33])


def test_a_function_of_x_and_y_is_a_grid_x_fastest(repo: SqliteRepo) -> None:
    _save(repo, "Plane", "a*x + b*y", "a, b")
    rows = _rows(repo, function_sql("Plane", {"a": 1.0, "b": 10.0}, GridRange(0, 2, 3), GridRange(1, 1e3, 4, "log")))
    x_grid, y_grid = np.meshgrid(np.linspace(0, 2, 3), np.logspace(0, 3, 4), indexing="xy")
    np.testing.assert_allclose(rows, np.column_stack((x_grid.ravel(), y_grid.ravel(),
                                                      x_grid.ravel() + 10 * y_grid.ravel())))


def test_an_untrusted_or_failing_function_draws_nothing(repo: SqliteRepo) -> None:
    # Written as a project made elsewhere has it: never saved, never trusted here (todo ID47).
    repo.save_sympy_function("Foreign", "expression", "a", "a*x + 0.123456")
    sql = function_sql("Foreign", {"a": 1.0}, GridRange(0, 1, 3))
    assert [row[1] for row in repo._con.execute(sql)] == [None, None, None]
    _save(repo, "Root", "sqrt(x)", "")
    assert [row[1] for row in repo._con.execute(function_sql("Root", {}, GridRange(-1, 1, 3)))] == [None, 0.0, 1.0]
    assert [row[1] for row in repo._con.execute(function_sql("Missing", {}, GridRange(0, 1, 2)))] == [None, None]
