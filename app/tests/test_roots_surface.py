"""Level curves (z = level) on a surface (a series with a z role).

The 1D bracket-then-SciPy machinery assumes y is single-valued in x, which a
surface's level set is not - a saddle's z = 0 set is two crossing lines.
``_solve_one_3d``/``_extract_level_curves`` reuse matplotlib's own contour
extraction instead. The 1D path (``test_roots_operation.py``) is untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.roots_dialog import SeriesRootsDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    """See test_calculus_surface.py's own fixture: SeriesRootsDialog's
    model_combo persists to the real user.json across runs."""
    clear_state("SeriesRootsDialog")
    yield
    clear_state("SeriesRootsDialog")


def _bare() -> SeriesRootsDialog:
    return SeriesRootsDialog.__new__(SeriesRootsDialog)


# ----------------------------------------------------------------------
# Pure numerics: known level curves on a clean grid
# ----------------------------------------------------------------------

def test_a_plane_gives_one_straight_diagonal_line() -> None:
    """z = x - y crosses zero exactly on the line y = x."""
    x_lin = np.linspace(-3.0, 3.0, 61)
    y_lin = np.linspace(-3.0, 3.0, 61)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = X - Y

    curves = _bare()._extract_level_curves(X, Y, Z, 0.0)
    assert len(curves) == 1
    xs, ys = curves[0]
    assert np.max(np.abs(xs - ys)) < 1e-6


def test_a_saddle_gives_two_diagonal_lines() -> None:
    """z = x^2 - y^2 = 0 factors as (x-y)(x+y) = 0: y = x and y = -x."""
    x_lin = np.linspace(-3.0, 3.0, 121)
    y_lin = np.linspace(-3.0, 3.0, 121)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = X**2 - Y**2

    curves = _bare()._extract_level_curves(X, Y, Z, 0.0)
    # Matplotlib may return the crossing lines as 2 or more segments
    # (it does not know they meet at the origin and is not required to
    # stitch them into one path) - what matters is that every point found
    # really does lie on y = x or y = -x.
    assert len(curves) >= 2
    for xs, ys in curves:
        on_positive_diagonal = np.abs(xs - ys) < 1e-2
        on_negative_diagonal = np.abs(xs + ys) < 1e-2
        assert np.all(on_positive_diagonal | on_negative_diagonal)


def test_no_crossing_gives_no_curves() -> None:
    """A bump entirely above zero never reaches level=0."""
    x_lin = np.linspace(-3.0, 3.0, 41)
    y_lin = np.linspace(-3.0, 3.0, 41)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = 1.0 + np.exp(-(X**2 + Y**2))

    curves = _bare()._extract_level_curves(X, Y, Z, 0.0)
    assert curves == []


# ----------------------------------------------------------------------
# End to end, through the dialog and a real series
# ----------------------------------------------------------------------

@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


def _make_series(repo: SqliteRepo, table: str, x, y, z) -> int:
    repo.import_dataframe(
        pd.DataFrame({"x": x, "y": y, "z": z}), table_name=table, normalize_columns=False
    )
    figure_id = repo.create_figure_descriptor(name=table)
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot (3D)",
        title=table,
        x_label="",
        y_label="",
        options={"projection": "3d"},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name=table,
        sql_query=f"SELECT x, y, z FROM {table}",
        roles={"x": "x", "y": "y", "z": "z"},
        style={},
    )
    return figure_id


def test_saddle_roots_end_to_end_on_an_exact_grid(qapp, repo: SqliteRepo) -> None:
    x_lin = np.linspace(-3.0, 3.0, 61)
    y_lin = np.linspace(-3.0, 3.0, 61)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = X**2 - Y**2
    figure_id = _make_series(repo, "saddle_src", X.ravel(), Y.ravel(), Z.ravel())

    dialog = SeriesRootsDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.select_all_series()

    results = dialog.compute_results()
    assert len(results) == 1
    result = results[0]
    assert result.metadata["is_3d"] is True
    assert result.metadata["interpolated"] is False
    assert result.roots

    for root in result.roots:
        on_positive_diagonal = abs(root.x - root.y) < 0.15
        on_negative_diagonal = abs(root.x + root.y) < 0.15
        assert on_positive_diagonal or on_negative_diagonal

    frame = result.to_frame()
    assert list(frame.columns) == ["x", "y", "z", "curve_index"]
    # z is the constant level (0.0) everywhere a point is not a NaN
    # separator row between two curves.
    assert np.allclose(frame["z"].dropna(), 0.0)


def test_plane_roots_on_a_jittered_scattered_series_use_interpolation(
    qapp, repo: SqliteRepo
) -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-3.0, 3.0, 800)
    y = rng.uniform(-3.0, 3.0, 800)
    z = x - y  # exact plane, no noise - only the sampling is irregular

    figure_id = _make_series(repo, "plane_src", x, y, z)

    dialog = SeriesRootsDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.select_all_series()

    results = dialog.compute_results()
    result = results[0]
    assert result.metadata["is_3d"] is True
    assert result.metadata["interpolated"] is True
    assert result.roots

    max_error = max(abs(root.x - root.y) for root in result.roots)
    assert max_error < 0.5
