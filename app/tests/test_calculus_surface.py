"""Derivative and integral of a surface (a series with a z role).

``_gradient_surface`` and ``_volume_surface`` are plain numerical methods -
tested directly here, the same way test_new_operations.py tests the 1D
derivative/integral methods - plus one end-to-end check through the dialog
and a real (interpolated) series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.utils.dialog_state import clear_state
from app.series_operations.calculus_dialog import (
    DERIV_GRADIENT_SURFACE,
    INTEGRAL_VOLUME_SURFACE,
    SeriesCalculusDialog,
)


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    """SeriesCalculusDialog remembers its model_combo across runs (see
    dialog_state.py) - a real user.json, not a per-test tmp file. Without
    this, a surface model selected here leaks into any other test that
    builds a fresh SeriesCalculusDialog and relies on its default model,
    such as test_datetime_x_operations.py's dated-series check."""
    clear_state("SeriesCalculusDialog")
    yield
    clear_state("SeriesCalculusDialog")


def _bare() -> SeriesCalculusDialog:
    return SeriesCalculusDialog.__new__(SeriesCalculusDialog)


# ----------------------------------------------------------------------
# Gradient: axis order verified against a known plane, z = 2x + 3y
# ----------------------------------------------------------------------

def test_the_gradient_of_a_plane_is_its_constant_coefficients() -> None:
    """z = 2x + 3y has dz/dx = 2 and dz/dy = 3 everywhere.

    This is exactly the check the module's own comment on _gradient_surface
    promises: np.gradient's axis order on a meshgrid-built (X, Y) pair is
    not obvious from the numpy docs alone, so it is pinned here rather than
    only asserted in a comment.
    """
    x_lin = np.linspace(0.0, 10.0, 21)
    y_lin = np.linspace(0.0, 6.0, 13)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = 2.0 * X + 3.0 * Y

    result = _bare()._gradient_surface("s", X, Y, Z, False)

    assert np.allclose(result.dz_dx, 2.0)
    assert np.allclose(result.dz_dy, 3.0)
    assert np.allclose(result.z, np.hypot(2.0, 3.0))
    assert result.model == DERIV_GRADIENT_SURFACE


def test_the_gradient_is_zero_on_a_flat_surface() -> None:
    x_lin = np.linspace(-2.0, 2.0, 11)
    y_lin = np.linspace(-2.0, 2.0, 11)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.full_like(X, 5.0)

    result = _bare()._gradient_surface("s", X, Y, Z, False)
    assert np.allclose(result.z, 0.0, atol=1e-10)


# ----------------------------------------------------------------------
# Volume: known shapes with a hand-computable answer
# ----------------------------------------------------------------------

def test_the_volume_under_a_constant_block_is_base_times_height() -> None:
    """A flat z=5 slab over a 10 x 6 rectangle has volume 300."""
    x_lin = np.linspace(0.0, 10.0, 41)
    y_lin = np.linspace(0.0, 6.0, 25)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.full_like(X, 5.0)

    result = _bare()._volume_surface("s", X, Y, Z, False)
    assert result.total == pytest.approx(300.0, rel=1e-6)
    assert result.model == INTEGRAL_VOLUME_SURFACE


def test_the_volume_under_a_pyramid_matches_the_textbook_formula() -> None:
    """z = h * (1 - |x|/a) * (1 - |y|/b) on [-a, a] x [-b, b].

    This separable "pyramid" (a tent, really - each cross-section is a
    triangle in x and a triangle in y) has a closed-form volume of
    (4/3) * a * b * h: the double integral of two independent triangular
    profiles, each contributing a factor of (side length)/2 times 2/3 to its
    own axis, or checked directly by separating the integral:
    ∫∫ h(1-|x|/a)(1-|y|/b) dx dy = h * a * b (each 1D integral of a
    triangular hat of half-width a is a, times height 1, times 1 - wait:
    ∫_{-a}^{a} (1-|x|/a) dx = a).  So volume = h * a * b... - verified
    numerically below against the closed form actually used: (4/3) a b h
    is for a *pointed* pyramid (linear falloff to a single apex over a
    rectangle's diagonal); this tent's separable profile integrates to
    exactly a * b * h. The assertion trusts the numerical double integral
    of the exact 1D triangle-integral identity, not a memorized formula.
    """
    a, b, h = 4.0, 3.0, 2.0
    x_lin = np.linspace(-a, a, 401)
    y_lin = np.linspace(-b, b, 301)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = h * np.clip(1.0 - np.abs(X) / a, 0.0, None) * np.clip(1.0 - np.abs(Y) / b, 0.0, None)

    result = _bare()._volume_surface("s", X, Y, Z, False)
    # exact separable integral: (integral of the x-triangle = a) * (integral
    # of the y-triangle = b) * h
    assert result.total == pytest.approx(a * b * h, rel=1e-3)


def test_nan_cells_are_treated_as_zero_and_reported() -> None:
    x_lin = np.linspace(0.0, 10.0, 41)
    y_lin = np.linspace(0.0, 6.0, 25)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.full_like(X, 5.0)
    Z[0:3, 0:3] = np.nan

    result = _bare()._volume_surface("s", X, Y, Z, True)
    assert result.total < 300.0  # less than the no-hole answer
    assert "0" in result.metadata["detail"]
    assert result.metadata["interpolated"] is True


# ----------------------------------------------------------------------
# End to end, through the dialog
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


def test_gradient_end_to_end_on_a_plane_series(qapp, repo: SqliteRepo) -> None:
    x_lin = np.linspace(0.0, 10.0, 31)
    y_lin = np.linspace(0.0, 6.0, 19)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = 2.0 * X + 3.0 * Y
    figure_id = _make_series(repo, "plane_src", X.ravel(), Y.ravel(), Z.ravel())

    dialog = SeriesCalculusDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.select_all_series()
    dialog.model_combo.setCurrentText(DERIV_GRADIENT_SURFACE)

    results = dialog.compute_results()
    assert len(results) == 1
    result = results[0]
    assert result.model == DERIV_GRADIENT_SURFACE
    assert np.allclose(result.z, np.hypot(2.0, 3.0))

    frame = result.to_frame()
    assert set(frame.columns) == {"x", "y", "z", "dz_dx", "dz_dy"}


def test_volume_end_to_end_on_a_block_series(qapp, repo: SqliteRepo) -> None:
    x_lin = np.linspace(0.0, 10.0, 31)
    y_lin = np.linspace(0.0, 6.0, 19)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.full_like(X, 5.0)
    figure_id = _make_series(repo, "block_src", X.ravel(), Y.ravel(), Z.ravel())

    dialog = SeriesCalculusDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.select_all_series()
    dialog.model_combo.setCurrentText(INTEGRAL_VOLUME_SURFACE)

    results = dialog.compute_results()
    result = results[0]
    assert result.model == INTEGRAL_VOLUME_SURFACE
    assert result.total == pytest.approx(300.0, rel=1e-6)
