"""Peak-finding on a surface (a series with a z role).

``scipy.signal.find_peaks`` is 1D; a series with x, y and z roles gets a 2D
local-maximum search instead (``_search_3d``/``_find_one_3d``), built around
``scipy.ndimage.maximum_filter``/``minimum_filter`` rather than find_peaks.
The 1D path (tested extensively in ``test_new_operations.py``) is untouched -
these tests only exercise the new branch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.peaks_dialog import (
    PEAKS_MAXIMA,
    SeriesPeaksDialog,
)
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    """See test_calculus_surface.py's own fixture: SeriesPeaksDialog's
    model_combo persists to the real user.json across runs."""
    clear_state("SeriesPeaksDialog")
    yield
    clear_state("SeriesPeaksDialog")


def _bare(cls):
    """Same helper as test_new_operations.py: numerics need no Qt window."""
    return cls.__new__(cls)


DEFAULT_PEAK_PARAMS = {
    "filter_by": "prominence",
    "threshold": 0.05,
    # Larger than the 1D default: 2D "prominence" here is height above the
    # minimum of a fixed-size neighbourhood (see _search_3d's own
    # docstring), not a true topographic prominence over the whole basin -
    # so the neighbourhood has to be wide enough, relative to the grid's own
    # spacing, to actually see the bump fall off. distance=1 (a 3x3 window)
    # sees only the immediate, nearly-flat neighbours of a smooth peak.
    "distance": 5,
    "min_width": 0.0,
    "limit": 50,
}


# ----------------------------------------------------------------------
# Pure numerics: a single Gaussian bump on a clean grid
# ----------------------------------------------------------------------

def _bump_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_lin = np.linspace(-5.0, 5.0, 61)
    y_lin = np.linspace(-5.0, 5.0, 61)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = 2.0 * np.exp(-((X - 1.0) ** 2 + (Y + 2.0) ** 2) / 2.0)
    return X, Y, Z


def test_a_single_bump_is_found_near_its_true_centre() -> None:
    X, Y, Z = _bump_grid()
    found = _bare(SeriesPeaksDialog)._search_3d(
        X, Y, Z, DEFAULT_PEAK_PARAMS, minimum=False
    )

    assert len(found) >= 1
    best = max(found, key=lambda peak: peak.prominence)
    assert best.x == pytest.approx(1.0, abs=0.2)
    assert best.y == pytest.approx(-2.0, abs=0.2)
    assert best.z == pytest.approx(2.0, abs=0.05)


def test_a_dip_is_found_by_inverting_the_signal() -> None:
    X, Y, Z = _bump_grid()
    found = _bare(SeriesPeaksDialog)._search_3d(
        X, Y, -Z, DEFAULT_PEAK_PARAMS, minimum=True
    )
    assert len(found) >= 1
    best = max(found, key=lambda peak: peak.prominence)
    assert best.x == pytest.approx(1.0, abs=0.2)
    assert best.y == pytest.approx(-2.0, abs=0.2)


def test_nan_cells_never_win_and_are_never_reported() -> None:
    """A hole in the grid (as an interpolated one has, outside the convex
    hull) must not be picked up as a peak, and must not inflate a real peak's
    prominence to infinity."""
    X, Y, Z = _bump_grid()
    Z = Z.copy()
    Z[0:5, 0:5] = np.nan  # a corner far from the real bump

    found = _bare(SeriesPeaksDialog)._search_3d(
        X, Y, Z, DEFAULT_PEAK_PARAMS, minimum=False
    )
    assert all(np.isfinite(peak.prominence) for peak in found)
    assert all(not (r < 5 and c < 5) for peak in found for r, c in [
        (int(np.argmin(np.abs(Y[:, 0] - peak.y))), int(np.argmin(np.abs(X[0, :] - peak.x))))
    ])


# ----------------------------------------------------------------------
# End to end: exact grid vs. interpolated (scattered/jittered) data
# ----------------------------------------------------------------------

@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


def _make_series(repo: SqliteRepo, table: str, x, y, z) -> tuple[int, int]:
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
    return figure_id, axis_id


def test_peaks_on_an_exact_grid_series(qapp, repo: SqliteRepo) -> None:
    x_lin = np.linspace(-5.0, 5.0, 41)
    y_lin = np.linspace(-5.0, 5.0, 41)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = 3.0 * np.exp(-((X - 2.0) ** 2 + (Y - 1.0) ** 2) / 2.0)
    figure_id, _axis_id = _make_series(
        repo, "grid_src", X.ravel(), Y.ravel(), Z.ravel()
    )

    from app.series_operations.peaks_dialog import SeriesPeaksDialog as Dialog

    dialog = Dialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.select_all_series()
    dialog.model_combo.setCurrentText(PEAKS_MAXIMA)

    results = dialog.compute_results()
    assert len(results) == 1
    result = results[0]
    assert result.metadata["is_3d"] is True
    assert result.metadata["interpolated"] is False
    assert result.peaks
    best = max(result.peaks, key=lambda p: p.prominence)
    assert best.x == pytest.approx(2.0, abs=0.3)
    assert best.y == pytest.approx(1.0, abs=0.3)

    frame = result.to_frame()
    assert list(frame.columns) == ["x", "y", "z", "prominence", "is_minimum"]


def test_peaks_on_a_jittered_scattered_series_use_interpolation(qapp, repo: SqliteRepo) -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-5.0, 5.0, 600)
    y = rng.uniform(-5.0, 5.0, 600)
    z = 3.0 * np.exp(-((x - 2.0) ** 2 + (y - 1.0) ** 2) / 2.0) + rng.normal(0.0, 0.01, 600)
    figure_id, _axis_id = _make_series(repo, "scattered_src", x, y, z)

    from app.series_operations.peaks_dialog import SeriesPeaksDialog as Dialog

    dialog = Dialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.select_all_series()
    dialog.model_combo.setCurrentText(PEAKS_MAXIMA)
    # See DEFAULT_PEAK_PARAMS's own comment: the interpolated grid is
    # resolution=120 over a span of 10, so the neighbourhood needs to be
    # wide enough in cells to actually see the bump (sigma=1) fall off.
    dialog.set_parameter_values({"distance": 15})

    results = dialog.compute_results()
    result = results[0]
    assert result.metadata["is_3d"] is True
    assert result.metadata["interpolated"] is True
    assert result.peaks
    best = max(result.peaks, key=lambda p: p.prominence)
    assert best.x == pytest.approx(2.0, abs=0.5)
    assert best.y == pytest.approx(1.0, abs=0.5)
