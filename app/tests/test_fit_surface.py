"""Surface (z = f(x, y)) fitting through the same dialog as the 1D case.

``_is_2d_fit`` used to always return False - the fit dialog only ever fit
``target = f(x)``. This exercises the completed path end to end: a Plane
model fit against a noisy plane of (x, y, z) points read straight off the
series' own x/y/z roles, with no extra "X2" picker involved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.data_source import parse_roles
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.fit_dialog import SeriesFitDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    """See test_calculus_surface.py's own fixture: SeriesFitDialog's
    model_combo/output-table entries persist to the real user.json, and
    dialog.apply() below writes them - clear both sides so a surface model
    selected here never leaks into another test's fresh SeriesFitDialog."""
    clear_state("SeriesFitDialog")
    yield
    clear_state("SeriesFitDialog")


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    rng = np.random.default_rng(12345)
    x = rng.uniform(-5.0, 5.0, 200)
    y = rng.uniform(-5.0, 5.0, 200)
    # z = 2 + 3x - 1.5y, plus a little noise - small enough that a plane fit
    # should land close to (2, 3, -1.5).
    z = 2.0 + 3.0 * x - 1.5 * y + rng.normal(0.0, 0.02, 200)
    built.import_dataframe(
        pd.DataFrame({"x": x, "y": y, "z": z}),
        table_name="surface_src",
        normalize_columns=False,
    )
    yield built
    built.close()


@pytest.fixture
def figure_id(repo: SqliteRepo) -> int:
    figure_id = repo.create_figure_descriptor(name="fig")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot (3D)",
        title="surface",
        x_label="",
        y_label="",
        options={"projection": "3d"},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="SURFACE",
        sql_query="SELECT x, y, z FROM surface_src",
        roles={"x": "x", "y": "y", "z": "z"},
        style={},
    )
    return figure_id


def _select_model_by_name(dialog: SeriesFitDialog, name: str) -> None:
    tree = dialog._models_tree
    for i in range(tree.topLevelItemCount()):
        category = tree.topLevelItem(i)
        for j in range(category.childCount()):
            child = category.child(j)
            payload = child.data(0, 0x0100)  # Qt.ItemDataRole.UserRole
            if isinstance(payload, dict) and payload.get("name") == name:
                tree.setCurrentItem(child)
                return
    raise AssertionError(f"Model {name!r} not found in the catalog tree")


@pytest.fixture
def dialog(qapp, repo: SqliteRepo, figure_id: int) -> SeriesFitDialog:
    built = SeriesFitDialog(repo=repo, figure_id=figure_id)
    built.series_selector.select_all_series()
    _select_model_by_name(built, "Plane")
    # Both boxes are restored from config.json across test runs (see
    # test_fit_accessory_charts.py's own dialog fixture for the same note);
    # this test cares about the main fit series only.
    built._residual_chart_check.setChecked(False)
    built._fit_vs_measured_chart_check.setChecked(False)
    return built


def test_plane_model_is_recognised_as_a_2d_fit(dialog: SeriesFitDialog) -> None:
    assert dialog._is_2d_fit() is True


def test_plane_fit_recovers_the_true_coefficients(dialog: SeriesFitDialog) -> None:
    dialog.on_fit()

    result = dialog._last_result
    assert result is not None
    assert result.fit_mode == "2D"

    params = dict(zip(result.param_names, result.params))
    assert params["a"] == pytest.approx(2.0, abs=0.1)
    assert params["b"] == pytest.approx(3.0, abs=0.1)
    assert params["c"] == pytest.approx(-1.5, abs=0.1)

    # A good plane fit on near-noiseless data should explain nearly all the
    # variance.
    assert result.metrics["r2"] > 0.99


def test_the_output_frame_carries_x_y_z_and_z_fit(dialog: SeriesFitDialog) -> None:
    dialog.on_fit()
    frame = dialog._last_result.frame
    for column in ("x", "y", "z", "z_fit", "residual"):
        assert column in frame.columns
    assert len(frame) == 200


def test_apply_draws_a_surface_series_with_xyz_roles(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    dialog.on_fit()
    assert dialog.apply() is True

    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    axis = descriptor.axes[0]
    names = [str(series.name) for series in axis.series]
    assert any(name.startswith("Fit:") for name in names)

    fit_series = next(series for series in axis.series if str(series.name).startswith("Fit:"))
    roles = parse_roles(fit_series.roles)
    assert roles.get("z") == "z"
    assert roles.get("x") == "x"
    assert roles.get("y") == "y"
