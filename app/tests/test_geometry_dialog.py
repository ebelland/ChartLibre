"""The Geometry operation writes a query, so the query is what must be right.

Every other operation computes numbers and stores them; this one hands SQL
to SQLite and lets it compute. So the test that matters is not "are these
the coordinates I expect" but "does the database agree with the transform
the preview drew" - two implementations of one affine map, which is exactly
the pair that can drift apart silently.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.geometry_dialog import (
    MIRROR,
    ROTATE,
    SCALE,
    SHEAR,
    TRANSLATE,
    SeriesGeometryDialog,
)

#: A unit square plus a point off-centre, so a mirror and a shear have
#: something asymmetric to move - four corners alone look the same after
#: several of these transforms.
SQUARE = pd.DataFrame(
    {
        "x": [0.0, 1.0, 1.0, 0.0, 0.25],
        "y": [0.0, 0.0, 1.0, 1.0, 0.75],
    }
)


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(SQUARE, table_name="square", normalize_columns=False)
    yield built
    built.close()


@pytest.fixture
def dialog(qapp, repo: SqliteRepo):
    figure_id = repo.create_figure_descriptor(name="Geometry test")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="square", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="square",
        sql_query='SELECT "x", "y" FROM "square"',
        roles={"x": "x", "y": "y"},
        style={},
    )
    built = SeriesGeometryDialog(repo=repo, figure_id=figure_id, parent=None)
    built.series_selector.reload(select_all_series=True)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _configure(dialog: SeriesGeometryDialog, model: str, **values) -> None:
    dialog.model_combo.setCurrentText(model)
    form = dialog._parameter_form_spec
    current = dict(form.values())
    current.update(values)
    form.set_values(current)


def _sql_result(repo: SqliteRepo, sql: str) -> pd.DataFrame:
    return repo.query_df(sql)


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (ROTATE, {"angle": 30.0, "use_data_centre": False, "cx": 0.0, "cy": 0.0}),
        (ROTATE, {"angle": -90.0, "use_data_centre": True}),
        (TRANSLATE, {"dx": 3.5, "dy": -2.25}),
        (SCALE, {"sx": 2.0, "sy": 0.5, "use_data_centre": False, "cx": 1.0, "cy": 1.0}),
        (MIRROR, {"mirror_line": 0.0, "use_data_centre": False, "cx": 0.0, "cy": 0.0}),
        (MIRROR, {"mirror_line": 90.0, "use_data_centre": True}),
        (MIRROR, {"mirror_line": 45.0, "use_data_centre": False, "cx": 0.0, "cy": 0.0}),
        (SHEAR, {"kx": 0.4, "ky": 0.0, "use_data_centre": False, "cx": 0.0, "cy": 0.0}),
        (SHEAR, {"kx": 0.0, "ky": -0.3, "use_data_centre": True}),
    ],
)
def test_the_database_computes_what_the_preview_drew(
    dialog: SeriesGeometryDialog, repo: SqliteRepo, model: str, values: dict
) -> None:
    """The generated SQL and the Python affine must agree, point for point."""
    _configure(dialog, model, **values)
    result = dialog.compute_results()[0]

    rows = _sql_result(repo, result.sql)

    assert list(rows.columns)[:2] == [result.x_role, result.y_role]
    np.testing.assert_allclose(rows[result.x_role].to_numpy(), result.after_x, atol=1e-9)
    np.testing.assert_allclose(rows[result.y_role].to_numpy(), result.after_y, atol=1e-9)


def test_a_rotation_is_written_as_trigonometry(dialog: SeriesGeometryDialog) -> None:
    """Not as a decimal: the query is meant to be read and edited.

    Guarded on the database actually having the math functions, because
    where it does not the operation deliberately writes the coefficients
    out instead of a query that would error.
    """
    _configure(dialog, ROTATE, angle=30.0)
    result = dialog.compute_results()[0]

    if dialog._supports_trig_sql():
        assert "cos(radians(30))" in result.sql
        assert "sin(radians(30))" in result.sql
    else:
        assert "0.866" in result.sql


def test_the_source_query_is_kept_whole(dialog: SeriesGeometryDialog) -> None:
    """The transform wraps the source rather than rewriting it."""
    _configure(dialog, TRANSLATE, dx=1.0, dy=1.0)
    result = dialog.compute_results()[0]

    assert 'SELECT "x", "y" FROM "square"' in result.sql
    assert result.sql.startswith("WITH source AS")


def test_a_rotation_of_zero_moves_nothing(dialog: SeriesGeometryDialog) -> None:
    _configure(dialog, ROTATE, angle=0.0)
    result = dialog.compute_results()[0]

    np.testing.assert_allclose(result.after_x, result.before_x, atol=1e-9)
    np.testing.assert_allclose(result.after_y, result.before_y, atol=1e-9)


def test_four_right_angles_return_the_shape_to_itself(
    dialog: SeriesGeometryDialog, repo: SqliteRepo
) -> None:
    """Composition check: the query is re-runnable, so feed it back in.

    This is the property that the operation's whole premise rests on - the
    result is a query over the source, so a transform of a transform is
    just another wrap - and it fails loudly if a centre is captured wrong.
    """
    _configure(dialog, ROTATE, angle=90.0, use_data_centre=False, cx=0.0, cy=0.0)
    result = dialog.compute_results()[0]

    sql = result.sql
    for _turn in range(3):
        rebuilt = dialog._build_sql(
            sql, result.transform, result.x_role, result.y_role, []
        )
        sql = rebuilt

    rows = _sql_result(repo, sql)
    np.testing.assert_allclose(rows["x"].to_numpy(), result.before_x, atol=1e-9)
    np.testing.assert_allclose(rows["y"].to_numpy(), result.before_y, atol=1e-9)


def test_the_centre_is_a_fixed_point(dialog: SeriesGeometryDialog) -> None:
    """Rotation, scale, mirror and shear all leave the centre alone."""
    for model, values in (
        (ROTATE, {"angle": 37.0}),
        (SCALE, {"sx": 3.0, "sy": 0.2}),
        (MIRROR, {"mirror_line": 45.0}),
        (SHEAR, {"kx": 0.8, "ky": -0.4}),
    ):
        _configure(dialog, model, use_data_centre=False, cx=2.5, cy=-1.5, **values)
        transform = dialog.compute_results()[0].transform
        moved_x, moved_y = transform.apply(np.array([2.5]), np.array([-1.5]))
        assert moved_x[0] == pytest.approx(2.5, abs=1e-9), model
        assert moved_y[0] == pytest.approx(-1.5, abs=1e-9), model


def test_extra_columns_survive_the_transform(
    qapp, repo: SqliteRepo, tmp_db_path: Path
) -> None:
    """A scatter's colour/size columns must come through untouched."""
    frame = SQUARE.copy()
    frame["label"] = ["a", "b", "c", "d", "e"]
    frame["weight"] = [1, 2, 3, 4, 5]
    repo.import_dataframe(frame, table_name="square_rich", normalize_columns=False)

    figure_id = repo.create_figure_descriptor(name="Passthrough")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="square", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="square_rich",
        sql_query='SELECT * FROM "square_rich"',
        roles={"x": "x", "y": "y"},
        style={},
    )
    built = SeriesGeometryDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        built.series_selector.reload(select_all_series=True)
        _configure(built, TRANSLATE, dx=10.0, dy=0.0)
        result = built.compute_results()[0]
        rows = repo.query_df(result.sql)

        assert list(rows["label"]) == ["a", "b", "c", "d", "e"]
        assert list(rows["weight"]) == [1, 2, 3, 4, 5]
        np.testing.assert_allclose(
            rows["x"].to_numpy(), SQUARE["x"].to_numpy() + 10.0, atol=1e-9
        )
    finally:
        built.close()
        applogger.set_status_bar(None)


def test_a_ninety_degree_turn_about_the_origin_is_exact(
    dialog: SeriesGeometryDialog, repo: SqliteRepo
) -> None:
    """(x, y) -> (-y, x). Worth stating in coordinates, not only as a matrix."""
    _configure(dialog, ROTATE, angle=90.0, use_data_centre=False, cx=0.0, cy=0.0)
    result = dialog.compute_results()[0]
    rows = _sql_result(repo, result.sql)

    np.testing.assert_allclose(rows["x"].to_numpy(), -SQUARE["y"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(rows["y"].to_numpy(), SQUARE["x"].to_numpy(), atol=1e-9)


def test_the_results_pane_is_a_plot_not_the_html_view(
    dialog: SeriesGeometryDialog,
) -> None:
    """The operation's own reason for overriding build_results_pane."""
    from app.series_operations.geometry_dialog import _BeforeAfterView

    assert isinstance(dialog._plot_view, _BeforeAfterView)
    _configure(dialog, ROTATE, angle=45.0)
    results = dialog.compute_results()
    dialog.format_results(results)

    axes = dialog._plot_view._figure.axes
    assert axes, "nothing was drawn"
    labels = [line.get_label() for line in axes[0].lines]
    assert "Before" in labels and "After" in labels


def test_no_table_is_written_for_a_geometry_series(
    dialog: SeriesGeometryDialog, repo: SqliteRepo
) -> None:
    """The whole premise: a query, not a copy of the data."""
    before = set(repo.list_user_tables())
    _configure(dialog, ROTATE, angle=15.0)
    results = dialog.compute_results()
    axis_id = dialog.series_selector.selected_axis_id()
    dialog.apply_results_to_axis(axis_id, results)

    assert set(repo.list_user_tables()) == before
