"""The Geometry operation writes a query, so the query is what must be right.

Every other operation computes numbers and stores them; this one hands SQL
to SQLite and lets it compute. So the test that matters is not "are these
the coordinates I expect" but "does the database agree with the transform
the preview drew" - two implementations of one affine map, which is exactly
the pair that can drift apart silently.
"""
from __future__ import annotations

from collections.abc import Iterator

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.geometry_dialog import (
    MIRROR,
    ROTATE,
    ROTO_TRANSLATE,
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
def repo(tmp_db_path: Path) -> Iterator[SqliteRepo]:
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
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData(model))
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
        (ROTO_TRANSLATE, {"angle": 45.0, "dx": 10.0, "dy": -4.0, "use_data_centre": True}),
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
            sql, result.transform, [result.x_role, result.y_role], []
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


def test_a_ninety_degree_turn_about_the_origin_is_exact(
    dialog: SeriesGeometryDialog, repo: SqliteRepo
) -> None:
    """(x, y) -> (-y, x). Worth stating in coordinates, not only as a matrix."""
    _configure(dialog, ROTATE, angle=90.0, use_data_centre=False, cx=0.0, cy=0.0)
    result = dialog.compute_results()[0]
    rows = _sql_result(repo, result.sql)

    np.testing.assert_allclose(rows["x"].to_numpy(), -SQUARE["y"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(rows["y"].to_numpy(), SQUARE["x"].to_numpy(), atol=1e-9)


def test_every_other_column_and_role_survives(
    qapp, repo: SqliteRepo, tmp_db_path: Path
) -> None:
    """A scatter coloured and sized by columns must still be, after moving.

    Two halves, and the second is the one that was wrong: the generated
    query carries the extra columns through, but the series descriptor has
    to keep naming them, or the rows arrive complete and the chart draws
    none of it - flat colour beside a source coloured by value.
    """
    frame = SQUARE.copy()
    frame["temperature"] = [10.0, 20.0, 30.0, 40.0, 50.0]
    frame["weight"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    frame["note"] = ["a", "b", "c", "d", "e"]
    repo.import_dataframe(frame, table_name="rich", normalize_columns=False)

    figure_id = repo.create_figure_descriptor(name="Rich")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="rich", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="rich",
        sql_query='SELECT * FROM "rich"',
        roles={"x": "x", "y": "y", "color": "temperature", "size": "weight"},
        style={},
    )
    built = SeriesGeometryDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        built.series_selector.reload(select_all_series=True)
        _configure(built, ROTATE, angle=90.0, use_data_centre=False, cx=0.0, cy=0.0)
        result = built.compute_results()[0]

        rows = repo.query_df(result.sql)
        for column in ("temperature", "weight", "note"):
            assert column in rows.columns, f"{column} was dropped from the query"
        assert list(rows["temperature"]) == [10.0, 20.0, 30.0, 40.0, 50.0]
        assert list(rows["note"]) == ["a", "b", "c", "d", "e"]

        spec = built.result_series_spec(axis_id, "", result)
        assert spec.roles["color"] == "temperature"
        assert spec.roles["size"] == "weight"
        assert spec.roles["x"] == "x"
        assert spec.roles["y"] == "y"
    finally:
        built.close()
        applogger.set_status_bar(None)


def test_an_awkward_column_name_does_not_break_the_query(
    qapp, repo: SqliteRepo, tmp_db_path: Path
) -> None:
    """A source query may return a column whose name contains a quote.

    ``SELECT "x"+"y"`` with no alias gives SQLite a column called
    ``x"+"y``. Wrapping that in quotes without doubling the inner ones
    closes the identifier early, and the generated query is a syntax
    error - which is the sort of thing that only shows up on somebody
    else's data.
    """
    figure_id = repo.create_figure_descriptor(name="Awkward")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="square", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="awkward",
        sql_query='SELECT "x", "y", "x"+"y", "y" AS "a space" FROM "square"',
        roles={"x": "x", "y": "y"},
        style={},
    )
    built = SeriesGeometryDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        built.series_selector.reload(select_all_series=True)
        _configure(built, ROTATE, angle=45.0)
        result = built.compute_results()[0]

        rows = repo.query_df(result.sql)

        assert '"x"+"y"' in rows.columns
        assert "a space" in rows.columns
        assert len(rows) == len(SQUARE)
    finally:
        built.close()
        applogger.set_status_bar(None)


CUBE = pd.DataFrame(
    {
        "x": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.3],
        "y": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.6],
        "z": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.8],
        "label": list("abcdefghi"),
    }
)


@pytest.fixture
def dialog_3d(qapp, repo: SqliteRepo):
    repo.import_dataframe(CUBE, table_name="cube", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="Geometry 3D")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot (3D)",
        title="cube", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="cube",
        sql_query='SELECT * FROM "cube"', roles={"x": "x", "y": "y", "z": "z"}, style={},
    )
    built = SeriesGeometryDialog(repo=repo, figure_id=figure_id, parent=None)
    built.series_selector.reload(select_all_series=True)
    yield built
    built.close()
    applogger.set_status_bar(None)


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (ROTATE, {"angle_x": 20.0, "angle_y": -35.0, "angle_z": 50.0, "use_data_centre": True}),
        (ROTO_TRANSLATE, {"angle_x": 90.0, "angle_y": 0.0, "angle_z": 30.0, "dx": 1.0, "dy": 2.0, "dz": -3.0, "use_data_centre": False, "cx": 0.0, "cy": 0.0, "cz": 0.0}),
        (TRANSLATE, {"dx": 1.0, "dy": 2.0, "dz": 3.0}),
        (SCALE, {"sx": 2.0, "sy": 1.0, "sz": 0.5, "use_data_centre": True}),
    ],
)
def test_3d_the_database_computes_what_the_preview_drew(
    dialog_3d: SeriesGeometryDialog, repo: SqliteRepo, model: str, values: dict
) -> None:
    _configure(dialog_3d, model, **values)
    result = dialog_3d.compute_results()[0]
    assert result.is_3d
    rows = repo.query_df(result.sql)
    assert list(rows.columns)[:4] == ["x", "y", "z", "label"]  # z moved, label carried through
    np.testing.assert_allclose(rows["x"].to_numpy(), result.after_x, atol=1e-9)
    np.testing.assert_allclose(rows["y"].to_numpy(), result.after_y, atol=1e-9)
    np.testing.assert_allclose(rows["z"].to_numpy(), result.after_z, atol=1e-9)
    assert dialog_3d.result_series_spec(0, "", result).roles["z"] == "z"


def test_a_3d_series_shows_three_angles_and_a_2d_one_does_not(
    dialog: SeriesGeometryDialog, dialog_3d: SeriesGeometryDialog
) -> None:
    def visible(d: SeriesGeometryDialog) -> set[str]:
        form = d._parameter_form_spec
        form.refresh_visibility()
        return {name for name, widget in form._widgets.items() if not widget.isHidden()}

    _configure(dialog_3d, ROTO_TRANSLATE)
    shown_3d = visible(dialog_3d)
    assert {"angle_x", "angle_y", "angle_z", "dx", "dy", "dz", "cz"} <= shown_3d
    assert "angle" not in shown_3d

    _configure(dialog, ROTO_TRANSLATE)
    shown_2d = visible(dialog)
    assert {"angle", "dx", "dy", "cx", "cy"} <= shown_2d
    assert not shown_2d & {"angle_x", "angle_y", "angle_z", "dz", "cz"}


def test_translated_model_names_still_show_their_parameters(
    dialog: SeriesGeometryDialog,
) -> None:
    """The bug: in Italian the combo said "Ruota", the rules said "Rotate",
    and every model-dependent parameter was hidden."""
    combo = dialog.model_combo
    index = combo.findData(ROTATE)
    combo.setItemText(index, "Ruota")
    combo.setCurrentIndex(index)
    form = dialog._parameter_form_spec
    form.refresh_visibility()
    assert not form._widgets["angle"].isHidden()
