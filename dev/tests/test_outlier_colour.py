"""Outliers: colour them instead of hiding them.

The rows found are written into a colour column of the source table and the
series' query is made to expose it as "color", which the scatter renderer
draws per point. A series that already has a colour is left alone, with a
message; and Cancel puts everything back.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.colors import to_rgba as _to_rgba  # noqa: E402

import app.series_operations.outlier_dialog as outlier_module  # noqa: E402
from app.charts.base import SeriesData  # noqa: E402
from app.charts.scatter import ScatterAxisRenderer  # noqa: E402
from app.data.series_frame import SeriesFrame  # noqa: E402
from app.data.select_sql import has_projection_alias, sql_insert_select_expression  # noqa: E402
from app.data.sqlite_repo import SqliteRepo  # noqa: E402
from app.series_operations.outlier_dialog import (  # noqa: E402
    ACTION_COLOUR,
    ACTION_HIDE,
    COLOUR_COLUMN,
    SeriesOutlierDialog,
)
from app.series_operations.parameter_form import ParameterForm  # noqa: E402
from app.series_operations.parameter_spec import ColorParam  # noqa: E402
from app.utils.dialog_state import clear_state  # noqa: E402

RED = "#d62728"


def _series_frame(frame: pd.DataFrame) -> SeriesFrame:
    """A query result as the columns of arrays a renderer reads."""
    return SeriesFrame({name: frame[name].to_numpy() for name in frame.columns})


def to_rgba(colour: str) -> tuple[float, float, float, float]:
    """The colour as RGBA (the matplotlib stubs type its argument as a masked array)."""
    return _to_rgba(cast(Any, colour))


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state() -> Iterator[None]:
    clear_state("SeriesOutlierDialog")
    yield
    clear_state("SeriesOutlierDialog")


def _figure(repo: SqliteRepo, sql: str = "SELECT n AS x, v AS y FROM t", roles: dict | None = None) -> tuple[int, int]:
    rng = np.random.default_rng(1)
    values = rng.normal(10.0, 0.5, 60)
    values[[10, 40]] = [60.0, -40.0]  # rowids 11 and 41
    repo.import_dataframe(pd.DataFrame({"n": np.arange(60.0), "v": values}), table_name="t", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot", title="s", x_label="n", y_label="v", options={},
    ))
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig", sql_query=sql,
        roles=roles or {"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


def _dialog(repo: SqliteRepo, figure_id: int, action: str, colour: str = RED) -> SeriesOutlierDialog:
    dialog = SeriesOutlierDialog(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    dialog._parameter_form_spec.set_values({"action": action, "colour": colour})
    return dialog


@pytest.fixture
def told(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    messages: list[tuple[str, dict]] = []
    monkeypatch.setattr(outlier_module, "show_message", lambda parent, key, **kw: messages.append((key, kw)))
    return messages


def _series_sql(repo: SqliteRepo, axis_id: int) -> str:
    return str(dict((repo.get_series(axis_id) or [])[0])["sql_query"])


def test_colouring_marks_exactly_the_outliers_and_hides_nothing(qapp, repo: SqliteRepo) -> None:
    figure_id, axis_id = _figure(repo)
    dialog = _dialog(repo, figure_id, ACTION_COLOUR)
    try:
        assert dialog.preview() is True
        dialog.ok()  # keep it: closing without OK takes a preview back
    finally:
        dialog.close()

    coloured = repo.query_df(f'SELECT rowid AS r, "{COLOUR_COLUMN}" AS c FROM t WHERE "{COLOUR_COLUMN}" IS NOT NULL')
    assert dict(zip(coloured["r"], coloured["c"])) == {11: RED, 41: RED}
    assert repo.hidden_rowids("t") == []  # nothing was hidden
    sql = _series_sql(repo, axis_id)
    assert has_projection_alias(sql, "color") and f'"{COLOUR_COLUMN}" AS "color"' in sql
    assert len(repo.query_df(sql)) == 60  # every point is still on the chart


def test_the_series_query_then_returns_a_colour_column_the_chart_can_draw(qapp, repo: SqliteRepo) -> None:
    figure_id, axis_id = _figure(repo)
    dialog = _dialog(repo, figure_id, ACTION_COLOUR, "#1f77b4")
    try:
        dialog.preview()
        dialog.ok()
    finally:
        dialog.close()
    frame = repo.query_df(_series_sql(repo, axis_id))
    assert list(frame.columns) == ["x", "y", "color"]
    assert frame["color"].notna().sum() == 2 and frame["color"].dropna().unique().tolist() == ["#1f77b4"]

    axes = Figure().add_subplot()
    ScatterAxisRenderer().render_axis(axes, [SeriesData(name="sig", df=_series_frame(frame), style={"color": "#888888"})], {})
    faces = axes.collections[0].get_facecolors()
    assert len(faces) == 60
    np.testing.assert_allclose(faces[10][:3], to_rgba("#1f77b4")[:3])
    np.testing.assert_allclose(faces[0][:3], to_rgba("#888888")[:3])  # everyone else keeps the series' colour


def test_a_series_that_already_has_a_colour_is_left_alone_with_a_message(
    qapp, repo: SqliteRepo, told: list[tuple[str, dict]]
) -> None:
    figure_id, axis_id = _figure(repo, sql='SELECT n AS x, v AS y, n AS "color" FROM t')
    before = _series_sql(repo, axis_id)
    dialog = _dialog(repo, figure_id, ACTION_COLOUR)
    try:
        assert dialog.preview() is False
    finally:
        dialog.close()

    assert told and told[-1][0] == "outliers.colour_present" and told[-1][1]["series"] == "sig"
    assert _series_sql(repo, axis_id) == before
    assert COLOUR_COLUMN not in repo.get_columns("t")


def test_hiding_is_still_the_default_and_still_works(qapp, repo: SqliteRepo) -> None:
    figure_id, axis_id = _figure(repo)
    dialog = _dialog(repo, figure_id, ACTION_HIDE)
    try:
        assert dialog.parameter_values()["action"] == ACTION_HIDE
        assert dialog.preview() is True
        dialog.ok()
    finally:
        dialog.close()
    assert repo.hidden_rowids("t") == [11, 41]
    assert COLOUR_COLUMN not in repo.get_columns("t")
    assert 'Hide' in _series_sql(repo, axis_id)


def test_previewing_again_does_not_stack_colours_or_projections(qapp, repo: SqliteRepo) -> None:
    figure_id, axis_id = _figure(repo)
    dialog = _dialog(repo, figure_id, ACTION_COLOUR)
    try:
        assert dialog.preview() and dialog.preview() and dialog.preview()
        dialog.ok()
    finally:
        dialog.close()
    assert _series_sql(repo, axis_id).lower().count(" as \"color\"") == 1
    assert repo.query_df(f'SELECT COUNT(*) AS n FROM t WHERE "{COLOUR_COLUMN}" IS NOT NULL')["n"][0] == 2


def test_cancel_puts_back_the_query_and_removes_the_column(qapp, repo: SqliteRepo) -> None:
    figure_id, axis_id = _figure(repo)
    before = _series_sql(repo, axis_id)
    dialog = _dialog(repo, figure_id, ACTION_COLOUR)
    try:
        dialog.preview()
        assert _series_sql(repo, axis_id) != before
        dialog.cancel_operation_changes(refresh=False)
    finally:
        dialog.close()
    assert _series_sql(repo, axis_id) == before
    assert COLOUR_COLUMN not in repo.get_columns("t")


def test_a_new_run_may_replace_its_own_earlier_colouring_and_cancel_restores_it(qapp, repo: SqliteRepo) -> None:
    figure_id, axis_id = _figure(repo)
    first = _dialog(repo, figure_id, ACTION_COLOUR, RED)
    try:
        first.preview()
        first.ok()
    finally:
        first.close()
    kept = repo.colour_snapshot("t", COLOUR_COLUMN)
    assert kept == {11: RED, 41: RED}

    second = _dialog(repo, figure_id, ACTION_COLOUR, "#2ca02c")
    try:
        assert second.preview() is True
        snapshot = repo.colour_snapshot("t", COLOUR_COLUMN)
        assert snapshot is not None and set(snapshot.values()) == {"#2ca02c"}
        second.cancel_operation_changes(refresh=False)
    finally:
        second.close()
    assert repo.colour_snapshot("t", COLOUR_COLUMN) == kept  # the first colouring is back


# ----------------------------------------------------------------------
# The pieces
# ----------------------------------------------------------------------
def test_a_text_column_is_really_added(repo: SqliteRepo) -> None:
    """ensure_column used to run an empty statement for any type but INTEGER."""
    repo.import_dataframe(pd.DataFrame({"a": [1, 2]}), table_name="c", normalize_columns=False)
    repo.ensure_column("c", "label", "TEXT")
    repo.ensure_column("c", "flag", "INTEGER")
    columns = repo.get_columns("c")
    assert "label" in columns and "flag" in columns
    assert repo.query_df('SELECT flag FROM c')["flag"].tolist() == [0, 0]  # integers start at 0
    assert repo.query_df('SELECT label FROM c')["label"].isna().all()  # text starts empty


def test_the_select_list_gets_a_projection_once() -> None:
    sql = "SELECT n AS x, v AS y FROM t WHERE n > 3"
    once = sql_insert_select_expression(sql, '"OutlierColor" AS "color"')
    assert once == 'SELECT n AS x, v AS y, "OutlierColor" AS "color" FROM t WHERE n > 3'
    assert sql_insert_select_expression(once, '"OutlierColor" AS "color"') == once
    assert has_projection_alias(once, "color") and has_projection_alias(once, "COLOR")
    assert not has_projection_alias(sql, "color")
    assert not has_projection_alias("SELECT n AS x, round(v, 2) AS y FROM t WHERE 'a' = 'color'", "color")


def test_a_colour_parameter_is_read_and_restored_as_hex(qapp) -> None:
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    form = ParameterForm([ColorParam("colour", "Colour:", default_value=RED)], host)
    assert form.values() == {"colour": RED}
    form.set_values({"colour": "#2ca02c"})
    assert form.values() == {"colour": "#2ca02c"}
    host.close()


def test_a_scatter_colour_column_with_empty_cells_is_not_a_category(qapp) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "color": ["#ff0000", None, ""]})
    axes = Figure().add_subplot()
    ScatterAxisRenderer().render_axis(axes, [SeriesData(name="s", df=_series_frame(frame), style={"color": "#00ff00"})], {})
    faces = axes.collections[0].get_facecolors()
    # (compared without the alpha: a scatter is drawn slightly translucent)
    np.testing.assert_allclose(faces[0][:3], to_rgba("#ff0000")[:3])
    np.testing.assert_allclose(faces[1][:3], to_rgba("#00ff00")[:3])
    np.testing.assert_allclose(faces[2][:3], to_rgba("#00ff00")[:3])


def test_after_ok_the_series_no_longer_returns_the_hidden_rows(qapp, repo: SqliteRepo) -> None:
    """What the user sees: hide, press OK, and the chart's query gives 58 rows, not 60."""
    figure_id, axis_id = _figure(repo)
    dialog = _dialog(repo, figure_id, ACTION_HIDE)
    try:
        dialog.preview()
        dialog.ok()
    finally:
        dialog.close()
    assert len(repo.query_df(_series_sql(repo, axis_id))) == 58
