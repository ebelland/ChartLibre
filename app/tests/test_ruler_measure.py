"""Ruler / measure mode from the chart's own context menu (todo.txt N-20).

Two right-clicks, not a drag: "Measure from here" arms the ruler at a point
(a transient marker only), "Measure to here" only appears once one is armed
on the *same* axes, and finishing writes the measurement to the axis'
"measurements" - the same storage a reference line or an annotation gets
(BaseAxisRenderer.apply_measurement draws it back) - rather than drawing a
transient artist: a measurement worth taking is worth it still being there
next time the chart opens, and worth deleting from the Overlay panel's
Measurements tab the same way a line is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QPoint

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel


@pytest.fixture
def two_axis_panel(qapp, repo: SqliteRepo):
    """One figure, two side-by-side axes, so a cross-axes measurement can be
    attempted and refused."""
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=2))
    axis_ids = []
    for index in range(2):
        axis_id = int(
            repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=index, chart_type="Scatter Plot",
                title=f"axis {index}", x_label="x", y_label="y", options={},
            )
        )
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=0, name="s",
            sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
        )
        axis_ids.append(axis_id)

    built = ChartPanel(repo, figure_id)
    built.resize(800, 400)
    built.show()
    qapp.processEvents()
    yield built, axis_ids
    built.close()
    repo.undo_store.discard_file()


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )
    built = ChartPanel(repo, figure_id)
    built.resize(600, 400)
    built.show()
    qapp.processEvents()
    yield built, axis_id
    built.close()
    repo.undo_store.discard_file()


def _centre(panel: ChartPanel) -> QPoint:
    canvas = panel._canvas
    return QPoint(canvas.width() // 2, canvas.height() // 2)


def _stored_measurements(repo: SqliteRepo, axis_id: int) -> list:
    return list((repo.get_axis_options(axis_id) or {}).get("measurements") or [])


# ----------------------------------------------------------------------
# The menu
# ----------------------------------------------------------------------
def test_measure_from_here_is_always_offered_inside_an_axes(panel) -> None:
    built, _axis_id = panel
    texts = [action.text() for action in built.context_menu_for(_centre(built)).actions()]
    assert "Measure from here" in texts


def test_measure_to_here_only_appears_once_a_point_is_pending(panel) -> None:
    built, axis_id = panel
    found = built._axis_at(_centre(built))
    assert found is not None
    _axis_id, axes, x_value, y_value = found

    before = [action.text() for action in built.context_menu_for(_centre(built)).actions()]
    assert "Measure to here" not in before

    built._measure_from_here(axes, axis_id, x_value, y_value)

    after = [action.text() for action in built.context_menu_for(_centre(built)).actions()]
    assert "Measure to here" in after


def test_cancel_measurement_only_appears_while_a_point_is_pending(panel) -> None:
    built, axis_id = panel
    found = built._axis_at(_centre(built))
    assert found is not None
    _axis_id, axes, x_value, y_value = found

    before = [action.text() for action in built.context_menu_for(_centre(built)).actions()]
    assert "Cancel measurement" not in before

    built._measure_from_here(axes, axis_id, x_value, y_value)
    during = [action.text() for action in built.context_menu_for(_centre(built)).actions()]
    assert "Cancel measurement" in during

    built._measure_to_here(axes, axis_id, x_value + 1.0, y_value + 1.0)
    after = [action.text() for action in built.context_menu_for(_centre(built)).actions()]
    assert "Cancel measurement" not in after


# ----------------------------------------------------------------------
# The pending point
# ----------------------------------------------------------------------
def test_measure_from_here_draws_a_pending_marker(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]

    built._measure_from_here(axes, axis_id, 2.0, 3.0)

    assert len(built._ruler_artists) == 1
    assert built._ruler_start == (axis_id, 2.0, 3.0)


def test_starting_a_new_pending_point_replaces_the_old_one(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]

    built._measure_from_here(axes, axis_id, 0.0, 0.0)
    built._measure_from_here(axes, axis_id, 5.0, 5.0)

    assert len(built._ruler_artists) == 1
    assert built._ruler_start == (axis_id, 5.0, 5.0)


def test_cancel_measurement_removes_the_pending_marker(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]
    built._measure_from_here(axes, axis_id, 0.0, 0.0)

    built._clear_ruler()

    assert built._ruler_artists == []
    assert built._ruler_start is None


def test_reloading_the_panel_forgets_the_pending_point_without_erroring(panel) -> None:
    """reload() clears the whole figure; the marker artist goes with it and
    must not be touched again afterwards."""
    built, axis_id = panel
    axes = built._figure.axes[0]
    built._measure_from_here(axes, axis_id, 0.0, 0.0)

    built.reload()

    assert built._ruler_start is None
    assert built._ruler_artists == []


# ----------------------------------------------------------------------
# The finished measurement: stored, not transient
# ----------------------------------------------------------------------
def test_finishing_a_measurement_stores_it_on_the_axis(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]

    built._measure_from_here(axes, axis_id, 0.0, 0.0)
    built._measure_to_here(axes, axis_id, 4.0, 2.0)

    assert built._ruler_start is None
    assert built._ruler_artists == []
    assert _stored_measurements(built._repo, axis_id) == [
        {"x0": 0.0, "y0": 0.0, "x1": 4.0, "y1": 2.0, "kwargs": {}}
    ]


def test_a_second_measurement_joins_the_first(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]

    built._measure_from_here(axes, axis_id, 0.0, 0.0)
    built._measure_to_here(axes, axis_id, 1.0, 1.0)
    built._measure_from_here(axes, axis_id, 2.0, 2.0)
    built._measure_to_here(axes, axis_id, 3.0, 3.0)

    assert len(_stored_measurements(built._repo, axis_id)) == 2


def test_the_measurement_is_drawn_after_it_is_added(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]

    built._measure_from_here(axes, axis_id, 0.0, 0.0)
    built._measure_to_here(axes, axis_id, 4.0, 2.0)

    axes = built._figure.axes[0]  # reload() rebuilt the axes
    assert any(
        list(line.get_xdata()) == [0.0, 4.0] and list(line.get_ydata()) == [0.0, 2.0]
        for line in axes.lines
    ), "the reload did not draw it"


def test_finishing_a_measurement_can_be_undone(panel) -> None:
    """It writes to the axis descriptor like any other edit, so it takes the
    same route back."""
    built, axis_id = panel
    axes = built._figure.axes[0]
    built._measure_from_here(axes, axis_id, 0.0, 0.0)
    built._measure_to_here(axes, axis_id, 1.0, 1.0)

    assert [entry.label for entry in built._repo.undo_entries()] == [
        "Add measurement"
    ]

    built._repo.undo_last()

    assert _stored_measurements(built._repo, axis_id) == []


def test_the_measurement_is_the_same_key_the_overlay_panel_edits(panel) -> None:
    """One storage, two ways in: dropped from the chart, edited in the panel."""
    from app.widgets.overlay_properties import OverlayPropertiesWidget

    built, axis_id = panel
    axes = built._figure.axes[0]
    built._measure_from_here(axes, axis_id, 0.0, 0.0)
    built._measure_to_here(axes, axis_id, 4.0, 2.0)

    widget = OverlayPropertiesWidget()
    widget.set_connected_figure(built._repo, built._figure_id, built._figure)
    widget.set_axis(axis_id)

    assert widget._measurements_table.rowCount() == 1
    assert widget._measurements_table.item(0, 2).text() == "4.0"  # End X


# ----------------------------------------------------------------------
# Two axes: a measurement does not cross between them
# ----------------------------------------------------------------------
def test_measuring_across_two_axes_is_refused(two_axis_panel) -> None:
    built, axis_ids = two_axis_panel
    axes_a, axes_b = built._figure.axes[0], built._figure.axes[1]

    built._measure_from_here(axes_a, axis_ids[0], 0.0, 0.0)
    built._measure_to_here(axes_b, axis_ids[1], 1.0, 1.0)

    # Refused: the pending point on axis A is still there, nothing stored.
    assert built._ruler_start == (axis_ids[0], 0.0, 0.0)
    assert len(built._ruler_artists) == 1
    assert _stored_measurements(built._repo, axis_ids[0]) == []
    assert _stored_measurements(built._repo, axis_ids[1]) == []
