"""Live updates: redraw a chart on a timer (todo.txt N-20, last item).

A flat poll rather than a SQLite change hook - see LIVE_REFRESH_INTERVAL_MS's
own docstring for why - so what this actually tests is the timer wiring and
the guard that skips a tick while the user is mid-gesture on the chart
(dragging an annotation, dragging a selection rectangle, or with a ruler
point pending), rather than any change-detection logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(5.0), "y": np.arange(5.0)}),
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
    yield built
    built.close()
    repo.undo_store.discard_file()


def test_off_by_default(panel) -> None:
    assert panel._live_updates_enabled is False
    assert not panel._live_timer.isActive()


def test_turning_it_on_starts_the_timer(panel) -> None:
    panel._on_toggle_live_updates(True)

    assert panel._live_updates_enabled is True
    assert panel._live_timer.isActive()


def test_turning_it_off_stops_the_timer(panel) -> None:
    panel._on_toggle_live_updates(True)

    panel._on_toggle_live_updates(False)

    assert panel._live_updates_enabled is False
    assert not panel._live_timer.isActive()


def test_a_tick_redraws_the_chart(panel) -> None:
    reloaded = []
    panel.reload = lambda: reloaded.append(True)  # type: ignore[method-assign]

    panel._on_live_timer_tick()

    assert reloaded == [True]


def test_a_tick_is_skipped_while_dragging_an_annotation(panel) -> None:
    reloaded = []
    panel.reload = lambda: reloaded.append(True)  # type: ignore[method-assign]
    panel._dragging_annotation = (1, 0, object(), panel._figure.axes[0])

    panel._on_live_timer_tick()

    assert reloaded == []


def test_a_tick_is_skipped_mid_box_selection(panel) -> None:
    reloaded = []
    panel.reload = lambda: reloaded.append(True)  # type: ignore[method-assign]
    panel._selection_start = (panel._figure.axes[0], 1.0, 1.0)

    panel._on_live_timer_tick()

    assert reloaded == []


def test_a_tick_is_skipped_with_a_pending_ruler_point(panel) -> None:
    reloaded = []
    panel.reload = lambda: reloaded.append(True)  # type: ignore[method-assign]
    panel._ruler_start = (1, 0.0, 0.0)

    panel._on_live_timer_tick()

    assert reloaded == []


def test_closing_the_panel_stops_the_timer(panel) -> None:
    panel._on_toggle_live_updates(True)

    panel.close()

    assert not panel._live_timer.isActive()
