"""Crosshair and linked x zoom/pan across a figure's axes (todo.txt N-20).

Both are "act across every axes of this figure" toggles from the chart's
actions menu, and both are pinned here: the crosshair moves the same x on
every chart axes but only shows y on the one under the pointer (axes rarely
share a y scale), and linked zoom copies one axes' x range onto the others
without the two bouncing the change back and forth forever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel


@dataclass
class _MotionEvent:
    inaxes: Any
    xdata: float | None
    ydata: float | None
    # Pixel coordinates, for the paths (_update_hover_readout's hit test)
    # that need them alongside data coordinates. Off in a corner nothing
    # is plotted near, so a hit test on these fixtures' data always misses
    # and the crosshair-only paths are what get exercised.
    x: float = -1000.0
    y: float = -1000.0


@pytest.fixture
def two_axis_panel(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=2))
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

    built = ChartPanel(repo, figure_id)
    built.resize(800, 400)
    built.show()
    qapp.processEvents()
    yield built
    built.close()
    repo.undo_store.discard_file()


# ----------------------------------------------------------------------
# The menu
# ----------------------------------------------------------------------
def test_the_menu_offers_both_toggles_off_by_default(two_axis_panel) -> None:
    menu = two_axis_panel._build_actions_menu()
    by_text = {action.text(): action for action in menu.actions()}

    assert "Crosshair" in by_text
    assert "Link zoom/pan across axes" in by_text
    assert not by_text["Crosshair"].isChecked()
    assert not by_text["Link zoom/pan across axes"].isChecked()


def test_toggling_crosshair_on_is_reflected_in_a_freshly_built_menu(two_axis_panel) -> None:
    two_axis_panel._on_toggle_crosshair(True)

    menu = two_axis_panel._build_actions_menu()
    by_text = {action.text(): action for action in menu.actions()}
    assert by_text["Crosshair"].isChecked()


# ----------------------------------------------------------------------
# Crosshair drawing
# ----------------------------------------------------------------------
def test_the_vertical_line_is_shared_by_every_axes(two_axis_panel) -> None:
    built = two_axis_panel
    built._on_toggle_crosshair(True)
    axes_a, axes_b = built._figure.axes[0], built._figure.axes[1]

    built._update_crosshair(_MotionEvent(inaxes=axes_a, xdata=5.0, ydata=3.0))

    assert axes_a in built._crosshair_vlines
    assert axes_b in built._crosshair_vlines
    assert list(built._crosshair_vlines[axes_a].get_xdata()) == [5.0, 5.0]
    assert list(built._crosshair_vlines[axes_b].get_xdata()) == [5.0, 5.0]


def test_the_horizontal_line_is_only_on_the_hovered_axes(two_axis_panel) -> None:
    built = two_axis_panel
    built._on_toggle_crosshair(True)
    axes_a, axes_b = built._figure.axes[0], built._figure.axes[1]

    built._update_crosshair(_MotionEvent(inaxes=axes_a, xdata=5.0, ydata=3.0))

    assert axes_a in built._crosshair_hlines
    assert built._crosshair_hlines[axes_a].get_visible()
    assert axes_b not in built._crosshair_hlines


def test_moving_to_the_other_axes_moves_the_horizontal_line_too(two_axis_panel) -> None:
    built = two_axis_panel
    built._on_toggle_crosshair(True)
    axes_a, axes_b = built._figure.axes[0], built._figure.axes[1]

    built._update_crosshair(_MotionEvent(inaxes=axes_a, xdata=5.0, ydata=3.0))
    built._update_crosshair(_MotionEvent(inaxes=axes_b, xdata=7.0, ydata=9.0))

    assert not built._crosshair_hlines[axes_a].get_visible()
    assert built._crosshair_hlines[axes_b].get_visible()
    # Still shared: both vertical lines moved to the new x.
    assert list(built._crosshair_vlines[axes_a].get_xdata()) == [7.0, 7.0]
    assert list(built._crosshair_vlines[axes_b].get_xdata()) == [7.0, 7.0]


def test_the_disabled_flag_stops_the_hover_timer_from_ever_calling_it(
    two_axis_panel,
) -> None:
    """_update_crosshair itself does not gate on the flag -
    _update_hover_readout does, before ever calling it - so the flag is
    what stops the timer-driven path from reaching it at all."""
    built = two_axis_panel
    assert not built._crosshair_enabled

    built._hover_event = _MotionEvent(inaxes=built._figure.axes[0], xdata=5.0, ydata=3.0)
    built._update_hover_readout()

    assert built._crosshair_vlines == {}


def test_turning_the_crosshair_off_hides_its_lines(two_axis_panel) -> None:
    built = two_axis_panel
    built._on_toggle_crosshair(True)
    axes_a = built._figure.axes[0]
    built._update_crosshair(_MotionEvent(inaxes=axes_a, xdata=5.0, ydata=3.0))

    built._on_toggle_crosshair(False)

    assert not built._crosshair_vlines[axes_a].get_visible()
    assert not built._crosshair_hlines[axes_a].get_visible()


def test_reloading_forgets_the_crosshair_artists_without_erroring(two_axis_panel) -> None:
    built = two_axis_panel
    built._on_toggle_crosshair(True)
    axes_a = built._figure.axes[0]
    built._update_crosshair(_MotionEvent(inaxes=axes_a, xdata=5.0, ydata=3.0))

    built.reload()

    assert built._crosshair_vlines == {}
    assert built._crosshair_hlines == {}
    # The toggle itself survives a reload - it is a person's standing choice.
    assert built._crosshair_enabled is True


# ----------------------------------------------------------------------
# Linked zoom/pan
# ----------------------------------------------------------------------
def test_changing_one_axes_x_range_leaves_the_other_alone_when_unlinked(
    two_axis_panel,
) -> None:
    built = two_axis_panel
    axes_a, axes_b = built._figure.axes[0], built._figure.axes[1]
    original_b = axes_b.get_xlim()

    axes_a.set_xlim(2.0, 8.0)

    assert axes_b.get_xlim() == original_b


def test_changing_one_axes_x_range_copies_to_the_other_when_linked(
    two_axis_panel,
) -> None:
    built = two_axis_panel
    built._on_toggle_link_x_zoom(True)
    axes_a, axes_b = built._figure.axes[0], built._figure.axes[1]

    axes_a.set_xlim(2.0, 8.0)

    assert axes_b.get_xlim() == (2.0, 8.0)


def test_linking_does_not_bounce_forever_between_two_axes(two_axis_panel) -> None:
    """The re-entrancy guard is the whole point: without it, axes_b's own
    xlim_changed (fired by the sync itself) would try to push the change
    straight back onto axes_a."""
    built = two_axis_panel
    built._on_toggle_link_x_zoom(True)
    axes_a, _axes_b = built._figure.axes[0], built._figure.axes[1]

    axes_a.set_xlim(2.0, 8.0)  # must simply return, not recurse/hang

    assert built._syncing_x_limits is False
