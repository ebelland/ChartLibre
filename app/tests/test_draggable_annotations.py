"""Dragging an axis annotation to reposition it (todo.txt N-20).

BaseAxisRenderer.apply_annotation tags the artist it draws with
``_dhub_annotation_index`` so ChartPanel's drag handlers can tell a real
axis annotation apart from the hover readout or a measurement's label -
none of those carry that tag, so a click on one never starts a drag. The
one non-obvious behaviour worth pinning beyond "it moves": what gets
written back is the *stored* x/y for both plain-text and arrow
annotations, even though an arrow's visible text sits at a ``xytext``
offset from that point rather than on it.
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
class _FakeEvent:
    inaxes: Any
    xdata: float | None
    ydata: float | None
    x: float = 0.0
    y: float = 0.0
    button: int = 1


def _annotation_pixel_center(panel: ChartPanel, axes: Any, index: int) -> tuple[float, float]:
    """Return the screen-pixel centre of the tagged annotation *index*."""
    artist = next(
        text for text in axes.texts if getattr(text, "_dhub_annotation_index", None) == index
    )
    bbox = artist.get_window_extent(panel._canvas.get_renderer())
    return bbox.x0 + bbox.width / 2.0, bbox.y0 + bbox.height / 2.0


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
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="t",
            x_label="x",
            y_label="y",
            options={
                "annotations": [
                    {"x": 5.0, "y": 5.0, "type": "text", "text": "hello", "kwargs": {}}
                ]
            },
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


def _stored_annotations(repo: SqliteRepo, axis_id: int) -> list:
    return list((repo.get_axis_options(axis_id) or {}).get("annotations") or [])


# ----------------------------------------------------------------------
# Starting a drag
# ----------------------------------------------------------------------
def test_pressing_on_the_annotation_starts_a_drag(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]
    px, py = _annotation_pixel_center(built, axes, 0)

    built._on_annotation_drag_press(_FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0, x=px, y=py))

    assert built._dragging_annotation is not None
    dragging_axis_id, index, _artist, dragging_axes = built._dragging_annotation
    assert dragging_axis_id == axis_id
    assert index == 0
    assert dragging_axes is axes


def test_pressing_on_empty_space_starts_nothing(panel) -> None:
    built, _axis_id = panel
    axes = built._figure.axes[0]

    built._on_annotation_drag_press(
        _FakeEvent(inaxes=axes, xdata=15.0, ydata=15.0, x=1.0, y=1.0)
    )

    assert built._dragging_annotation is None


def test_a_right_click_never_starts_a_drag(panel) -> None:
    built, _axis_id = panel
    axes = built._figure.axes[0]
    px, py = _annotation_pixel_center(built, axes, 0)

    built._on_annotation_drag_press(
        _FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0, x=px, y=py, button=3)
    )

    assert built._dragging_annotation is None


def test_no_drag_starts_while_a_navigation_tool_is_active(panel) -> None:
    """Pan/zoom clicks belong to the Matplotlib toolbar, not to this."""
    built, _axis_id = panel
    axes = built._figure.axes[0]
    px, py = _annotation_pixel_center(built, axes, 0)
    built._toolbar.pan()  # arms the toolbar's own pan mode
    try:
        built._on_annotation_drag_press(
            _FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0, x=px, y=py)
        )
        assert built._dragging_annotation is None
    finally:
        built._toolbar.pan()  # toggle back off


# ----------------------------------------------------------------------
# Following the pointer, and finishing
# ----------------------------------------------------------------------
def test_finishing_a_drag_stores_the_new_position(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]
    px, py = _annotation_pixel_center(built, axes, 0)

    built._on_annotation_drag_press(_FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0, x=px, y=py))
    built._on_annotation_drag_motion(_FakeEvent(inaxes=axes, xdata=8.0, ydata=2.0))
    built._on_annotation_drag_release(_FakeEvent(inaxes=axes, xdata=8.0, ydata=2.0))

    assert built._dragging_annotation is None
    stored = _stored_annotations(built._repo, axis_id)
    assert stored[0]["x"] == 8.0
    assert stored[0]["y"] == 2.0
    assert stored[0]["text"] == "hello"  # everything else survives untouched


def test_dropping_outside_every_axes_snaps_back(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]
    px, py = _annotation_pixel_center(built, axes, 0)

    built._on_annotation_drag_press(_FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0, x=px, y=py))
    built._on_annotation_drag_release(_FakeEvent(inaxes=None, xdata=None, ydata=None))

    assert built._dragging_annotation is None
    assert _stored_annotations(built._repo, axis_id)[0]["x"] == 5.0


def test_finishing_a_drag_can_be_undone(panel) -> None:
    built, axis_id = panel
    axes = built._figure.axes[0]
    px, py = _annotation_pixel_center(built, axes, 0)

    built._on_annotation_drag_press(_FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0, x=px, y=py))
    built._on_annotation_drag_release(_FakeEvent(inaxes=axes, xdata=9.0, ydata=9.0))

    assert [entry.label for entry in built._repo.undo_entries()] == ["Move annotation"]

    built._repo.undo_last()

    assert _stored_annotations(built._repo, axis_id)[0]["x"] == 5.0


def test_reloading_mid_drag_forgets_it_without_erroring(panel) -> None:
    built, _axis_id = panel
    axes = built._figure.axes[0]
    px, py = _annotation_pixel_center(built, axes, 0)
    built._on_annotation_drag_press(_FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0, x=px, y=py))

    built.reload()

    assert built._dragging_annotation is None


# ----------------------------------------------------------------------
# Only tagged annotation artists are draggable
# ----------------------------------------------------------------------
def test_the_hover_readout_is_never_draggable(panel) -> None:
    """The hover annotation is a plain Text/Annotation too, but never
    tagged with _dhub_annotation_index - _annotation_artist_at must not
    pick it up even if the pointer happens to land on it."""
    built, _axis_id = panel
    axes = built._figure.axes[0]
    annotation = built._hover_annotation_for(axes)
    annotation.xy = (5.0, 5.0)
    annotation.set_visible(True)

    hit = built._annotation_artist_at(axes, _FakeEvent(inaxes=axes, xdata=5.0, ydata=5.0))
    assert hit is None or hit[1] is not annotation
