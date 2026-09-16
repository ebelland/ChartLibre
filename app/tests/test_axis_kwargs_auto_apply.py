"""Editing a kwargs row alone must queue an auto-apply.

The Apply button was dropped from AxisPropertiesWidget in favour of
auto-apply (see _connect_auto_apply), which wires every static form control
to _queue_auto_apply. The renderer kwargs editor is not a static control -
it is rebuilt from scratch per axis/renderer in rebuild_kwargs_editor - and
that rebuild never connected DictEditorPanel.valuesChanged to
_queue_auto_apply. A kwargs-only edit therefore queued nothing: it was only
ever persisted as a side effect of touching some other field afterwards,
which called clean_kwargs() and picked up whatever the editor happened to
hold at that moment.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets.axis_properties import AxisPropertiesWidget


@pytest.fixture
def widget(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]}),
        table_name="t",
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
            options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
    )

    built = AxisPropertiesWidget()
    built.set_connected_figure(repo, figure_id, figure=None)
    built.rebuild_kwargs_editor(axis_id)
    return built


def test_editing_a_kwarg_alone_queues_an_auto_apply(widget: AxisPropertiesWidget) -> None:
    editor = widget._kwargs_editor
    assert editor is not None
    assert not widget._auto_apply_timer.isActive()

    a_key = next(iter(editor.config))
    editor.set_value_for_key(a_key, "a value nothing defaults to")

    assert widget._auto_apply_timer.isActive()


def test_running_the_queued_apply_emits_the_kwarg(
    widget: AxisPropertiesWidget,
) -> None:
    """The widget only emits axis_options_requested - MainWindow is what
    persists it (_on_axis_options_requested) - so this checks the payload
    the signal carries rather than the repo, which nothing here writes to."""
    editor = widget._kwargs_editor
    assert editor is not None

    a_key = next(iter(editor.config))
    editor.set_value_for_key(a_key, "a value nothing defaults to")
    assert widget._auto_apply_timer.isActive()

    payloads: list[dict] = []
    widget.axis_options_requested.connect(payloads.append)

    widget._auto_apply_timer.stop()
    widget._run_auto_apply()

    assert len(payloads) == 1
    assert widget.clean_kwargs()[a_key] == "a value nothing defaults to"


def test_resetting_to_defaults_also_queues_an_auto_apply(
    widget: AxisPropertiesWidget,
) -> None:
    """The reset button is just another edit to the live editor - it must
    reach auto-apply the same way typing a value does."""
    editor = widget._kwargs_editor
    assert editor is not None

    a_key = next(iter(editor.config))
    editor.set_value_for_key(a_key, "a value nothing defaults to")
    widget._auto_apply_timer.stop()

    widget._reset_kwargs_to_defaults()

    assert widget._auto_apply_timer.isActive()
