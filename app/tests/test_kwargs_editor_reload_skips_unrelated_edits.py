"""Reordering a series must not rebuild the selected axis's kwargs editor.

``MainWindow._reload_property_widgets`` runs on every series reorder, axis
move and layout-preset apply - none of which touch the selected axis's own
renderer or kwargs - yet it always finished with
``self._axis_widget.rebuild_kwargs_editor(current_axis_id)``. Before
``AxisPropertiesWidget.rebuild_kwargs_editor`` learned to skip when nothing
about the axis actually changed, every one of those actions tore down and
rebuilt the whole kwargs ``DictEditorPanel`` - a QTreeWidget with one row per
kwarg - for no visible change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger


@dataclass
class _Ids:
    axis_id: int
    series_ids: list[int]


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(
        pd.DataFrame({"x": np.arange(5), "y": np.arange(5) ** 2}),
        table_name="src",
        normalize_columns=False,
    )
    yield built
    built.close()


@pytest.fixture
def ids(repo: SqliteRepo) -> _Ids:
    figure_id = repo.create_figure_descriptor(name="fig")
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="t",
            x_label="",
            y_label="",
            options={},
        )
    )
    series_ids = [
        int(
            repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=index,
                name=f"s{index}",
                sql_query="SELECT x, y FROM src",
                roles={"x": "x", "y": "y"},
                style={},
            )
        )
        for index in range(2)
    ]
    return _Ids(axis_id=axis_id, series_ids=series_ids)


@pytest.fixture
def window(qapp, repo: SqliteRepo, ids: _Ids, tmp_db_path: Path):
    # Depends on `ids` (unused by name) so the figure/axis/series exist
    # before MainWindow loads its chart tabs from the repo - fixture
    # dependency order is by the graph, not by parameter position, and
    # `window` otherwise has no edge forcing it to run after `ids`.
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def test_reordering_series_reuses_the_kwargs_editor(
    window: MainWindow, ids: _Ids
) -> None:
    window._axis_widget.rebuild_kwargs_editor(ids.axis_id)
    editor_before = window._axis_widget._kwargs_editor
    assert editor_before is not None

    window._series_widget.set_current_axis_id(ids.axis_id)
    window._on_series_order_requested(list(reversed(ids.series_ids)))

    assert window._axis_widget._kwargs_editor is editor_before


def test_a_real_kwarg_change_still_rebuilds(
    window: MainWindow, repo: SqliteRepo, ids: _Ids
) -> None:
    window._axis_widget.rebuild_kwargs_editor(ids.axis_id)
    editor_before = window._axis_widget._kwargs_editor
    assert editor_before is not None

    a_key = next(iter(editor_before.config))
    options = dict(repo.get_axis_options(ids.axis_id) or {})
    options["axis_kwargs"] = {a_key: "a value nothing defaults to"}
    repo.set_axis_options(ids.axis_id, options)

    window._axis_widget.rebuild_kwargs_editor(ids.axis_id)

    assert window._axis_widget._kwargs_editor is not editor_before
