"""Only the current chart tab renders when a project first opens.

MainWindow._reload_tabs() used to build a ChartPanel per figure and let each
one's own __init__ run a full render (query + matplotlib draw) immediately,
even though QTabWidget shows exactly one of them at a time. A project with
many figures paid for a render of every single one before the window became
interactive. ChartPanel now takes defer_render=True there and renders lazily
through ensure_rendered(), called from
MainWindow._update_properties_for_current_chart (every "the current chart is
now this panel" path) and from ChartPanel.showEvent as a backstop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger
from app.widgets.chart_panel import ChartPanel


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
    for name in ("fig one", "fig two", "fig three"):
        figure_id = built.create_figure_descriptor(name=name)
        axis_id = built.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title=name,
            x_label="",
            y_label="",
            options={},
        )
        built.create_series_descriptor(
            axis_id=axis_id,
            series_index=0,
            name="s",
            sql_query="SELECT x, y FROM src",
            roles={"x": "x", "y": "y"},
            style={},
        )
    yield built
    built.close()


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _panels(window: MainWindow) -> list[ChartPanel]:
    return [
        window._tabs.widget(index)
        for index in range(window._tabs.count())
    ]


def test_only_the_current_tab_is_rendered_after_opening(window: MainWindow) -> None:
    panels = _panels(window)
    assert len(panels) == 3

    current = window._current_chart_panel()
    assert current is not None
    assert len(current.figure.axes) > 0, "the shown tab must already be rendered"

    others = [p for p in panels if p is not current]
    assert others, "need at least one non-current tab to prove it stayed deferred"
    for panel in others:
        assert panel._needs_initial_render is True
        assert len(panel.figure.axes) == 0, "a hidden tab must not have rendered yet"


def test_switching_tabs_renders_the_newly_current_one(window: MainWindow) -> None:
    panels = _panels(window)
    target_index = next(
        index for index, p in enumerate(panels) if p is not window._current_chart_panel()
    )
    target_panel = panels[target_index]
    assert target_panel._needs_initial_render is True

    window._tabs.setCurrentIndex(target_index)

    assert target_panel._needs_initial_render is False
    assert len(target_panel.figure.axes) > 0


def test_ensure_rendered_is_a_no_op_the_second_time(window: MainWindow) -> None:
    panel = window._current_chart_panel()
    assert panel is not None
    axes_before = panel.figure.axes[0]

    panel.ensure_rendered()

    assert panel.figure.axes[0] is axes_before, "a second call must not re-render"
