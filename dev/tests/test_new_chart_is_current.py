"""A chart just created is the one shown."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import app.dialogs.main_window as main_window_module
from app.data.sqlite_repo import SqliteRepo
from app.dialogs.create_chart_dialog import NewPlotTabResult
from app.dialogs.main_window import MainWindow


def _figure(repo: SqliteRepo, name: str) -> int:
    figure_id = int(repo.create_figure_descriptor(name=name))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title=name, x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=name,
                                  sql_query="SELECT a AS x, b AS y FROM t", roles={"x": "x", "y": "y"}, style={})
    return figure_id


def test_a_new_plot_becomes_the_current_tab(qapp, tmp_path: Path, monkeypatch) -> None:
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), table_name="t", normalize_columns=False)
    first = _figure(repo, "first")
    _figure(repo, "second")
    window = MainWindow(repo, tmp_path / "p.dhub")
    window._tabs.setCurrentIndex(window._tab_index_of_figure(first))

    class _NewPlot:
        Icon = ""

        def __init__(self, repo_, **_kwargs) -> None:
            self.chart_result: NewPlotTabResult | None = None
            self._repo = repo_

        def setWindowIcon(self, _icon) -> None:
            pass

        def exec(self) -> int:
            figure_id = _figure(self._repo, "third")
            self.chart_result = NewPlotTabResult(figure_id=figure_id, axis_id=0, series_count=1)
            return 1

    monkeypatch.setattr(main_window_module, "NewPlotTabDialog", _NewPlot)
    window._on_new_plot_tab()
    panel = window._current_chart_panel()
    assert panel is not None and window._tabs.tabText(window._tabs.currentIndex()) == "third"
    window.close()
    repo.close()
