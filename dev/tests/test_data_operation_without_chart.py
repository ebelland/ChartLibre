"""A data operation opens even when the project has only tables: on a scratch chart, dropped if unused."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.main_window import main_window as main_window_module


class _FakeSignal:
    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:  # noqa: ANN001
        self.slots.append(slot)


def _dialog_class(draws: bool):
    class Dialog:
        opened_on: list[int] = []

        def __init__(self, repo, figure_id, parent=None) -> None:  # noqa: ANN001
            self.repo, self.figure_id = repo, figure_id
            type(self).opened_on.append(figure_id)
            self.applied = _FakeSignal()
            self.results_published = _FakeSignal()

        def setWindowIcon(self, _icon) -> None:  # noqa: ANN001, N802
            pass

        def exec(self) -> None:
            if draws:
                descriptor = self.repo.load_figure_descriptor(self.figure_id)
                axis_id = descriptor.axes[0].id
                self.repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s",
                                                   sql_query='SELECT 1 AS x, 2 AS y', roles={"x": "x", "y": "y"}, style={})

    return Dialog


def _window(qapp, tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "ops.dhub")
    repo.import_dataframe(pd.DataFrame({"t": np.arange(5.0), "v": np.arange(5.0)}), table_name="measures",
                          normalize_columns=False)
    return repo, main_window_module.MainWindow(repo=repo, db_path=repo.db_path)


def test_an_operation_opens_on_a_scratch_chart_that_is_dropped_when_it_draws_nothing(qapp, tmp_path: Path) -> None:
    repo, window = _window(qapp, tmp_path)
    try:
        for figure_id, _name in repo.load_figures_from_db():
            repo.delete_figure(int(figure_id))
        window._reload_tabs()
        assert window._current_chart_panel() is None
        dialog = _dialog_class(draws=False)
        window._open_data_operation(dialog)
        assert len(dialog.opened_on) == 1  # it opened - it used to return without a word
        assert list(repo.load_figures_from_db()) == []
    finally:
        window.close()
        repo.close()


def test_the_scratch_chart_stays_when_the_operation_drew_on_it(qapp, tmp_path: Path) -> None:
    repo, window = _window(qapp, tmp_path)
    try:
        for figure_id, _name in repo.load_figures_from_db():
            repo.delete_figure(int(figure_id))
        window._reload_tabs()
        dialog = _dialog_class(draws=True)
        window._open_data_operation(dialog)
        assert [int(i) for i, _n in repo.load_figures_from_db()] == dialog.opened_on
        assert window._current_chart_panel() is not None
    finally:
        window.close()
        repo.close()
