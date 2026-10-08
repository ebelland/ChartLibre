"""A series operation on a table's columns: drawn as a figure of their own, then run on it."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from app.analysis.smoothing import SMOOTH_MOVING_AVERAGE
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.smoothing_dialog import SeriesSmoothingDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state() -> Iterator[None]:
    clear_state("SeriesSmoothingDialog")
    yield
    clear_state("SeriesSmoothingDialog")


@pytest.fixture
def chart_and_table(repo: SqliteRepo) -> int:
    """An empty chart to open the window on, and a table it does not draw."""
    figure_id = int(repo.create_figure_descriptor(name="chart"))
    repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                title="t", x_label="x", y_label="y", options={})
    t = np.arange(50.0)
    repo.import_dataframe(pd.DataFrame({"time": t, "signal": np.sin(t / 5), "label": ["a"] * 50}),
                          table_name="measures", normalize_columns=False)
    return figure_id


def _figures(repo: SqliteRepo) -> set[int]:
    return {int(figure_id) for figure_id, _name in repo.load_figures_from_db()}


def test_preview_and_apply_on_a_tables_columns(qapp, repo: SqliteRepo, chart_and_table: int) -> None:
    before = _figures(repo)
    dialog = SeriesSmoothingDialog(repo=repo, figure_id=chart_and_table)
    try:
        dialog.source_table_radio.setChecked(True)
        source = dialog.table_source
        source.set_spec("measures", "time", ["signal"])
        assert source.spec() == ("measures", "time", ("signal",))
        dialog.method_combo.setCurrentText(SMOOTH_MOVING_AVERAGE)

        assert dialog.preview()
        drawn = _figures(repo) - before
        assert len(drawn) == 1  # the columns, drawn
        assert dialog._figure_id == next(iter(drawn))
        assert [row["name"] for row in dialog.selected_series()] == ["signal"]
        assert dialog.preview()  # again: the same figure, not a second one
        assert _figures(repo) - before == drawn

        applied: list[bool] = []
        dialog.apply(then=applied.append)
        assert applied == [True]
    finally:
        dialog.close()
    figure = repo.load_figure_descriptor(next(iter(drawn)))
    names = [series.name for axis in figure.axes for series in axis.series]
    assert "signal" in names and len(names) == 2  # the data and its smoothing


def test_closing_without_apply_removes_the_drawn_figure(qapp, repo: SqliteRepo, chart_and_table: int) -> None:
    before = _figures(repo)
    dialog = SeriesSmoothingDialog(repo=repo, figure_id=chart_and_table)
    dialog.source_table_radio.setChecked(True)
    dialog.table_source.set_spec("measures", "", ["signal"])  # x: the row number
    dialog.method_combo.setCurrentText(SMOOTH_MOVING_AVERAGE)
    assert dialog.preview()
    assert len(_figures(repo) - before) == 1
    dialog.source_chart_radio.setChecked(True)  # back to the chart: the drawing goes
    assert _figures(repo) == before and dialog._figure_id == chart_and_table
    dialog.source_table_radio.setChecked(True)
    assert dialog.preview()
    dialog.reject()
    assert _figures(repo) == before


def test_the_window_remembers_its_parameters_and_columns_and_reverts(qapp, repo: SqliteRepo, chart_and_table: int) -> None:
    from app.series_operations.peaks_dialog import SeriesPeaksDialog

    clear_state("SeriesPeaksDialog")
    dialog = SeriesPeaksDialog(repo=repo, figure_id=chart_and_table)
    defaults = dialog.parameter_values()
    form = dialog._parameter_form_spec
    form.set_values({"distance": defaults["distance"] + 7})
    dialog.source_table_radio.setChecked(True)
    dialog.table_source.set_spec("measures", "time", ["signal"])
    dialog.reject()  # remembers, like every way out

    again = SeriesPeaksDialog(repo=repo, figure_id=chart_and_table)
    try:
        assert again.parameter_values()["distance"] == defaults["distance"] + 7
        assert again.reads_table() and again._source_stack.currentIndex() == 1
        assert again.table_source.spec() == ("measures", "time", ("signal",))
        again.revert_entries()
        assert again.parameter_values() == defaults
        assert not again.reads_table() and again._source_stack.currentIndex() == 0
    finally:
        again.reject()
        clear_state("SeriesPeaksDialog")
