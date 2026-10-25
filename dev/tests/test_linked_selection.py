"""R-07: a box selection on one chart marks its table rows Selected; every chart of that table and the preview show them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel
from app.widgets.table_preview import LazyTableModel


def _scatter(repo: SqliteRepo, name: str, sql: str, chart_type: str = "Scatter Plot",
             roles: dict | None = None) -> int:
    figure_id = repo.create_figure_descriptor(name=name)
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type=chart_type,
                                          title=name, x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=name, sql_query=sql,
                                  roles=roles or {"x": "x", "y": "y"}, style={})
    return figure_id


def test_a_box_on_one_chart_rings_the_same_rows_on_another_and_tints_them_in_the_preview(qapp, tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    x = np.arange(10, dtype=float)
    repo.import_dataframe(pd.DataFrame({"a": x, "b": x * 2.0}), table_name="t", normalize_columns=False)
    first = ChartPanel(repo, _scatter(repo, "first", 'SELECT "a" AS x, "b" AS y FROM "t"'))
    second = ChartPanel(repo, _scatter(repo, "second", 'SELECT "b" AS x, "a" AS y FROM "t"'))
    try:
        for panel in (first, second):
            panel.resize(600, 400)
            panel.ensure_rendered()
        axes = first._canvas.figure.axes[0]
        tables: list[list[str]] = []
        first.rows_selected.connect(tables.append)

        selection = first._collect_selection(axes._dhub_axis_id, axes, 2.5, 5.5, -1.0, 100.0)
        first._select_table_rows(selection)
        assert repo.selected_row_ids("t") == [4, 5, 6] and tables == [["t"]]

        def rings(panel: ChartPanel) -> list[tuple[float, float]]:
            return [tuple(map(float, p)) for a in panel._canvas.figure.axes[0].collections
                    if a.get_label() == panel.SELECTED_ROWS_LABEL for p in a.get_offsets()]

        assert rings(first) == [(3.0, 6.0), (4.0, 8.0), (5.0, 10.0)]
        second.show_selected_rows(["t"])
        assert rings(second) == [(6.0, 3.0), (8.0, 4.0), (10.0, 5.0)]
        second.show_selected_rows(["other"])  # not one of its tables: left as it is
        assert len(rings(second)) == 3

        model = LazyTableModel(repo, "t")
        model.data(model.index(0, 0))  # reads the first rows
        tint = Qt.ItemDataRole.BackgroundRole
        assert model.data(model.index(3, 0), tint) is not None and model.data(model.index(2, 0), tint) is None
        assert "_id" not in [model.column_name(i) for i in range(model.columnCount())]
        assert repo.row_position("t", 4) == 3
    finally:
        first.close()
        second.close()
        repo.close()


def test_a_click_selects_a_point_shift_adds_or_takes_it_away_and_a_click_on_nothing_clears(qapp,
                                                                                           tmp_path: Path) -> None:
    from matplotlib.backend_bases import MouseEvent

    repo = SqliteRepo(db_path=tmp_path / "c.dhub")
    x = np.arange(10, dtype=float)
    repo.import_dataframe(pd.DataFrame({"a": x, "b": x * 2.0}), table_name="t", normalize_columns=False)
    panel = ChartPanel(repo, _scatter(repo, "c", 'SELECT "a" AS x, "b" AS y FROM "t"'))
    try:
        panel.resize(600, 400)
        panel.ensure_rendered()
        panel._on_toggle_select_mode(True)
        axes = panel._canvas.figure.axes[0]

        def click(x_data: float, y_data: float, key: str | None = None, to: tuple[float, float] | None = None) -> None:
            start = axes.transData.transform((x_data, y_data))
            end = axes.transData.transform(to) if to is not None else start
            panel._on_box_select_press(MouseEvent("button_press_event", panel._canvas, *start, button=1, key=key))
            panel._on_box_select_release(MouseEvent("button_release_event", panel._canvas, *end, button=1, key=key))

        click(3.0, 6.0)
        assert repo.selected_row_ids("t") == [4]
        click(7.0, 14.0, key="shift")
        assert repo.selected_row_ids("t") == [4, 8]
        click(3.0, 6.0, key="shift")  # again: taken away
        assert repo.selected_row_ids("t") == [8]
        click(0.5, 0.5, key="shift", to=(1.5, 4.5))  # Shift-drag: the box added
        assert repo.selected_row_ids("t") == [2, 8]
        click(6.0, 2.0)  # on nothing: cleared
        assert repo.selected_row_ids("t") == []
    finally:
        panel.close()
        repo.close()


def test_a_series_over_a_saved_query_names_its_rows(tmp_path: Path) -> None:
    from app.data.data_source import DataSource

    repo = SqliteRepo(db_path=tmp_path / "q.dhub")
    x = np.arange(10, dtype=float)
    repo.import_dataframe(pd.DataFrame({"a": x, "b": x * 2.0}), table_name="t", normalize_columns=False)
    try:
        source = DataSource.query("big", 'SELECT "a", "b" FROM "t" WHERE "a" > 5')
        sql = f'SELECT "a" AS x, "b" AS y FROM {source.from_clause()}'
        with_id = repo.sql_with_row_id(sql)
        assert with_id.count('"_id"') == 2 and repo.row_table(with_id) == "t"
        rows = repo.series_rows(with_id, {"x": "x", "y": "y"})
        assert sorted(rows["_id"].astype(int)) == [7, 8, 9, 10]
        grouped = 'SELECT "a" AS x, COUNT(*) AS y FROM (SELECT "a" FROM "t") AS "q" GROUP BY "a"'
        assert repo.sql_with_row_id(grouped) == grouped  # a point is not one row: left as it is
    finally:
        repo.close()


def test_dates_and_categories_on_x_select_and_ring_their_rows(qapp, tmp_path: Path) -> None:
    from matplotlib.backend_bases import MouseEvent

    repo = SqliteRepo(db_path=tmp_path / "d.dhub")
    days = pd.date_range("2024-01-01", periods=10).strftime("%Y-%m-%d")
    repo.import_dataframe(pd.DataFrame({"day": days, "kind": list("ABCDEFGHIJ"), "v": np.arange(10, dtype=float)}),
                          table_name="t", normalize_columns=False)
    dated = ChartPanel(repo, _scatter(repo, "dated", 'SELECT "day" AS x, "v" AS y FROM "t"'))
    named = ChartPanel(repo, _scatter(repo, "named", 'SELECT "kind" AS X, "v" AS Y FROM "t"', "Bar Chart",
                                   {"X": "X", "Y": "Y"}))
    try:
        for panel in (dated, named):
            panel.resize(600, 400)
            panel.ensure_rendered()
        axes = dated._canvas.figure.axes[0]
        start, end = axes.xaxis.convert_units(np.array(["2024-01-03", "2024-01-05"], dtype="datetime64[D]"))
        selection = dated._collect_selection(axes._dhub_axis_id, axes, float(start) - 0.1, float(end) + 0.1, -1, 100)
        dated._select_table_rows(selection)
        assert repo.selected_row_ids("t") == [3, 4, 5]
        rings = [a for a in axes.collections if a.get_label() == dated.SELECTED_ROWS_LABEL]
        assert rings and len(rings[0].get_offsets()) == 3

        named._on_toggle_select_mode(True)
        axes = named._canvas.figure.axes[0]
        named._canvas.draw()
        at = {label.get_text(): loc for loc, label in zip(axes.xaxis.get_ticklocs(), axes.xaxis.get_ticklabels())}
        point = axes.transData.transform((float(at["H"]), 7.0))  # the top of the bar of H
        named._on_box_select_press(MouseEvent("button_press_event", named._canvas, *point, button=1))
        named._on_box_select_release(MouseEvent("button_release_event", named._canvas, *point, button=1))
        assert repo.selected_row_ids("t") == [8]
    finally:
        dated.close()
        named.close()
        repo.close()
