"""A stored function as SQL: its grid and its values computed by SQLite at each read (function_sql)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.data.repo.function_sql import GridRange, function_sql, grid_point
from app.data.sqlite_repo import SqliteRepo
from app.data_operations import sympy_store


@pytest.fixture
def repo(tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "f.dhub")
    yield repo
    repo.close()


def _save(repo: SqliteRepo, name: str, code: str, symbols: str) -> None:
    sympy_store.save(repo, {"function_name": name, "source_type": "expression", "symbols": symbols,
                            "code_string": code})


def _rows(repo: SqliteRepo, sql: str) -> np.ndarray:
    return np.array([tuple(row) for row in repo._con.execute(sql).fetchall()], dtype=float)


def test_the_grid_is_numpys_linear_or_log_range() -> None:
    np.testing.assert_allclose([grid_point(n, -2, 3, 11, "linear") for n in range(11)], np.linspace(-2, 3, 11))
    np.testing.assert_allclose([grid_point(n, 0.1, 1000, 9, "log") for n in range(9)], np.logspace(-1, 3, 9))
    assert grid_point(3, 0, 10, 11, "linear") == pytest.approx(3.0) and grid_point(10, 0, 0.3, 11, "linear") == 0.3
    assert grid_point(0, -1, 10, 5, "log") is None


def test_a_function_is_drawn_from_its_formula_with_each_charts_parameters(repo: SqliteRepo) -> None:
    _save(repo, "Line", "a*x + b", "a, b")
    sql = function_sql("Line", {"a": 2.0, "b": 1.0}, GridRange(0.0, 4.0, 5))
    np.testing.assert_allclose(_rows(repo, sql), [[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]])
    # Another chart, other parameters - a missing one at its default.
    other = function_sql("Line", {"a": -1.0}, GridRange(0.0, 4.0, 5))
    np.testing.assert_allclose(_rows(repo, other)[:, 1], [1, 0, -1, -2, -3])
    # Read as a saved query is: a subquery.
    assert repo._con.execute(f'SELECT COUNT(*), MAX("y") FROM ({sql}) AS "_src"').fetchone()[:] == (5, 9.0)

    # Saved again with a new formula: the same query draws the new one.
    _save(repo, "Line", "a*x**2 + b", "a, b")
    np.testing.assert_allclose(_rows(repo, sql)[:, 1], [1, 3, 9, 19, 33])


def test_a_function_of_x_and_y_is_a_grid_x_fastest(repo: SqliteRepo) -> None:
    _save(repo, "Plane", "a*x + b*y", "a, b")
    rows = _rows(repo, function_sql("Plane", {"a": 1.0, "b": 10.0}, GridRange(0, 2, 3), GridRange(1, 1e3, 4, "log")))
    x_grid, y_grid = np.meshgrid(np.linspace(0, 2, 3), np.logspace(0, 3, 4), indexing="xy")
    np.testing.assert_allclose(rows, np.column_stack((x_grid.ravel(), y_grid.ravel(),
                                                      x_grid.ravel() + 10 * y_grid.ravel())))


def test_an_untrusted_or_failing_function_draws_nothing(repo: SqliteRepo) -> None:
    # Written as a project made elsewhere has it: never saved, never trusted here (todo ID47).
    repo.save_sympy_function("Foreign", "expression", "a", "a*x + 0.123456")
    sql = function_sql("Foreign", {"a": 1.0}, GridRange(0, 1, 3))
    assert [row[1] for row in repo._con.execute(sql)] == [None, None, None]
    _save(repo, "Root", "sqrt(x)", "")
    assert [row[1] for row in repo._con.execute(function_sql("Root", {}, GridRange(-1, 1, 3)))] == [None, 0.0, 1.0]
    assert [row[1] for row in repo._con.execute(function_sql("Missing", {}, GridRange(0, 1, 2)))] == [None, None]


def test_a_stored_function_is_a_data_source_with_its_stored_values(repo: SqliteRepo) -> None:
    sympy_store.save(repo, {"function_name": "Line", "source_type": "expression", "symbols": "a, b",
                            "code_string": "a*x + b", "from": "0", "to": "2", "points": "3",
                            "values": {"a": 3.0, "b": -1.0}})
    assert sympy_store.get(repo, "Line")["values"] == '{"a": 3.0, "b": -1.0}'
    assert sympy_store.catalog(repo)[0]["p0"] == [3.0, -1.0]  # Fit starts from them too

    source = repo.get_data_source("Line")
    assert source is not None and source.is_function and source.is_query
    assert repo.data_source_columns(source) == ["x", "y"]
    frame = repo.query_df(f'SELECT "x" AS "X", "y" AS "Y" FROM {source.from_clause()}')
    np.testing.assert_allclose(frame[["X", "Y"]].to_numpy(), [[0, -1], [1, 2], [2, 5]])
    # A chart's own values, the stored ones for the rest.
    np.testing.assert_allclose(_rows(repo, sympy_store.source_sql(sympy_store.get(repo, "Line"), {"b": 10.0}))[:, 1],
                               [10, 13, 16])

    # Listed for New Chart beside the tables and queries; not in the table list.
    listed = repo.list_data_sources(functions=True)
    assert listed.loc[listed["Table"] == "Line", "kind"].tolist() == ["function"]
    assert "Line" not in repo.list_data_sources()["Table"].tolist()


def test_a_chart_of_a_function_follows_its_formula_and_its_trust(qapp, repo: SqliteRepo, monkeypatch) -> None:
    from app.dialogs.function_trust_dialog import FunctionTrustDialog, allowed_to_run

    sympy_store.save(repo, {"function_name": "Curve", "source_type": "expression", "symbols": "a",
                            "code_string": "a*x", "from": "0", "to": "2", "points": "3", "values": {"a": 2}})
    sql = f'SELECT "x" AS "X", "y" AS "Y" FROM {repo.get_data_source("Curve").from_clause()}'
    assert repo.sql_with_row_id(sql) == sql and repo.row_table(sql) is None and repo.scan_user_sql() == []
    assert repo.series_frame(sql)["Y"].tolist() == [0.0, 2.0, 4.0]
    sympy_store.save(repo, {**sympy_store.get(repo, "Curve"), "code_string": "a*x**3"})
    assert repo.series_frame(sql)["Y"].tolist() == [0.0, 2.0, 16.0]  # the series cache follows the database

    # Changed elsewhere: empty until trusted, then drawn - the cache told.
    repo.save_sympy_function("Curve", "expression", "a", "a*x + 0.5", 0.0, 2.0, 3, values='{"a": 2}')
    assert repo.series_frame(sql)["Y"].isna().all()
    monkeypatch.setattr(FunctionTrustDialog, "exec", lambda _dialog: FunctionTrustDialog.DialogCode.Accepted)
    assert allowed_to_run(None, repo, sympy_store.get(repo, "Curve"))
    assert repo.series_frame(sql)["Y"].tolist() == [0.5, 2.5, 4.5]


def test_new_chart_draws_a_function_with_each_charts_parameters_and_range(qapp, repo: SqliteRepo) -> None:
    from app.dialogs.create_chart_dialog import NewPlotTabDialog

    sympy_store.save(repo, {"function_name": "Line", "source_type": "expression", "symbols": "a, b",
                            "code_string": "a*x + b", "from": "0", "to": "2", "points": "3",
                            "values": {"a": 1.0, "b": 0.0}})
    sympy_store.save(repo, {"function_name": "Plane", "source_type": "expression", "symbols": "c",
                            "code_string": "c*x*y", "y_from": "1", "y_to": "2", "y_points": "2"})

    def chart(a: float, stop: float) -> str:
        dialog = NewPlotTabDialog(repo, current_table="Line")
        try:
            assert dialog._combo_table.currentText() == "Line  (function)"
            assert dialog._select_renderer_by_name("Scatter Plot")
            panel = dialog._function_panel
            assert panel.is_row_shown(panel._x) and not panel.is_row_shown(panel._y)
            assert panel.parameters() == {"a": 1.0, "b": 0.0}  # the stored values, to start from
            # x and y place themselves: no roles, groups or separate axes to choose.
            form = dialog._editor_form_layout
            assert not form.isRowVisible(dialog._roles_scroll) and not form.isRowVisible(dialog._combo_group)
            assert not dialog._axis_form_layout.isRowVisible(dialog._combo_axes_by)
            assert dialog._current_role_mapping()["x"] == "x" and dialog._current_role_mapping()["y"] == "y"
            panel._values["a"].setValue(a)
            panel._x.stop.setValue(stop)
            dialog._on_accept()
            result = dialog.chart_result
            assert result is not None
        finally:
            dialog.close()
        return str(repo.load_figure_descriptor(figure_id=result.figure_id).axes[0].series[0].sql_query)

    def drawn(sql: str) -> list[float]:
        return repo.query_df(sql).iloc[:, 1].tolist()

    first, second = chart(2.0, 2.0), chart(-1.0, 4.0)
    assert (drawn(first), drawn(second)) == ([0.0, 2.0, 4.0], [0.0, -2.0, -4.0])
    # Both charts follow a new formula, each with its own parameters and range.
    sympy_store.save(repo, {**sympy_store.get(repo, "Line"), "code_string": "a*x + b + 100"})
    assert (drawn(first), drawn(second)) == ([100.0, 102.0, 104.0], [100.0, 98.0, 96.0])

    # A function of x and y: its y range too.
    dialog = NewPlotTabDialog(repo, current_table="Plane")
    try:
        panel = dialog._function_panel
        assert panel.is_row_shown(panel._y) and panel._y.grid() == GridRange(1.0, 2.0, 2)
        assert dialog._columns_from_table() == ["x", "y", "z"]
        assert dialog._select_renderer_by_name("Surface Plot")
        assert dialog._current_role_mapping().get("z") == "z"
        assert not dialog._editor_form_layout.isRowVisible(dialog._roles_scroll)
        dialog._combo_table.setCurrentIndex(dialog._combo_table.findData("Line"))  # f(x): no z for a surface
        assert dialog._editor_form_layout.isRowVisible(dialog._roles_scroll)
        dialog._combo_table.setCurrentIndex(-1)  # no function: its rows hidden, the roles shown
        assert not panel.is_row_shown(panel._x) and dialog._editor_form_layout.isRowVisible(dialog._roles_scroll)
    finally:
        dialog.close()


def test_a_chart_of_an_untrusted_function_says_so_and_draws_once_trusted(qapp, repo: SqliteRepo, monkeypatch) -> None:
    from app.data.repo.function_sql import function_names
    from app.dialogs.function_trust_dialog import FunctionTrustDialog
    from app.widgets.chart_panel import ChartPanel

    # A project made elsewhere: its function never saved here (todo ID47).
    repo.save_sympy_function("Wave", "expression", "a", "a*sin(x) + 0.25", 0.0, 3.0, 4)
    sql = f'SELECT "x" AS x, "y" AS y FROM {repo.get_data_source("Wave").from_clause()}'
    assert function_names([sql, 'SELECT 1']) == {"Wave"}
    figure_id = repo.create_figure_descriptor(name="Wave")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="Wave", x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Wave", sql_query=sql,
                                  roles={"x": "x", "y": "y"}, style={})
    panel = ChartPanel(repo, figure_id)
    try:
        panel.resize(600, 400)
        panel.ensure_rendered()
        assert panel.untrusted_functions() == ["Wave"] and not panel._filter_note.isHidden()
        assert "'Wave'" in panel._filter_note.text() and "trust-functions:" in panel._filter_note.text()

        trusted: list[bool] = []
        panel.functions_trusted.connect(lambda: trusted.append(True))
        monkeypatch.setattr(FunctionTrustDialog, "exec", lambda _dialog: FunctionTrustDialog.DialogCode.Accepted)
        panel._on_filter_note_link("trust-functions:")
        panel.reload()
        assert trusted == [True] and panel.untrusted_functions() == [] and panel._filter_note.isHidden()
        assert repo.series_frame(sql)["y"].notna().all()
    finally:
        panel.close()


def test_function_draws_a_stored_function_as_a_live_series(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.function_dialog import SeriesFunctionDialog

    sympy_store.save(repo, {"function_name": "Ramp", "source_type": "expression", "symbols": "a",
                            "code_string": "a*x", "from": "0", "to": "2", "points": "3", "values": {"a": 3.0}})
    figure_id = repo.create_figure_descriptor(name="F")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="F", x_label="x", y_label="y", options={})
    tables_before = set(repo.list_table_names())
    dialog = SeriesFunctionDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.series_selector.reload(select_all_series=True)
        dialog._apply_serialised(sympy_store.get(repo, "Ramp"))
        assert dialog._function_params()[1].tolist() == [3.0]  # the stored value
        dialog._set_param_rows(["a"], [2.0])
        dialog.preview()
        previewed = repo.get_series(axis_id)
        assert len(previewed) == 1 and repo.query_df(previewed[0]["sql_query"])["y"].tolist() == [0.0, 2.0, 4.0]
        dialog.close()  # Cancel: the preview and the function it saved go back
    finally:
        dialog.deleteLater()
    assert repo.get_series(axis_id) == [] and sympy_store.stored_values(sympy_store.get(repo, "Ramp")) == {"a": 3.0}

    dialog = SeriesFunctionDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.series_selector.reload(select_all_series=True)
        dialog._apply_serialised(sympy_store.get(repo, "Ramp"))
        dialog._set_param_rows(["a"], [2.0])
        dialog.ok()
    finally:
        dialog.deleteLater()
    series = repo.get_series(axis_id)
    assert len(series) == 1 and "chartlibre_function('Ramp'" in series[0]["sql_query"]
    assert set(repo.list_table_names()) == tables_before  # no result table: a live series
    assert repo.query_df(series[0]["sql_query"])["y"].tolist() == [0.0, 2.0, 4.0]
    assert sympy_store.stored_values(sympy_store.get(repo, "Ramp")) == {"a": 2.0}  # kept with the function
    sympy_store.save(repo, {**sympy_store.get(repo, "Ramp"), "code_string": "a*x + 1"})
    assert repo.query_df(series[0]["sql_query"])["y"].tolist() == [1.0, 3.0, 5.0]


def test_fit_curve_reads_a_function_series_as_any_other(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.fit_dialog import SeriesFitDialog

    sympy_store.save(repo, {"function_name": "Data", "source_type": "expression", "symbols": "a, b",
                            "code_string": "a*x + b", "from": "0", "to": "5", "points": "20",
                            "values": {"a": 1.5, "b": -2.0}})
    sympy_store.save(repo, {"function_name": "Model", "source_type": "expression", "symbols": "m, q",
                            "code_string": "m*x + q"})
    figure_id = repo.create_figure_descriptor(name="Fit")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="Data", x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Data",
                                  sql_query=f'SELECT "x" AS x, "y" AS y FROM {repo.get_data_source("Data").from_clause()}',
                                  roles={"x": "x", "y": "y"}, style={})
    dialog = SeriesFitDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.series_selector.reload(select_all_series=True)
        assert dialog.select_model("Model")
        dialog.evaluate(dialog._after_fit, background=False, optimise=True)
        np.testing.assert_allclose(dialog.compute_results()[0].params, [1.5, -2.0], atol=1e-6)
    finally:
        dialog.close()


def test_opening_a_project_that_draws_untrusted_functions_asks_once(qapp, tmp_path: Path, monkeypatch) -> None:
    from app.dialogs.function_trust_dialog import FunctionTrustDialog
    from app.main_window.main_window import MainWindow

    path = tmp_path / "elsewhere.dhub"
    repo = SqliteRepo(db_path=path)
    repo.save_sympy_function("Bump", "expression", "a", "a*exp(-x**2) + 0.75", -1.0, 1.0, 5)
    figure_id = repo.create_figure_descriptor(name="Bump")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="Bump", x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Bump", roles={"x": "x", "y": "y"},
                                  sql_query=f'SELECT "x" AS x, "y" AS y FROM {repo.get_data_source("Bump").from_clause()}',
                                  style={})
    asked: list[str] = []

    def exec_(dialog: FunctionTrustDialog) -> int:
        asked.append(dialog._code.toPlainText())
        return FunctionTrustDialog.DialogCode.Rejected

    monkeypatch.setattr(FunctionTrustDialog, "exec", exec_)
    window = MainWindow(repo, path)
    try:
        qapp.processEvents()
        assert len(asked) == 1 and "a*exp(-x**2) + 0.75" in asked[0]
        assert not sympy_store.is_trusted(sympy_store.get(repo, "Bump"))
    finally:
        window.close()
        window.deleteLater()
        repo.close()
