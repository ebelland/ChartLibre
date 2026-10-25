"""Table analyses preview their charts (todo O-01): drawn inside a savepoint, rolled back, kept on OK."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def _wait(qapp, dialog) -> None:
    deadline = time.monotonic() + 30
    while dialog.job_running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def _figures(repo: SqliteRepo) -> list[int]:
    return [int(row[0]) for row in repo._con.execute("SELECT id FROM __figure_descriptors__").fetchall()]


def _dialog(repo: SqliteRepo):
    from app.table_operations.distribution_dialog import DistributionDialog

    repo.import_dataframe(pd.DataFrame({"y": np.random.default_rng(0).normal(size=50)}), table_name="t",
                          normalize_columns=False)
    dialog = DistributionDialog(repo=repo, table="t")
    dialog.roles_widget.set_casting({"response": ["y"]})
    return dialog


def test_preview_draws_a_figure_that_cancel_takes_away(qapp, repo: SqliteRepo) -> None:
    dialog = _dialog(repo)
    shown: list[int] = []
    dialog.preview_drawn.connect(shown.append)
    try:
        dialog.preview()
        _wait(qapp, dialog)
        assert len(_figures(repo)) == 1 and shown[-1] in _figures(repo)
        dialog.preview()
        _wait(qapp, dialog)
        assert len(_figures(repo)) == 1  # the second preview replaced the first
        dialog.reject()
        assert _figures(repo) == [] and shown[-1] == -1
    finally:
        dialog.close()


def test_ok_after_preview_keeps_one_figure_with_its_undo_step(qapp, repo: SqliteRepo) -> None:
    dialog = _dialog(repo)
    try:
        dialog.preview()
        _wait(qapp, dialog)
        dialog.ok()
        _wait(qapp, dialog)
        assert len(_figures(repo)) == 1
        repo.undo_last()
        assert _figures(repo) == []
    finally:
        dialog.close()


def test_new_axes_in_the_current_figure(qapp, repo: SqliteRepo) -> None:
    current = repo.create_figure_descriptor(name="mine")
    repo.create_axis_descriptor(figure_id=current, axis_index=0, chart_type="Scatter Plot", title="",
                                x_label="", y_label="", options={})
    dialog = _dialog(repo)
    try:
        dialog.current_figure_id = current
        dialog.destination_combo.setCurrentIndex(dialog.destination_combo.findData(dialog.DEST_NEW_AXIS))
        dialog.ok()
        _wait(qapp, dialog)
        assert _figures(repo) == [current]
        assert len(repo.load_figure_descriptor(current).axes) > 1
    finally:
        dialog.close()


def test_distribution_fits_the_chosen_or_best_distribution(qapp, repo: SqliteRepo) -> None:
    from app.table_operations.distribution_dialog import DistributionDialog

    repo.import_dataframe(pd.DataFrame({"y": np.random.default_rng(1).lognormal(1.0, 0.6, 400)}), table_name="ln",
                          normalize_columns=False)
    dialog = DistributionDialog(repo=repo, table="ln")
    try:
        dialog.roles_widget.set_casting({"response": ["y"]})
        chosen = {**dialog.parameter_values(), "distribution": "best"}
        dialog.parameter_values = lambda: chosen
        dialog.ok()
        _wait(qapp, dialog)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        options = {axis.name: axis.options for axis in figure.axes}
        assert options["Histogram"]["distribution_fit"] == "lognorm"
        assert options["Q-Q Plot"]["distribution"] == "lognorm"
    finally:
        dialog.close()


def test_a_preview_written_through_pandas_is_still_rolled_back(tmp_path) -> None:
    """pandas' to_sql commits after it writes: that ended the Preview's savepoint, and a cancelled Preview
    stayed in the project - its tables (Optimize found them unreferenced) and its figure."""
    import datetime as dt

    from app.data.sqlite_repo import SqliteRepo

    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    repo.begin_preview_savepoint("preview_test")
    # A date column: written by pandas, not by the repository's own executemany.
    repo.import_dataframe(pd.DataFrame({"d": [dt.datetime(2024, 1, 1)], "v": [1.0]}), table_name="previewed",
                          normalize_columns=False)
    assert repo.rollback_preview_savepoint()
    assert "previewed" not in set(repo.list_user_tables()["Table"])
    repo.close()


def test_an_analysis_reads_a_charts_axis_as_a_table(qapp, tmp_path) -> None:
    """Data from: Chart - the axis' series a column each, their shared x one; written as a table by Preview
    (rolled back with it) and by OK, in OK's one undo step."""
    import numpy as np

    from app.data.sqlite_repo import SqliteRepo
    from app.table_operations.distribution_dialog import DistributionDialog

    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    rng = np.random.default_rng(1)
    repo.import_dataframe(pd.DataFrame({"t": np.arange(40.0), "a": rng.normal(size=40), "b": rng.normal(size=40)}),
                          table_name="t", normalize_columns=False)
    figure = repo.create_figure_descriptor(name="Signals")
    axis = repo.create_axis_descriptor(figure_id=figure, axis_index=0, chart_type="Scatter Plot", title="Two",
                                       x_label="time", y_label="v", options={})
    for index, name in enumerate("ab"):
        repo.create_series_descriptor(axis_id=axis, series_index=index, name=name,
                                      sql_query=f'SELECT t AS x, {name} AS y FROM "t"', roles={"x": "x", "y": "y"},
                                      style={})
    dialog = DistributionDialog(repo=repo, table="t")
    try:
        dialog.current_figure_id = figure
        dialog.source_chart_radio.setChecked(True)
        assert dialog.table == "Signals_Two" and list(dialog._frame.columns) == ["time", "a", "b"]
        dialog.roles_widget.set_casting({"response": ["a", "b"]})
        dialog.preview()
        _wait(qapp, dialog)
        assert "Signals_Two" in set(repo.list_user_tables()["Table"])
        dialog._discard_preview()
        assert "Signals_Two" not in set(repo.list_user_tables()["Table"])
        dialog.ok()
        _wait(qapp, dialog)
        assert "Signals_Two" in set(repo.list_user_tables()["Table"])
        assert [entry.label for entry in repo.undo_entries()] == ["Distribution on 'Signals_Two'"]
    finally:
        dialog.close()
        repo.close()
