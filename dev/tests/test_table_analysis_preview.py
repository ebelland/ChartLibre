"""Table analyses preview their charts (todo O-01): drawn inside a savepoint, rolled back, kept on OK."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def _wait(qapp, dialog) -> None:
    deadline = time.monotonic() + 30
    while dialog._task is not None and time.monotonic() < deadline:
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
        dialog.draw_on_combo.setCurrentIndex(dialog.draw_on_combo.findData(dialog.DRAW_NEW_AXIS))
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
