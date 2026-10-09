"""Every series operation runs on a table's columns as on a chart's series (Data from: Table)."""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QDialog

import app.series_operations as package
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import SeriesOperationDialogBase
from app.utils import dialog_state


def _operations() -> list[type[SeriesOperationDialogBase]]:
    found = []
    for path in sorted(Path(next(iter(package.__path__))).glob("*_dialog.py")):
        module = importlib.import_module(f"app.series_operations.{path.stem}")
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if (issubclass(cls, SeriesOperationDialogBase) and cls.__module__ == module.__name__
                    and cls.READS_TABLES and cls.SHOWS_SERIES_SELECTOR and cls.SHOWS_AXIS_SERIES_PAGE):
                found.append(cls)
    return found


@pytest.mark.parametrize("operation", _operations(), ids=lambda cls: cls.__name__)
def test_the_operation_runs_on_a_tables_columns(qapp, repo: SqliteRepo, operation, monkeypatch) -> None:
    dialog_state.clear_state(operation.__name__)
    told: list[str] = []
    for module in (sys.modules[operation.__module__], sys.modules[SeriesOperationDialogBase.__module__]):
        # A message box would wait for a click that never comes.
        monkeypatch.setattr(module, "show_message", lambda _parent, key, **k: told.append(key), raising=False)
    for level in ("error", "critical"):
        monkeypatch.setattr(applogger, level, lambda message, *a, **k: told.append(str(message) % a if a else str(message)))
    figure_id = int(repo.create_figure_descriptor(name="chart"))
    repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                title="t", x_label="x", y_label="y", options={})
    t = np.linspace(0, 20, 200)
    rng = np.random.default_rng(0)
    repo.import_dataframe(pd.DataFrame({
        "time": t, "signal": np.sin(t) + rng.normal(0, 0.1, t.size), "level": 2 * t + rng.normal(0, 0.5, t.size),
    }), table_name="measures", normalize_columns=False)
    before = {int(f) for f, _n in repo.load_figures_from_db()}

    dialog = operation(repo=repo, figure_id=figure_id, parent=None)
    # The window remembers Table on the way out: not for the next test.
    monkeypatch.setattr(dialog, "_remember_state", lambda: None)
    try:
        dialog.source_table_radio.setChecked(True)
        dialog.table_source.set_spec("measures", "time", ["signal", "level"])
        # Drawn as soon as the columns are chosen: what the operation reads
        # before Preview (its own buttons, its lists) is the table's already.
        assert dialog._figure_id != figure_id
        assert sorted(str(row["name"]) for row in dialog.selected_series()) == ["level", "signal"]
        assert dialog.preview(), told
        dialog.ok()
        assert dialog.result() == QDialog.DialogCode.Accepted, told
    finally:
        dialog.close()
    assert {int(f) for f, _n in repo.load_figures_from_db()} - before, "the columns were not drawn"
