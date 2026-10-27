"""Every data operation's pages fit their panel: no control wider than it, cut off at its edge."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
from PySide6.QtWidgets import QScrollArea

import app.data_operations as package
from app.data.sqlite_repo import SqliteRepo
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase
from app.utils import dialog_state


def _operations() -> list[type[DataOperationDialogBase]]:
    found = []
    for path in sorted(Path(next(iter(package.__path__))).glob("*_dialog.py")):
        module = importlib.import_module(f"app.data_operations.{path.stem}")
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, DataOperationDialogBase) and cls.__module__ == module.__name__:
                found.append(cls)
    return found


@pytest.mark.parametrize("operation", _operations(), ids=lambda cls: cls.__name__)
def test_every_page_fits_its_panel(qapp, repo: SqliteRepo, operation) -> None:
    dialog_state.clear_state(operation.__name__)
    figure_id = int(repo.create_figure_descriptor(name="chart"))
    repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                title="t", x_label="x", y_label="y", options={})
    dialog = operation(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.resize(dialog.minimumSizeHint())  # the narrowest it can be: the case that cuts
        dialog.show()
        qapp.processEvents()
        tabs = dialog.left_tabs
        cut = []
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            qapp.processEvents()
            scroll = tabs.widget(index)
            if isinstance(scroll, QScrollArea) and scroll.widget() is not None:
                page = scroll.widget()
                if page.width() > scroll.viewport().width():
                    cut.append(f"{tabs.tabText(index)}: {page.width()} > {scroll.viewport().width()}")
        assert cut == []
        assert dialog.minimumSizeHint().width() <= 1280  # still on a laptop's screen
    finally:
        dialog.close()
