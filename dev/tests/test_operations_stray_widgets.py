"""No operation leaves a widget loose on its window: one outside every layout is drawn over the top-left corner
(the series selector over "Settings" in Fit Y by X and Fit Model)."""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLayout, QWidget

import app.data_operations as package
from app.data.sqlite_repo import SqliteRepo
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase


def _operations() -> list[type[DataOperationDialogBase]]:
    found = []
    for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        module = importlib.import_module(info.name)
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, DataOperationDialogBase) and cls.__module__ == module.__name__ \
                    and not inspect.isabstract(cls) and getattr(cls, "Name", ""):
                found.append(cls)
    return found


def _in_a_layout(layout: QLayout | None, widget: QWidget) -> bool:
    if layout is None:
        return False
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is widget or _in_a_layout(item.layout(), widget):
            return True
    return False


@pytest.mark.parametrize("operation", _operations(), ids=lambda cls: cls.__name__)
def test_no_widget_is_loose_on_the_window(qapp, repo: SqliteRepo, operation) -> None:
    repo.import_dataframe(pd.DataFrame({"a": np.arange(30.0), "b": np.random.default_rng(0).normal(size=30)}),
                          table_name="t", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="A", x_label="a", y_label="b", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="b", style={},
                                  sql_query='SELECT a AS x, b AS y FROM "t"', roles={"x": "x", "y": "y"})
    dialog = operation(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.show()
        qapp.processEvents()
        loose = [(type(child).__name__, child.objectName()) for child in dialog.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
                 if child.isVisible() and not child.isWindow() and not _in_a_layout(dialog.layout(), child)]
        assert not loose
    finally:
        dialog.close()


@pytest.mark.parametrize("operation", _operations(), ids=lambda cls: cls.__name__)
def test_no_tab_is_empty(qapp, repo: SqliteRepo, operation) -> None:
    """Every tab the window offers has something on it: no Model without models, no Parameters without any."""
    from PySide6.QtWidgets import (
        QAbstractButton,
        QAbstractItemView,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
    )

    repo.import_dataframe(pd.DataFrame({"a": np.arange(30.0), "b": np.random.default_rng(0).normal(size=30)}),
                          table_name="t", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="A", x_label="a", y_label="b", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="b", style={},
                                  sql_query='SELECT a AS x, b AS y FROM "t"', roles={"x": "x", "y": "y"})
    dialog = operation(repo=repo, figure_id=figure_id, parent=None)
    controls = (QAbstractButton, QAbstractItemView, QAbstractSpinBox, QComboBox, QLineEdit)
    try:
        dialog.show()
        empty = []
        for index in range(dialog.left_tabs.count()):
            if not dialog.left_tabs.isTabVisible(index):
                continue
            dialog.left_tabs.setCurrentIndex(index)
            qapp.processEvents()
            page = dialog.left_tabs.widget(index)
            if not any(child.isVisible() for kind in controls for child in page.findChildren(kind)):
                empty.append(dialog.left_tabs.tabText(index))
        assert not empty
    finally:
        dialog.close()


def test_every_scanned_operation_runs_on_the_one_engine() -> None:
    """ID48: one base - every operation the Analysis panel offers is a DataOperationDialogBase, roles or not."""
    from app.scanners.data_operation_scanner import (
        data_operations,
        import_class_from_file,
    )

    loaded = [(operation.get("name"), import_class_from_file(operation)) for operation in data_operations]
    assert loaded and not [name for name, cls in loaded if cls is None or not issubclass(cls, DataOperationDialogBase)]
