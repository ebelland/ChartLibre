"""Operations that declare MODELS get the standard Model combo and Docs link."""
from __future__ import annotations

from typing import Any, cast

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app.series_operations as package
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import SeriesOperationDialogBase
from app.series_operations.interpolate_dialog import (
    MODEL_EXPONENTIAL,
    MODEL_SCIPY_PCHIP,
    SeriesInterpolateDialog,
)


def _operations_with_models() -> list[type[SeriesOperationDialogBase]]:
    found = []
    for module_info in pkgutil.iter_modules(package.__path__):
        if not module_info.name.endswith("_dialog"):
            continue
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if (
                cls.__module__ == module.__name__
                and issubclass(cls, SeriesOperationDialogBase)
                and cls.MODELS
            ):
                found.append(cls)
    return found


@pytest.fixture
def figure(tmp_db_path: Path) -> Iterator[tuple[SqliteRepo, int]]:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    repo = SqliteRepo(db_path=tmp_db_path)
    x = np.linspace(1.0, 10.0, 40)
    repo.import_dataframe(
        pd.DataFrame({"x": x, "y": np.exp(0.2 * x)}), table_name="curve", normalize_columns=False
    )
    figure_id = repo.create_figure_descriptor(name="Models")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Line Plot",
        title="curve", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="curve",
        sql_query='SELECT "x", "y" FROM "curve"', roles={"x": "x", "y": "y"}, style={},
    )
    yield repo, figure_id
    repo.close()


def _items(combo) -> list[str]:
    # Separators between groups are items with no text.
    return [combo.itemText(i) for i in range(combo.count()) if combo.itemText(i)]


@pytest.mark.parametrize("cls", _operations_with_models(), ids=lambda cls: cls.__name__)
def test_the_combo_lists_the_models_and_the_link_follows(qapp, figure, cls) -> None:
    repo, figure_id = figure
    dialog = cls(repo=repo, figure_id=figure_id, parent=None)
    try:
        assert _items(dialog.model_combo) == list(cls.MODELS)
        for index in range(dialog.model_combo.count()):
            name = dialog.model_combo.itemText(index)
            if not name:
                continue
            dialog.model_combo.setCurrentIndex(index)
            url = cls.MODELS[name].doc_url
            if url:
                assert url in dialog._doc_link.text(), name
    finally:
        dialog.close()
        applogger.set_status_bar(None)


def test_interpolation_start_parameters_are_offered_for_the_fitted_models(qapp, figure) -> None:
    """They are curve_fit's p0: a fitted model takes them, an interpolant has none."""
    repo, figure_id = figure
    dialog = SeriesInterpolateDialog(repo=repo, figure_id=figure_id, parent=cast(Any, None))
    try:
        dialog.show()
        dialog.model_combo.setCurrentText(MODEL_EXPONENTIAL)
        qapp.processEvents()
        assert not dialog._params_edit.isHidden()
        assert dialog._start_params() != {}

        dialog.model_combo.setCurrentText(MODEL_SCIPY_PCHIP)
        qapp.processEvents()
        assert dialog._params_edit.isHidden()
        assert dialog._start_params() == {}
    finally:
        dialog.close()
        applogger.set_status_bar(None)
