"""Preview and Apply on a worker thread (todo R-02): the same tables, Stop, the bar, OK.

Every other test runs the operations on the GUI thread (see conftest); these
are marked ``background`` and run them the way the app does.
"""
from __future__ import annotations

import importlib
import pkgutil
import re
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QStatusBar

import app.series_operations as operations
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import SeriesOperationDialogBase
from app.utils.dialog_state import clear_state

#: module, class: every operation that computes in the background.
BACKGROUND_OPERATIONS = [
    ("calculus_dialog", "SeriesCalculusDialog"),
    ("cluster_dialog", "SeriesClusterDialog"),
    ("control_chart_dialog", "SeriesControlChartDialog"),
    ("decomposition_dialog", "SeriesDecompositionDialog"),
    ("gp_regression_dialog", "SeriesGPRegressionDialog"),
    ("interpolate_dialog", "SeriesInterpolateDialog"),
    ("peaks_dialog", "SeriesPeaksDialog"),
    ("regression_dialog", "SeriesRegressionDialog"),
    ("roots_dialog", "SeriesRootsDialog"),
    ("smoothing_dialog", "SeriesSmoothingDialog"),
    ("transform_dialog", "SeriesTransformDialog"),
]


def _wait_for(condition: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the background run")
        QApplication.processEvents()
        time.sleep(0.005)


def _dialog_class(module: str, name: str) -> type[SeriesOperationDialogBase]:
    return getattr(importlib.import_module(f"app.series_operations.{module}"), name)


@pytest.fixture
def figure_id(repo: SqliteRepo) -> int:
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 12.0, 160)
    repo.import_dataframe(
        pd.DataFrame({
            "t": t,
            "a": np.sin(t) + rng.normal(0.0, 0.05, t.size),
            "b": np.cos(0.7 * t) + rng.normal(0.0, 0.05, t.size),
        }),
        table_name="sig", normalize_columns=False,
    )
    figure = int(repo.create_figure_descriptor(name="f"))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure, axis_index=0, chart_type="Scatter Plot",
        title="t", x_label="t", y_label="y", options={},
    ))
    for index, column in enumerate(("a", "b")):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=column,
            sql_query=f"SELECT t AS x, {column} AS y FROM sig",
            roles={"x": "x", "y": "y"}, style={},
        )
    return figure


@pytest.fixture(autouse=True)
def _fresh_dialog_state() -> Iterator[None]:
    for _module, name in BACKGROUND_OPERATIONS:
        clear_state(name)
    yield
    for _module, name in BACKGROUND_OPERATIONS:
        clear_state(name)


def _open(cls: type[SeriesOperationDialogBase], repo: SqliteRepo, figure_id: int) -> SeriesOperationDialogBase:
    dialog = cls(repo=repo, figure_id=figure_id, parent=None)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


def _preview_tables(dialog: SeriesOperationDialogBase, repo: SqliteRepo) -> dict[str, pd.DataFrame]:
    # Named after the axis they land on, and an operation that draws on a
    # new axis gets a new one each run: the number is not part of the result.
    return {
        re.sub(r"axis\d+", "axis", name): repo.query_df(f'SELECT * FROM "{name}"')
        for name in sorted(getattr(dialog, "_preview_table_names", set()))
    }


def test_every_background_operation_is_covered_here() -> None:
    found = []
    for info in pkgutil.iter_modules(operations.__path__):
        if not info.name.endswith("_dialog"):
            continue
        module = importlib.import_module(f"app.series_operations.{info.name}")
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, SeriesOperationDialogBase)
                and value.__module__ == module.__name__
                and value.RUN_IN_BACKGROUND
            ):
                found.append((info.name, value.__name__))
    assert sorted(found) == sorted(BACKGROUND_OPERATIONS)


def _comparable(result: object) -> object:
    for attribute in ("to_df", "result_to_frame"):
        method = getattr(result, attribute, None)
        if callable(method):
            return method()
    return getattr(result, "frame", repr(result))


def _preview(dialog: SeriesOperationDialogBase) -> list[object]:
    """Preview, and what was handed to the chart."""
    delivered: list[object] = []
    original = dialog.preview_results_to_axis

    def spy(axis_id: int, results: list[object]) -> None:
        delivered.extend(_comparable(result) for result in results)
        original(axis_id, results)

    dialog.preview_results_to_axis = spy  # type: ignore[method-assign]
    np.random.seed(0)  # k-means starts from random centroids
    assert dialog.preview()
    _wait_for(lambda: not dialog.evaluating)
    return delivered


def _assert_same(got: object, expected: object) -> None:
    if isinstance(expected, pd.DataFrame):
        assert isinstance(got, pd.DataFrame)
        pd.testing.assert_frame_equal(got, expected)
    else:
        assert got == expected


@pytest.mark.background
@pytest.mark.parametrize(("module", "name"), BACKGROUND_OPERATIONS)
def test_a_background_preview_writes_what_the_gui_thread_writes(
    qapp, repo: SqliteRepo, figure_id: int, module: str, name: str
) -> None:
    cls = _dialog_class(module, name)

    SeriesOperationDialogBase.BACKGROUND_ENABLED = False
    dialog = _open(cls, repo, figure_id)
    expected_results = _preview(dialog)
    expected_tables = _preview_tables(dialog, repo)
    dialog.close()

    SeriesOperationDialogBase.BACKGROUND_ENABLED = True
    dialog = _open(cls, repo, figure_id)
    got_results = _preview(dialog)
    got_tables = _preview_tables(dialog, repo)
    dialog.close()

    assert expected_results, "the GUI-thread preview delivered nothing to compare against"
    assert len(got_results) == len(expected_results)
    for got, expected in zip(got_results, expected_results):
        _assert_same(got, expected)
    assert got_tables.keys() == expected_tables.keys()
    for table, frame in expected_tables.items():
        pd.testing.assert_frame_equal(got_tables[table], frame)


@pytest.mark.background
def test_ok_closes_the_window_only_once_the_background_result_is_applied(
    qapp, repo: SqliteRepo, figure_id: int
) -> None:
    dialog = _open(_dialog_class("transform_dialog", "SeriesTransformDialog"), repo, figure_id)
    dialog.show()
    dialog.ok()
    assert dialog.evaluating
    assert dialog.isVisible()
    _wait_for(lambda: not dialog.isVisible())
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog._applied


@pytest.mark.background
def test_stop_shows_the_bar_and_then_writes_nothing(
    qapp, repo: SqliteRepo, figure_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = _dialog_class("regression_dialog", "SeriesRegressionDialog")
    release = threading.Event()
    original = cls.compute_series

    def slow(self, name, data, settings):  # noqa: ANN001 - the dialog's own signature
        release.wait(10.0)
        return original(self, name, data, settings)

    monkeypatch.setattr(cls, "compute_series", slow)
    dialog = _open(cls, repo, figure_id)
    dialog.show()
    assert dialog.preview()
    assert not dialog.preview_button.isEnabled()
    # Two series: the bar counts them, once the run has outlasted the delay.
    _wait_for(lambda: dialog.stop_button.isVisible())
    assert dialog.progress_bar.isVisible()
    assert dialog.progress_bar.maximum() == 2

    dialog.stop_evaluation()
    release.set()
    _wait_for(lambda: not dialog.evaluating)
    assert not dialog.stop_button.isVisible()
    assert not dialog.progress_bar.isVisible()
    assert dialog.preview_button.isEnabled()
    assert not dialog._preview_active
    assert not _preview_tables(dialog, repo)
    dialog.close()


@pytest.mark.background
def test_a_quick_run_never_shows_stop(qapp, repo: SqliteRepo, figure_id: int) -> None:
    dialog = _open(_dialog_class("transform_dialog", "SeriesTransformDialog"), repo, figure_id)
    dialog.show()
    shown: list[bool] = []
    assert dialog.preview()
    _wait_for(lambda: (shown.append(dialog.stop_button.isVisible()) or not dialog.evaluating))
    assert not any(shown)
    assert dialog._preview_active
    dialog.close()


def test_a_warning_from_a_worker_thread_reaches_the_status_bar(qapp) -> None:
    bar = QStatusBar()
    applogger.set_status_bar(bar)
    try:
        worker = threading.Thread(
            target=lambda: applogger.warning("from the worker")
        )
        worker.start()
        worker.join()
        _wait_for(lambda: bar.currentMessage() == "from the worker", timeout=5.0)
    finally:
        applogger.set_status_bar(None)
