"""The status bar: quiet while all is well, amber on a warning, red on an error until dismissed."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.main_window import main_window as main_window_module


def _window(qapp, tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "s.dhub")
    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0]}), table_name="t", normalize_columns=False)
    window = main_window_module.MainWindow(repo=repo, db_path=repo.db_path)
    window.show()
    qapp.processEvents()
    return repo, window


def test_an_error_turns_the_bar_red_and_stays_until_dismissed(qapp, tmp_path: Path) -> None:
    repo, window = _window(qapp, tmp_path)
    try:
        assert window._status_current == "normal"
        assert "#FDECEC" not in window.statusBar().styleSheet()
        applogger.error("The chart could not be drawn.\nTraceback follows")
        qapp.processEvents()
        assert window._status_current == "error"
        assert "#FDECEC" in window.statusBar().styleSheet()
        # The first line only: a status bar has one.
        assert window._status_state.text() == "The chart could not be drawn."
        assert window._status_actions.isVisibleTo(window)
        # A warning does not cover the error still showing.
        applogger.warning("Something minor")
        qapp.processEvents()
        assert window._status_current == "error"
        window._on_status_action("dismiss")
        assert window._status_current == "normal"
        assert window._status_state.text() == "Ready"
        assert not window._status_actions.isVisibleTo(window)
    finally:
        window.close()
        repo.close()


def test_a_warning_is_amber_and_fades_back_to_ready(qapp, tmp_path: Path) -> None:
    repo, window = _window(qapp, tmp_path)
    try:
        applogger.warning("Twelve rows were skipped")
        qapp.processEvents()
        assert window._status_current == "warning"
        assert window._status_timer.isActive()
        window._status_timer.timeout.emit()
        assert window._status_current == "normal"
        # Information is not news: the bar stays as it is.
        applogger.info("Saved")
        qapp.processEvents()
        assert window._status_current == "normal"
    finally:
        window.close()
        repo.close()
