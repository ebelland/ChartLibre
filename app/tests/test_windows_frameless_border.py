"""The frameless window (Windows only) draws its own 1px border.

Qt.FramelessWindowHint strips every bit of native chrome - border, corner,
drop shadow - so without a border of its own the window had no visible edge
at all against whatever is behind it ("no border under Windows"). The
central host widget gets objectName "windowFrame" and WA_StyledBackground
only when IS_WINDOWS; fluent_win11.qss draws the actual 1px outline on that
object name.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger
import app.dialogs.main_window as main_window_module


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


def _build(repo: SqliteRepo, tmp_db_path: Path) -> MainWindow:
    return MainWindow(repo=repo, db_path=tmp_db_path)


def test_windows_gets_a_named_styled_central_host(
    repo: SqliteRepo, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    monkeypatch.setattr(main_window_module, "IS_WINDOWS", True)
    window = _build(repo, tmp_db_path)
    try:
        assert window._central_host.objectName() == "windowFrame"
        assert window._central_host.testAttribute(
            Qt.WidgetAttribute.WA_StyledBackground
        )
    finally:
        window.close()
        applogger.set_status_bar(None)


def test_elsewhere_the_central_host_is_left_unstyled(
    repo: SqliteRepo, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    monkeypatch.setattr(main_window_module, "IS_WINDOWS", False)
    window = _build(repo, tmp_db_path)
    try:
        assert window._central_host.objectName() != "windowFrame"
    finally:
        window.close()
        applogger.set_status_bar(None)
