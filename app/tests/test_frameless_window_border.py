"""The frameless window (Windows and macOS) draws its own 1px border.

Qt.FramelessWindowHint strips every bit of native chrome - border, corner,
drop shadow - so without a border of its own the window had no visible edge
at all against whatever is behind it ("no border under Windows"). macOS got
the same treatment later: its own native title-bar-plus-traffic-lights strip
is replaced by CustomTitleBar the same way Windows' was. The central host
widget gets objectName "windowFrame" and WA_StyledBackground on both
platforms; fluent_win11.qss and macos_native.qss each draw the actual 1px
outline on that object name. Linux keeps its window manager's own native
chrome - there is no equivalent "elsewhere" gap to fill in.
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


@pytest.mark.parametrize("platform_flag", ["IS_WINDOWS", "IS_MACOS"])
def test_a_frameless_platform_gets_a_named_styled_central_host(
    repo: SqliteRepo,
    tmp_db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qapp,
    platform_flag: str,
) -> None:
    monkeypatch.setattr(main_window_module, "IS_WINDOWS", platform_flag == "IS_WINDOWS")
    monkeypatch.setattr(main_window_module, "IS_MACOS", platform_flag == "IS_MACOS")
    window = _build(repo, tmp_db_path)
    try:
        assert window._central_host.objectName() == "windowFrame"
        assert window._central_host.testAttribute(
            Qt.WidgetAttribute.WA_StyledBackground
        )
        assert window._custom_title_bar is not None
    finally:
        window.close()
        applogger.set_status_bar(None)


def test_elsewhere_the_central_host_is_left_unstyled(
    repo: SqliteRepo, tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    """Neither platform flag set (a stand-in for Linux, whose window manager
    already draws its own native title bar)."""
    monkeypatch.setattr(main_window_module, "IS_WINDOWS", False)
    monkeypatch.setattr(main_window_module, "IS_MACOS", False)
    window = _build(repo, tmp_db_path)
    try:
        assert window._central_host.objectName() != "windowFrame"
        assert window._custom_title_bar is None
    finally:
        window.close()
        applogger.set_status_bar(None)
