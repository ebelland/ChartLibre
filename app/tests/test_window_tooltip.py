"""The main window itself must carry no tooltip.

Qt forwards an unhandled ToolTip event up the parent chain: a widget with no
tooltip of its own falls back to showing its nearest ancestor's, all the way
up to the top-level window. ``_update_window_title`` used to call
``self.setToolTip(str(db_path))`` on the MainWindow itself, which made the
open project's full .dhub path the fallback tooltip for almost every control
in the app that had not been given a more specific one of its own - reported
as "tooltips mostly show the database name instead of the widget". The status
bar's own project label is the one place that path belongs, and it keeps it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(pd.DataFrame({"x": [1, 2, 3]}), table_name="t", normalize_columns=False)
    yield built
    built.close()


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def test_the_main_window_carries_no_tooltip_of_its_own(window: MainWindow) -> None:
    assert window.toolTip() == ""


def test_the_status_bar_project_label_still_carries_the_path(
    window: MainWindow, tmp_db_path: Path
) -> None:
    assert window._status_project.toolTip() == str(tmp_db_path)


def test_reopening_a_project_does_not_reintroduce_the_window_tooltip(
    window: MainWindow,
) -> None:
    """``_update_window_title`` also runs on open/save, not just at
    construction - the guard has to hold on every call, not only the first."""
    window._update_window_title()
    assert window.toolTip() == ""
