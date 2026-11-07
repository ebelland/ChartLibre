"""The File page's rows and Settings' log option (the developer tools went to dev/ChartLibreDeveloper.py)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.main_window import main_window as main_window_module
from app.utils import config


def test_the_file_page_has_project_save_with_export_demo_and_recent(qapp, tmp_path: Path) -> None:
    """Import is on New table, Project info with Settings: the File page keeps the project's own steps."""
    from PySide6.QtWidgets import QFrame, QPushButton

    repo = SqliteRepo(db_path=tmp_path / "f.dhub")
    window = main_window_module.MainWindow(repo=repo, db_path=repo.db_path)
    try:
        assert window.findChild(QFrame, "fileImportCard") is None
        assert window.findChild(QFrame, "fileProjectInfoCard") is None
        save = window.findChild(QFrame, "fileSaveCard")
        assert [b.text() for b in save.findChildren(QPushButton)][-1] == "Export…"
        # Project info is on the Project page, with what else is about the open project.
        assert window.findChild(QFrame, "projectInfoCard") is not None
        assert window._left_panel.button("project_info") is None
    finally:
        window.close()
        repo.close()


def test_settings_offers_to_clear_the_log_at_start(qapp) -> None:
    """Settings > Advanced: "Clear the log at start", kept in user.json; ChartLibre.py empties the log when it is on."""
    from app.dialogs.settings_dialog import SettingsDialog

    dialog = SettingsDialog()
    try:
        assert not dialog._clear_log_check.isChecked()  # off unless asked
        dialog._clear_log_check.setChecked(True)
        dialog._save()
    finally:
        dialog.close()
    assert config.clear_log_at_start()
    assert "clear_log_at_start()" in Path(__file__).resolve().parents[2].joinpath("ChartLibre.py").read_text()
    config.set_value(config.CLEAR_LOG_KEY, False)
