"""H-13: the tools for contributors show on the Project page only in Settings' Developer mode."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.main_window import main_window as main_window_module
from app.utils import config


def test_the_tools_for_contributors_follow_developer_mode(qapp, tmp_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "t.dhub")
    repo.import_dataframe(pd.DataFrame({"a": [1.0]}), table_name="t", normalize_columns=False)
    config.set_value(config.DEVELOPER_MODE_KEY, False)
    window = main_window_module.MainWindow(repo=repo, db_path=repo.db_path)
    try:
        window.show()
        window._select_navigation("developer")
        qapp.processEvents()
        names = [card.objectName() for card in window._developer_cards]
        assert names == ["edit_localizationCard", "data_operation_builderCard",
                         "function_creatorCard", "renderer_helperCard"]
        assert not any(card.isVisible() or card.parentWidget().isVisible() for card in window._developer_cards)
        assert window._left_panel._buttons["developer"].toolTip().startswith("The project's history")

        config.set_value(config.DEVELOPER_MODE_KEY, True)
        window._apply_developer_mode()
        qapp.processEvents()
        assert all(card.isVisible() and card.parentWidget().isVisible() for card in window._developer_cards)
    finally:
        config.set_value(config.DEVELOPER_MODE_KEY, False)
        window.close()
        repo.close()


def test_settings_shows_the_deepl_key_only_in_developer_mode(qapp) -> None:
    from app.dialogs.settings_dialog import SettingsDialog

    config.set_value(config.DEVELOPER_MODE_KEY, False)
    dialog = SettingsDialog()
    try:
        dialog.show()
        qapp.processEvents()
        card = dialog._deepl_key_edit.parentWidget()
        assert not dialog._developer_check.isChecked() and not dialog._deepl_key_edit.isVisibleTo(card)
        assert not card.isVisibleTo(dialog)  # Advanced starts closed when nothing in it was changed
        dialog._developer_check.setChecked(True)
        assert dialog._deepl_key_edit.isVisibleTo(card)
        dialog._save()
        assert config.developer_mode()
    finally:
        config.set_value(config.DEVELOPER_MODE_KEY, False)
        dialog.close()


def test_the_file_page_has_project_save_with_export_demo_and_recent(qapp, tmp_path: Path) -> None:
    """Import is on New table, Project info with Settings: the File page keeps the project's own steps."""
    from PySide6.QtWidgets import QFrame, QPushButton

    from app.main_window.main_window import IS_MACOS

    repo = SqliteRepo(db_path=tmp_path / "f.dhub")
    window = main_window_module.MainWindow(repo=repo, db_path=repo.db_path)
    try:
        assert window.findChild(QFrame, "fileImportCard") is None
        assert window.findChild(QFrame, "fileProjectInfoCard") is None
        save = window.findChild(QFrame, "fileSaveCard")
        assert [b.text() for b in save.findChildren(QPushButton)][-1] == "Export…"
        if IS_MACOS:
            items = [item for _title, group in window._app_menu_items() for item in group if item is not None]
            assert any("Project info" in str(getattr(item, "text", "")) for item in items)
        else:
            assert window._left_panel.button("project_info") is not None
    finally:
        window.close()
        repo.close()
