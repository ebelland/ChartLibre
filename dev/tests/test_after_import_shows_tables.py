"""After an import, a new table or a project opened, the Tables page is shown with the new table selected."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.main_window import main_window as main_window_module


def _window(qapp, tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "t.dhub")
    repo.import_dataframe(pd.DataFrame({"a": [1.0, 2.0]}), table_name="first", normalize_columns=False)
    repo.import_dataframe(pd.DataFrame({"b": [3.0, 4.0]}), table_name="last_imported", normalize_columns=False)
    window = main_window_module.MainWindow(repo=repo, db_path=repo.db_path)
    window.show()
    qapp.processEvents()
    return repo, window


def test_show_table_goes_to_tables_and_selects_the_table(qapp, tmp_path: Path) -> None:
    repo, window = _window(qapp, tmp_path)
    try:
        window._select_navigation("developer")
        window._show_table("last_imported")
        assert window._left_panel.selected_key == "tables"
        assert window._table_panel.current == "last_imported"
    finally:
        window.close()
        repo.close()


def test_switching_project_goes_to_tables(qapp, tmp_path: Path) -> None:
    repo, window = _window(qapp, tmp_path)
    try:
        other = tmp_path / "other.dhub"
        window._select_navigation("developer")
        window._switch_database(other)
        assert window._left_panel.selected_key == "tables"
    finally:
        window.close()
        window._repo.close()


def test_import_from_clipboard_opens_the_dialog_on_it_and_honours_the_transposed_box(qapp, tmp_path: Path, monkeypatch) -> None:
    from app.dialogs import import_data_dialog as dialog_module

    repo, window = _window(qapp, tmp_path)
    opened: list[tuple[str, bool]] = []
    try:
        qapp.clipboard().setText("alpha\tbeta\n1\t2\n3\t4\n")
        monkeypatch.setattr(dialog_module.ImportDataDialog, "exec",
                            lambda self: opened.append((self._source_mode, self._clipboard_transposed)) or 0)
        window._clipboard_transposed.setChecked(False)
        window._on_import_clipboard()
        window._clipboard_transposed.setChecked(True)
        window._on_import_clipboard()
        assert opened == [("clipboard", False), ("clipboard", True)]
        # an empty clipboard says so and opens nothing
        qapp.clipboard().setText("")
        monkeypatch.setattr(dialog_module, "show_message", lambda *a, **k: None)
        window._on_import_clipboard()
        assert len(opened) == 2
    finally:
        window.close()
        repo.close()


def test_the_import_button_does_not_open_a_file_called_false(qapp, tmp_path: Path, monkeypatch) -> None:
    from app.dialogs import import_data_dialog as dialog_module

    repo, window = _window(qapp, tmp_path)
    loaded: list[object] = []
    try:
        monkeypatch.setattr(dialog_module.ImportDataDialog, "exec", lambda self: 0)
        monkeypatch.setattr(dialog_module.ImportDataDialog, "load_file", lambda self, path: loaded.append(path))
        window._on_import_data(False)  # what a button's clicked(bool) delivers
        assert loaded == []
        window._on_import_data(tmp_path / "data.csv")
        assert loaded == [tmp_path / "data.csv"]
    finally:
        window.close()
        repo.close()


def test_the_workspace_row_stays_selected_while_the_panels_are_hidden(qapp, tmp_path: Path) -> None:
    repo, window = _window(qapp, tmp_path)
    try:
        bar = window._left_panel
        bar.select("tables")
        assert bar.button("tables").isChecked() and not bar.button("workspace").isChecked()
        bar.select("workspace")  # the action row: hides the panels
        assert not bar.panel_visible
        assert bar.button("workspace").isChecked() and not bar.button("tables").isChecked()
        bar.select("tables")  # shows them again
        assert bar.button("tables").isChecked() and not bar.button("workspace").isChecked()
    finally:
        window.close()
        repo.close()
