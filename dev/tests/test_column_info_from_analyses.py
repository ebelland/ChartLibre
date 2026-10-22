"""Column Info opens from an analysis' column list, and the analysis sees what it changed (R-08)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data import column_metadata as cm
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.fit_model_dialog import FitModelDialog


@pytest.fixture
def repo(tmp_path: Path):
    database = SqliteRepo(db_path=tmp_path / "ci.dhub")
    database.import_dataframe(pd.DataFrame({"grade": [1, 2, 3, 1, 2], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}),
                              table_name="t", normalize_columns=False)
    yield database
    database.close()


def test_choosing_column_info_on_a_column_asks_for_it(qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.dialogs import column_info_dialog

    # The window's own handler opens the real dialog: declined here (a modal one would wait for a click).
    monkeypatch.setattr(column_info_dialog.ColumnInfoDialog, "exec", lambda self: 0)
    dialog = FitModelDialog(repo=repo, table="t")
    asked: list[str] = []
    try:
        dialog.roles_widget.column_info_requested.connect(asked.append)
        item = dialog.roles_widget._column_list.item(0)
        dialog.roles_widget.request_column_info(item)
        assert asked == [item.data(0x0100)]  # Qt.UserRole: the column's name
        dialog.roles_widget.request_column_info(None)  # nothing under the pointer: nothing asked
        assert len(asked) == 1
    finally:
        dialog.close()


def test_the_analysis_reads_the_table_again_after_column_info(qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.dialogs import column_info_dialog

    dialog = FitModelDialog(repo=repo, table="t")
    try:
        assert dialog.roles_widget.casting().kinds["grade"] != "nominal"

        def accepted(self) -> int:  # the person fills the form and presses OK
            cm.set_column_info(repo, "t", "grade", cm.ColumnInfo(kind="nominal"))
            return 1

        monkeypatch.setattr(column_info_dialog.ColumnInfoDialog, "exec", accepted)
        dialog._open_column_info("grade")
        assert dialog.roles_widget.casting().kinds["grade"] == "nominal"
    finally:
        dialog.close()
