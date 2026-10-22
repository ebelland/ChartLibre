"""R-08 Column Info: a column's name, type and description - and the analyses taking it into account."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import column_metadata as cm
from app.data.sqlite_repo import SqliteRepo


def _survey() -> pd.DataFrame:
    return pd.DataFrame({"sex": [1, 2, 1, 2, 1, 2], "score": [10.0, -999.0, 12.0, 14.0, -999.0, 9.0],
                         "grade": ["3", "4", "5", "3", "4", "5"]})


def test_missing_codes_and_value_labels_in_the_frame() -> None:
    frame = _survey()
    infos = {"score": cm.ColumnInfo(missing=["-999"]), "sex": cm.ColumnInfo(value_labels={"1": "male", "2": "female"})}
    read = cm.apply_to_frame(frame, infos)
    assert read["score"].isna().sum() == 2 and read["score"].dropna().tolist() == [10.0, 12.0, 14.0, 9.0]
    assert read["sex"].tolist() == ["male", "female"] * 3
    assert frame["score"].isna().sum() == 0  # the table's own values are untouched
    assert cm.forced_kinds({"grade": cm.ColumnInfo(kind="ordinal"), "sex": cm.ColumnInfo()}) == {"grade": "ordinal"}


def test_kept_with_the_table_renamed_and_forgotten(repo: SqliteRepo) -> None:
    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    cm.set_column_info(repo, "survey", "score", cm.ColumnInfo(unit="pt", missing=["-999"]))
    assert cm.column_info(repo, "survey")["score"].unit == "pt"
    cm.rename_column_info(repo, "survey", "score", "points")
    assert set(cm.column_info(repo, "survey")) == {"points"}
    cm.set_column_info(repo, "survey", "points", cm.ColumnInfo())
    assert cm.column_info(repo, "survey") == {}


def test_the_dialog_renames_casts_and_describes_in_one_undo_step(qapp, repo: SqliteRepo) -> None:
    from app.dialogs.column_info_dialog import ColumnInfoDialog

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    dialog = ColumnInfoDialog(repo, "survey", "grade")
    dialog.name.setText("level")
    dialog.data_type.setCurrentIndex(dialog.data_type.findData("INTEGER"))
    dialog.modelling.setCurrentIndex(dialog.modelling.findData("ordinal"))
    dialog.unit.setText("stars")
    dialog._fill_from_data()
    assert [dialog.labels.item(r, 0).text() for r in range(dialog.labels.rowCount())] == ["3", "4", "5"]
    dialog.labels.item(0, 1).setText("fair")
    dialog._apply()
    assert dialog.result() == dialog.DialogCode.Accepted
    declared = {str(row[1]): str(row[2]) for row in repo.table_info("survey")}
    assert declared.get("level") == "INTEGER" and "grade" not in declared
    info = cm.column_info(repo, "survey")["level"]
    assert (info.kind, info.unit, info.value_labels) == ("ordinal", "stars", {"3": "fair"})
    repo.undo_last()
    assert "grade" in repo.table_frame("survey").columns and cm.column_info(repo, "survey") == {}


def test_an_analysis_takes_the_forced_type_and_the_missing_codes(qapp, repo: SqliteRepo) -> None:
    from app.table_operations.distribution_dialog import DistributionDialog

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    cm.set_column_info(repo, "survey", "score", cm.ColumnInfo(missing=["-999"]))
    cm.set_column_info(repo, "survey", "sex", cm.ColumnInfo(kind="nominal", value_labels={"1": "m", "2": "f"}))
    dialog = DistributionDialog(repo=repo, table="survey")
    try:
        assert dialog._frame["score"].isna().sum() == 2
        assert set(dialog._frame["sex"]) == {"m", "f"}
        assert dialog.roles_widget.casting().kinds["sex"] == "nominal"
        assert len(dialog.row_ids) == 6 and np.array_equal(dialog.row_ids.to_numpy(), np.arange(1, 7))
    finally:
        dialog.close()


def test_the_preview_menu_offers_column_info(qapp, repo: SqliteRepo) -> None:
    from app.widgets.table_preview import TablePreviewPanel

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    panel = TablePreviewPanel(None, repo=repo)
    panel.set_context(repo, "survey")
    qapp.processEvents()
    panel.view.setCurrentIndex(panel.view.model().index(0, 1))
    menu = panel._build_context_menu(panel.view.visualRect(panel.view.model().index(0, 1)).center())
    assert any(action.text() == "Column Info..." for action in menu.actions())


def test_the_format_shows_values_without_changing_them() -> None:
    assert cm.format_value(1234.5678, cm.ColumnInfo(format="fixed")) == "1234.57"
    assert cm.format_value(0.125, cm.ColumnInfo(format="percent", decimals=1)) == "12.5%"
    assert cm.format_value(1234.5678, cm.ColumnInfo(format="scientific")) == "1.23e+03"
    assert cm.format_value(1234.5678, cm.ColumnInfo(format="thousands", decimals=0)) == "1,235"
    assert cm.format_value("2024-12-31", cm.ColumnInfo(format="datetime", pattern="%d/%m/%Y")) == "31/12/2024"
    assert cm.format_value("n/d", cm.ColumnInfo(format="fixed")) == "n/d"
    assert cm.format_value(None, cm.ColumnInfo(format="fixed")) == ""
    assert cm.ColumnInfo.from_dict({"format": "nonsense", "decimals": 99}).format == ""
    assert cm.ColumnInfo.from_dict({"format": "fixed", "decimals": 99}).decimals == 15


def test_the_preview_shows_the_format_and_copies_the_stored_value(qapp, repo: SqliteRepo) -> None:
    from PySide6.QtCore import Qt

    from app.dialogs.column_info_dialog import ColumnInfoDialog
    from app.widgets.table_preview import TablePreviewPanel

    repo.import_dataframe(pd.DataFrame({"share": [0.125, 0.5]}), table_name="t", normalize_columns=False)
    dialog = ColumnInfoDialog(repo, "t", "share")
    dialog.format.setCurrentIndex(dialog.format.findData("percent"))
    dialog.decimals.setValue(1)
    assert dialog.example.text() == "0.125 is shown 12.5%"
    dialog._apply()
    panel = TablePreviewPanel(None, repo=repo)
    panel.set_context(repo, "t")
    qapp.processEvents()
    model = panel.view.model()
    column = next(c for c in range(model.columnCount()) if model._columns[c] == "share")
    index = model.index(0, column)
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "12.5%"
    assert float(model.data(index, Qt.ItemDataRole.EditRole)) == 0.125


def test_value_labels_are_what_the_preview_shows_for_a_stored_value() -> None:
    info = cm.ColumnInfo(value_labels={"1": "male", "2": "female"})
    assert cm.format_value(1, info) == "male"
    assert cm.format_value(2.0, info) == "female"  # a number column keeps integers as floats
    assert cm.format_value("1", info) == "male"
    assert cm.format_value(3, info) == "3"  # no label: as stored
    assert cm.format_value(None, info) == ""
    # a label wins over the number format of the same column
    both = cm.ColumnInfo(format="fixed", decimals=1, value_labels={"1": "male"})
    assert cm.format_value(1, both) == "male" and cm.format_value(2, both) == "2.0"


def test_the_preview_shows_value_labels_and_edits_the_stored_value(qapp, repo: SqliteRepo) -> None:
    from PySide6.QtCore import Qt

    from app.widgets.table_preview import TablePreviewPanel

    repo.import_dataframe(pd.DataFrame({"sex": [1, 2, 1, 3]}), table_name="people", normalize_columns=False)
    cm.set_column_info(repo, "people", "sex", cm.ColumnInfo(value_labels={"1": "male", "2": "female"}))
    panel = TablePreviewPanel(None, repo=repo)
    panel.set_context(repo, "people")
    qapp.processEvents()
    model = panel.view.model()
    column = next(c for c in range(model.columnCount()) if model._columns[c] == "sex")
    shown = [model.data(model.index(row, column), Qt.ItemDataRole.DisplayRole) for row in range(4)]
    stored = [model.data(model.index(row, column), Qt.ItemDataRole.EditRole) for row in range(4)]
    assert shown == ["male", "female", "male", "3"]
    assert stored == ["1", "2", "1", "3"]  # editing and copying see the values
