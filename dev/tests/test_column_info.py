"""R-08 Column Info: a column's name, type and description - and the analyses taking it into account."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import column_metadata as cm
from app.data.sqlite_repo import SqliteRepo


def _survey() -> pd.DataFrame:
    return pd.DataFrame({"sex": [1, 2, 1, 2, 1, 2], "score": [10.0, -999.0, 12.0, 14.0, -999.0, 9.0],
                         "grade": ["3", "4", "5", "3", "4", "5"]})


def test_the_frame_as_the_analyses_read_it() -> None:
    frame = _survey()
    read = cm.apply_to_frame(frame, {"sex": cm.ColumnInfo(kind="nominal")})
    assert read["score"].isna().sum() == 0  # a code such as -999 is the data's: Fill missing values empties it
    assert read["sex"].tolist() == frame["sex"].tolist()  # value labels taken out: the codes as stored
    assert frame["score"].isna().sum() == 0  # the table's own values are untouched
    assert cm.forced_kinds({"grade": cm.ColumnInfo(kind="ordinal"), "sex": cm.ColumnInfo()}) == {"grade": "ordinal"}


def test_kept_with_the_table_renamed_and_forgotten(repo: SqliteRepo) -> None:
    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    cm.set_column_info(repo, "survey", "score", cm.ColumnInfo(unit="pt"))
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
    dialog._apply()
    assert dialog.result() == dialog.DialogCode.Accepted
    declared = {str(row[1]): str(row[2]) for row in repo.table_info("survey")}
    assert declared.get("level") == "INTEGER" and "grade" not in declared
    info = cm.column_info(repo, "survey")["level"]
    assert (info.kind, info.unit) == ("ordinal", "stars")
    repo.undo_last()
    assert "grade" in repo.table_frame("survey").columns and cm.column_info(repo, "survey") == {}


def test_an_analysis_takes_the_forced_type_and_the_emptied_codes(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.time_series_dialog import TimeSeriesDialog

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    repo.fill_missing("survey", "score", "null", codes=["-999"])  # the table editor's Fill missing values
    cm.set_column_info(repo, "survey", "sex", cm.ColumnInfo(kind="nominal"))
    dialog = TimeSeriesDialog(repo=repo, table="survey")
    try:
        assert dialog._frame["score"].isna().sum() == 2
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
    # Column Info is the button beside the Data title, no longer repeated in the menu.
    assert not any(action.text() == "Column Info..." for action in menu.actions())


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


def test_the_arrows_go_through_the_columns_saving_each_on_the_way(qapp, repo: SqliteRepo) -> None:
    """ID52: previous / next column without closing; a changed column is saved first, one undo step each."""
    from app.dialogs.column_info_dialog import ColumnInfoDialog

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    columns = [str(row[1]) for row in repo.table_info("survey") if str(row[1]) not in ("Hide", "Selected", "_id")]
    dialog = ColumnInfoDialog(repo, "survey", columns[0])
    try:
        assert not dialog.previous_button.isEnabled() and dialog.next_button.isEnabled()
        assert dialog.position.text() == f"Column 1 of {len(columns)}"
        def last_step() -> int:  # the store keeps its last few steps: compared by the newest one's id, not a count
            return max((entry.entry_id for entry in repo.undo_entries()), default=0)

        before = last_step()
        dialog._go(1)  # nothing changed: nothing saved
        assert dialog.name.text() == columns[1] and last_step() == before
        dialog.unit.setText("pts")
        dialog._go(-1)
        assert dialog.name.text() == columns[0] and cm.column_info(repo, "survey")[columns[1]].unit == "pts"
        assert last_step() > before and repo.undo_entries()[0].label.startswith("Column Info of")
        for _step in columns[1:]:
            dialog._go(1)
        assert dialog.name.text() == columns[-1] and not dialog.next_button.isEnabled()
        dialog.reject()  # a column was saved on the way: Close reports it, so the preview reloads
        assert dialog.result() == dialog.DialogCode.Accepted
    finally:
        dialog.close()


def test_the_best_format_shows_as_many_digits_as_the_column_is_wide() -> None:
    def fits(width: int):
        return lambda text: len(text) <= width

    assert cm.best_text(1 / 3, fits(20)) == "0.3333333333333333"  # it fits as stored
    assert cm.best_text(1 / 3, fits(8)) == "0.333333"
    assert cm.best_text(1 / 3, fits(5)) == "0.333"
    assert cm.best_text(123456789.123, fits(9)) == "123456789"
    assert cm.best_text(123456789.123, fits(8)) == "1.23e+08"  # the integer digits no longer fit
    assert cm.best_text(123456789.123, fits(3)) == "#"
    assert cm.best_text("text that is long", fits(3)) == "text that is long"  # not a number: as it is


def test_the_preview_fits_a_number_to_its_columns_width(qapp, repo: SqliteRepo) -> None:
    from PySide6.QtWidgets import QStyleOptionViewItem

    from app.widgets.table_preview import TablePreviewPanel

    repo.import_dataframe(pd.DataFrame({"ratio": [1 / 3, 2 / 3], "name": ["a", "b"]}), table_name="t",
                          normalize_columns=False)
    panel = TablePreviewPanel(None, repo=repo)
    panel.set_context(repo, "t")
    qapp.processEvents()
    model, delegate = panel.view.model(), panel.view.itemDelegate()
    column = next(c for c in range(model.columnCount()) if model.column_name(c) == "ratio")

    def shown(width: int) -> str:
        option = QStyleOptionViewItem()
        option.rect.setWidth(width)
        option.fontMetrics = panel.view.fontMetrics()
        delegate.initStyleOption(option, model.index(0, column))
        return option.text

    narrow, wide = shown(50), shown(400)
    assert wide == str(1 / 3) and len(narrow) < len(wide) and narrow.startswith("0.3")
    assert float(model.data(model.index(0, column))) == 1 / 3  # stored and copied whole


def test_column_info_is_two_columns_with_the_charts_arrows(qapp, repo: SqliteRepo) -> None:
    from app.dialogs.column_info_dialog import ColumnInfoDialog

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    dialog = ColumnInfoDialog(repo, "survey", "score")
    try:
        assert dialog.column(0).isAncestorOf(dialog.name)
        assert dialog.previous_button.objectName() == dialog.next_button.objectName() == "chartJumpArrow"
        assert dialog.minimumSizeHint().height() <= 690
    finally:
        dialog.close()


def test_a_saved_query_and_a_stored_function_are_described_not_renamed(qapp, repo: SqliteRepo) -> None:
    """Column Info on a query's or a function's column: its description kept, its name and type its SQL's."""
    from app.data_operations import sympy_store
    from app.dialogs.column_info_dialog import ColumnInfoDialog

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    repo.save_query("high", 'SELECT sex, score FROM "survey" WHERE score > 0')
    sympy_store.save(repo, {"function_name": "Line", "source_type": "expression", "symbols": "a",
                            "code_string": "a*x", "values": {"a": 2.0}, "from": "0", "to": "2", "points": "3"})
    for source, column, unit, kind in (("high", "score", "pt", "REAL"), ("Line", "y", "mV", "REAL")):
        dialog = ColumnInfoDialog(repo, source, column)
        try:
            assert not dialog.name.isEnabled() and not dialog.data_type.isEnabled()
            assert dialog.data_type.currentData() == kind
            dialog.unit.setText(unit)
            dialog._apply()
            assert dialog.result() == dialog.DialogCode.Accepted
        finally:
            dialog.close()
        assert cm.column_info(repo, source)[column].unit == unit
    assert "score" in repo.table_frame("survey").columns  # the table itself untouched


def test_the_preview_of_a_query_offers_column_info(qapp, repo: SqliteRepo) -> None:
    from app.widgets.table_preview import TablePreviewPanel

    repo.import_dataframe(_survey(), table_name="survey", normalize_columns=False)
    repo.save_query("high", 'SELECT sex, score FROM "survey" WHERE score > 0')
    panel = TablePreviewPanel(None, repo=repo)
    panel.set_context(repo, "high")
    qapp.processEvents()
    panel.view.setCurrentIndex(panel.view.model().index(0, 1))
    menu = panel._build_context_menu(panel.view.visualRect(panel.view.model().index(0, 1)).center())
    texts = [action.text() for action in menu.actions()]
    assert "Column Info..." not in texts and not any(text.startswith("Statistics of") for text in texts)


def test_a_cell_with_no_value_is_missing_without_a_code(repo: SqliteRepo) -> None:
    """NULL, and empty or blank text, as JMP reads an empty character cell: not a level of its own."""
    frame = pd.DataFrame({"grade": ["A", "", "B", None, "   ", " C "], "score": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    repo.import_dataframe(frame, table_name="marks", normalize_columns=False)
    stored = repo.query_df('SELECT "grade", "score" FROM "marks"')
    read = cm.apply_to_frame(stored, {})
    assert read["grade"].isna().tolist() == [False, True, False, True, True, False]
    assert read["grade"].dropna().tolist() == ["A", "B", " C "]  # text with a value is kept as it is
    assert read["score"].isna().sum() == 1  # numbers: only the NULL


def test_blank_as_missing_leaves_numbers_and_mixed_values_alone() -> None:
    assert cm.blank_as_missing(pd.Series([1.0, np.nan])).isna().tolist() == [False, True]
    mixed = cm.blank_as_missing(pd.Series(["", 3, " x ", None], dtype=object))
    assert mixed.isna().tolist() == [True, False, False, True] and mixed[1] == 3
