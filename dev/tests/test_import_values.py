"""Import: missing values and dates written the local way - read as empty cells, numbers and ISO dates."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.utils import import_cleaning as ic

_CSV = ("day;reading;note\n"
        "31/12/2024;1.5;ok\n"
        "01/02/2025;n/d;\n"
        "15/03/2025;2.5;  \n")


def test_the_order_is_read_from_the_values_else_the_computers() -> None:
    assert ic.guess_date_order(["31/12/2024", "01/02/2025"]) == ic.DAY_FIRST
    assert ic.guess_date_order(["12/31/2024", "01/02/2025"]) == ic.MONTH_FIRST
    assert ic.guess_date_order(["2024-12-31"]) == ic.YEAR_FIRST
    assert ic.guess_date_order(["01/02/2025"], default=ic.DAY_FIRST) == ic.DAY_FIRST
    assert ic.convert_dates(pd.Series(["01/02/2025"]), "DATE", default=ic.DAY_FIRST).tolist() == ["2025-02-01"]
    assert ic.convert_dates(pd.Series(["01.02.2025 10:30"]), "DATETIME", ic.FORMAT, "%d.%m.%Y %H:%M").tolist() == [
        "2025-02-01 10:30:00"]


def test_missing_codes_and_blank_text_become_empty() -> None:
    frame = pd.DataFrame({"x": ["1", "n/d", " ", "#N/A"], "y": [1.0, 2.0, 3.0, 4.0]})
    cleaned = ic.clean_missing(frame)
    assert cleaned["x"].isna().tolist() == [False, True, True, True]
    kept = ic.clean_missing(frame, ["#N/A"], empty_is_missing=False)
    assert kept["x"].isna().tolist() == [False, False, False, True]


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "log.csv"
    path.write_text(_CSV, encoding="utf-8")
    return path


def test_the_dialog_types_and_imports_them(qapp, tmp_path: Path, csv_file: Path) -> None:
    from app.dialogs.import_data_dialog import ImportDataDialog

    repo = SqliteRepo(db_path=tmp_path / "values.dhub")
    dialog = ImportDataDialog(repo)
    try:
        dialog.load_file(csv_file)
        assert dialog._df["reading"].isna().tolist() == [False, True, False]
        assert dialog._df["note"].isna().tolist() == [False, True, True]
        types = dialog._selected_types()
        assert types["day"] == "DATE" and types["reading"] == "REAL"
        dialog._table.setText("log")
        dialog._on_accept()
        table = repo.table_frame("log")
        assert table["day"].tolist() == ["2024-12-31", "2025-02-01", "2025-03-15"]
        assert table["reading"].isna().tolist() == [False, True, False]
        # Update link reads the file the same way.
        from app.utils.import_runner import refresh_link

        csv_file.write_text(_CSV + "02/04/2025;3.5;x\n", encoding="utf-8")
        link = repo.get_table_link("log")
        refresh_link(repo, link_id=int(link["id"]))
        assert repo.table_frame("log")["day"].tolist()[-1] == "2025-04-02"
    finally:
        repo.close()
