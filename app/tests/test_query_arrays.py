"""SqliteRepo.query_arrays: query_df's numeric-only sibling.

query_df().to_numpy() was the pattern at the overwhelming majority of its
call sites - a DataFrame built, then immediately discarded for a plain
array a line or two later. query_arrays reads straight from the DB-API
cursor instead, and never builds the DataFrame at all. These tests pin it
against query_df's own output wherever the two should agree, so a future
change to either cannot let them quietly drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def counts_table(repo: SqliteRepo) -> SqliteRepo:
    repo.import_dataframe(
        pd.DataFrame(
            {
                "sample": [1, 2, 3, 4],
                "defectives": [3, "5", None, 2],  # a mixed column, on purpose
                "note": ["a", "b", "c", "d"],
            }
        ),
        table_name="qc",
        normalize_columns=False,
    )
    return repo


# ----------------------------------------------------------------------
# Basic shape
# ----------------------------------------------------------------------
def test_the_column_names_match_the_select(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays("SELECT sample, note FROM qc")
    assert result.columns == ("sample", "note")


def test_a_numeric_column_comes_back_as_float64(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays(
        "SELECT sample FROM qc ORDER BY sample", numeric=("sample",)
    )
    assert result["sample"].dtype == np.float64
    assert list(result["sample"]) == [1.0, 2.0, 3.0, 4.0]


# ----------------------------------------------------------------------
# Coercion - matching pd.to_numeric(errors="coerce")
# ----------------------------------------------------------------------
def test_none_and_unparseable_values_become_nan(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays(
        "SELECT sample, defectives FROM qc ORDER BY sample", numeric=("defectives",)
    )
    defectives = result["defectives"]
    assert defectives[0] == pytest.approx(3.0)   # int, stored as int
    assert defectives[1] == pytest.approx(5.0)   # "5", a numeric string
    assert np.isnan(defectives[2])               # None
    assert defectives[3] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# QueryColumns itself
# ----------------------------------------------------------------------


