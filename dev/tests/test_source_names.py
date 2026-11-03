"""Tables, saved queries and stored functions share one namespace: a name is one source's."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


def test_a_name_belongs_to_one_source(tmp_path) -> None:
    repo = SqliteRepo(db_path=tmp_path / "names.dhub")
    try:
        repo.import_dataframe(pd.DataFrame({"x": [1, 2]}), table_name="Query 1")
        repo.save_sympy_function("Function1", "expression", "a", "a*x")
        repo.save_query("Means", 'SELECT * FROM "Query 1"')

        assert (repo.source_kind("query 1"), repo.source_kind("MEANS"), repo.source_kind("function1")) == \
            ("table", "query", "function")
        assert repo.next_query_name() == "Query 2"  # a table holds "Query 1"
        assert repo.next_source_name("Function", separator="") == "Function2"

        with pytest.raises(ValueError, match="table"):
            repo.save_query("query 1", "SELECT 1")
        with pytest.raises(ValueError, match="query"):
            repo.save_sympy_function("means", "expression", "", "x")
        with pytest.raises(ValueError, match="function"):
            repo.save_query("Function1", "SELECT 1")
        repo.save_query("Means", "SELECT 2")  # the same source again: an update
        assert repo.get_query("Means").sql == "SELECT 2"
    finally:
        repo.close()
