"""A table imported during a dialog's Preview stays inside its savepoint: Close takes it away again."""
from __future__ import annotations

import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def _exists(repo: SqliteRepo, name: str) -> bool:
    return repo._con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def test_an_import_inside_a_preview_is_rolled_back_with_it(repo: SqliteRepo) -> None:
    repo.begin_preview_savepoint("preview_test")
    repo.import_dataframe(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}), table_name="_Preview_t",
                          normalize_columns=False)
    assert _exists(repo, "_Preview_t")
    # The import's commit used to end the savepoint, and this found it gone.
    assert repo.rollback_preview_savepoint() is True
    assert not _exists(repo, "_Preview_t")
