"""Thinning a large series out: the curve's order is kept, a coloured row never goes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo


def _table(repo: SqliteRepo, n: int = 5000) -> None:
    x = np.arange(n, dtype=float)
    colour = np.where(x % 997 == 0, "#d62728", None)  # a handful of outliers, scattered
    repo.import_dataframe(pd.DataFrame({"x": x, "y": np.sin(x / 50), "c": colour}),
                          table_name="big", normalize_columns=False)


def test_a_curve_is_thinned_in_its_own_order(repo: SqliteRepo) -> None:
    _table(repo)
    frame = repo.downsampled_series_frame("SELECT x, y FROM big ORDER BY x", threshold=500)
    x = np.asarray(frame["x"], dtype=float)
    assert 400 <= len(x) <= 600
    assert np.all(np.diff(x) > 0) and x[0] == 0  # every tenth row, in x order


def test_coloured_rows_survive_the_thinning(repo: SqliteRepo) -> None:
    _table(repo)
    frame = repo.downsampled_series_frame('SELECT x, y, c AS "color" FROM big ORDER BY x', threshold=500)
    colours = [value for value in frame["color"] if isinstance(value, str) and value]
    assert len(colours) == 6  # every one of the coloured rows: x = 0, 997, ..., 4985
