"""PLS (AN-17): the components a synthetic spectrum holds, its band found by VIP; the dialog writes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.analysis.pls import pls
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.pls_dialog import PLSDialog

WAVELENGTHS = np.linspace(900, 1700, 120)


def _spectra(n: int = 60) -> pd.DataFrame:
    """Two absorbers, at 1200 and 1450 nm; only the first is the response."""
    rng = np.random.default_rng(0)
    conc, other = rng.uniform(0, 1, n), rng.uniform(0, 1, n)

    def band(centre: float, width: float) -> np.ndarray:
        return np.exp(-((WAVELENGTHS - centre) / width) ** 2)

    x = np.outer(conc, band(1200, 30)) + np.outer(other, band(1450, 60)) + 0.3 + rng.normal(0, 0.01, (n, 120))
    frame = pd.DataFrame(x, columns=[f"nm{int(w)}" for w in WAVELENGTHS])
    frame["conc"] = conc
    return frame


def test_pls_predicts_and_vip_points_at_the_band() -> None:
    frame = _spectra()
    result = pls(frame, ["conc"], [c for c in frame.columns if c != "conc"])
    assert 2 <= result.components <= 4
    assert result.summary["Q² (cross-validated)"] > 0.95
    top = result.vip.sort_values("VIP", ascending=False)["Variable"].head(5)
    assert all(1150 <= int(name[2:]) <= 1250 for name in top)
    fixed = pls(frame, ["conc"], [c for c in frame.columns if c != "conc"], components=1)
    assert fixed.components == 1 and fixed.summary["Q² (cross-validated)"] < result.summary["Q² (cross-validated)"]


def test_ok_writes_the_scores_and_draws_four_charts(qapp, repo: SqliteRepo) -> None:
    frame = _spectra()
    repo.import_dataframe(frame, table_name="nir", normalize_columns=False)
    dialog = PLSDialog(repo=repo, table="nir")
    try:
        dialog.roles_widget.set_casting({"response": ["conc"], "predictor": [c for c in frame.columns if c != "conc"]})
        assert not dialog.problems()
        dialog.ok()
        deadline = time.monotonic() + 60
        while dialog._task is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        rows = repo.table_frame("_nir_pls")
        assert {"T1", "T2", "Predicted conc"} <= set(rows.columns)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert len(figure.axes) == 4
    finally:
        dialog.close()
