"""AN-04 Factor Analysis: two factors put in the data come back, rotated; checked against scikit-learn's varimax."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from app.analysis import multivariate as mv


def _two_factors(correlation: float = 0.0, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    first = rng.normal(size=n)
    second = correlation * first + np.sqrt(1 - correlation ** 2) * rng.normal(size=n)
    columns = {f"a{i}": 0.8 * first + rng.normal(0, 0.6, n) for i in range(3)}
    columns |= {f"b{i}": 0.8 * second + rng.normal(0, 0.6, n) for i in range(3)}
    return pd.DataFrame(columns)


@pytest.mark.parametrize("method", [mv.ML, mv.PRINCIPAL_AXIS])
def test_varimax_finds_the_two_factors(method: str) -> None:
    frame = _two_factors()
    result = mv.factor_analysis(frame, list(frame.columns), method=method)
    assert result.factors == 2  # the eigenvalues above 1
    loadings = result.loadings.set_index("Variable")
    assert (loadings.loc[["a0", "a1", "a2"]].abs().idxmax(axis=1) == "Factor1").all()
    assert (loadings.loc[["b0", "b1", "b2"]].abs().idxmax(axis=1) == "Factor2").all()
    assert loadings.abs().max().min() == pytest.approx(0.8, abs=0.05)
    assert result.communalities["Communality"].between(0.55, 0.75).all()
    assert result.variance["Cum Percent"].iloc[-1] == pytest.approx(result.summary["Common variance explained %"])


def test_varimax_matches_scikit_learn() -> None:
    from sklearn.decomposition import FactorAnalysis

    frame = _two_factors(n=500)
    ours = mv.factor_analysis(frame, list(frame.columns), factors=2).loadings[["Factor1", "Factor2"]].to_numpy()
    standardized = (frame - frame.mean()) / frame.std(ddof=1)
    theirs = FactorAnalysis(2, rotation="varimax", random_state=0).fit(standardized).components_.T
    # The same up to each factor's sign and their order.
    matched = np.column_stack([theirs[:, np.argmax(np.abs(ours[:, k] @ theirs))] for k in range(2)])
    assert np.abs(ours) == pytest.approx(np.abs(matched), abs=0.01)


def test_promax_gives_the_factors_correlation() -> None:
    frame = _two_factors(correlation=0.5)
    result = mv.factor_analysis(frame, list(frame.columns), rotation=mv.PROMAX)
    assert result.factor_correlations.set_index("Factor").loc["Factor1", "Factor2"] == pytest.approx(0.5, abs=0.06)
    assert result.variance["Cum Percent"].isna().all()


def test_what_cannot_be_computed_is_said() -> None:
    frame = _two_factors(n=50)
    with pytest.raises(ValueError, match="three columns"):
        mv.factor_analysis(frame, ["a0", "a1"])
    with pytest.raises(ValueError, match="fewer factors than columns"):
        mv.factor_analysis(frame, ["a0", "a1", "a2"], factors=3)
    with pytest.raises(ValueError, match="does not vary"):
        mv.factor_analysis(frame.assign(flat=1.0), ["a0", "a1", "flat"])


def test_ok_writes_the_scores_and_draws_the_charts(qapp, tmp_path) -> None:
    from matplotlib.figure import Figure

    from app.data.sqlite_repo import SqliteRepo
    from app.plugin_api.charts.render_figure import render_figure_from_descriptor
    from app.table_operations.factor_analysis_dialog import FactorAnalysisDialog

    repo = SqliteRepo(db_path=tmp_path / "f.dhub")
    repo.import_dataframe(_two_factors(n=200), table_name="survey", normalize_columns=False)
    dialog = FactorAnalysisDialog(repo=repo, table="survey")
    try:
        dialog.roles_widget.set_casting({"response": ["a0", "a1", "a2", "b0", "b1", "b2"]})
        assert not dialog.problems()
        dialog.ok()
        deadline = time.monotonic() + 60
        while dialog._task is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert {"Factor1", "Factor2"} <= set(repo.table_frame("_survey_factors").columns)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.title for axis in figure.axes] == ["Scree plot", "Factor loading plot", "Score plot"]
        render_figure_from_descriptor(figure=Figure(), descriptor=figure, repo=repo)
    finally:
        dialog.close()
        repo.close()
