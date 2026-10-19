"""AN-04: correlations with p-values and intervals, and principal components - checked against SciPy and scikit-learn."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.analysis import multivariate as mv


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    a = rng.normal(size=60)
    return pd.DataFrame({"a": a, "b": 2 * a + rng.normal(scale=0.5, size=60),
                         "c": rng.normal(size=60) * 10, "d": -a + rng.normal(scale=2, size=60)})


def test_pearson_pairs_match_scipy_with_a_fisher_interval(frame: pd.DataFrame) -> None:
    frame.loc[3, "c"] = np.nan
    result = mv.correlations(frame, ["a", "b", "c"])
    row = result.pairs.set_index(["Variable", "by Variable"]).loc[("a", "c")]
    both = frame[["a", "c"]].dropna()
    r, p = stats.pearsonr(both["a"], both["c"])
    assert row["Correlation"] == pytest.approx(r) and row["Prob>|t|"] == pytest.approx(p) and row["N"] == 59
    interval = stats.pearsonr(both["a"], both["c"]).confidence_interval(0.95)
    assert row["Lower 95%"] == pytest.approx(interval.low) and row["Upper 95%"] == pytest.approx(interval.high)
    matrix = result.matrix.set_index("Variable")
    assert matrix.loc["a", "a"] == 1 and matrix.loc["c", "a"] == pytest.approx(r)
    assert result.notes  # pairwise: a row misses a value


def test_spearman_and_kendall_are_scipys(frame: pd.DataFrame) -> None:
    spearman = mv.correlations(frame, ["a", "b"], method=mv.SPEARMAN).pairs.iloc[0]
    assert spearman["Correlation"] == pytest.approx(stats.spearmanr(frame["a"], frame["b"])[0])
    kendall = mv.correlations(frame, ["a", "b"], method=mv.KENDALL).pairs.iloc[0]
    assert kendall["Correlation"] == pytest.approx(stats.kendalltau(frame["a"], frame["b"])[0])
    assert np.isnan(kendall["Lower 95%"])


def test_principal_components_match_scikit_learn(frame: pd.DataFrame) -> None:
    from sklearn.decomposition import PCA

    columns = ["a", "b", "c", "d"]
    result = mv.principal_components(frame, columns)
    standardized = (frame - frame.mean()) / frame.std(ddof=1)
    reference = PCA().fit(standardized.to_numpy())
    assert result.eigenvalues["Eigenvalue"].to_numpy() == pytest.approx(reference.explained_variance_)
    assert result.eigenvalues["Cum Percent"].iloc[-1] == pytest.approx(100.0)
    assert result.eigenvalues["Eigenvalue"].sum() == pytest.approx(len(columns))  # a correlation matrix's trace
    # Scores up to each component's sign; loadings are the columns' correlations with the scores.
    scores = result.scores.to_numpy()
    assert np.abs(scores) == pytest.approx(np.abs(reference.transform(standardized.to_numpy())))
    loadings = result.loadings.set_index("Variable")
    for k, name in enumerate(result.scores.columns[:2]):
        for column in columns:
            assert loadings.loc[column, name] == pytest.approx(np.corrcoef(frame[column], scores[:, k])[0, 1])
    vectors = result.eigenvectors.set_index("Variable").to_numpy()
    assert all(vectors[np.argmax(np.abs(vectors[:, k])), k] > 0 for k in range(vectors.shape[1]))


def test_on_covariances_the_loadings_are_still_correlations(frame: pd.DataFrame) -> None:
    result = mv.principal_components(frame, ["a", "c"], on=mv.COVARIANCES)
    assert result.eigenvalues["Eigenvalue"].sum() == pytest.approx(frame[["a", "c"]].var(ddof=1).sum())
    first = result.scores["Prin1"]
    loadings = result.loadings.set_index("Variable")
    assert loadings.loc["c", "Prin1"] == pytest.approx(np.corrcoef(frame["c"], first)[0, 1])


def test_what_cannot_be_computed_is_said(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="two columns"):
        mv.correlations(frame, ["a"])
    frame["k"] = 1.0
    with pytest.raises(ValueError, match="does not vary"):
        mv.principal_components(frame, ["a", "k"])


def _run(qapp, dialog) -> None:
    import time

    dialog.ok()
    deadline = time.monotonic() + 60
    while dialog._task is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def test_ok_writes_the_tables_and_draws_the_charts(qapp, tmp_path, frame: pd.DataFrame) -> None:
    from app.data.sqlite_repo import SqliteRepo
    from app.table_operations.multivariate_dialog import MultivariateDialog
    from app.table_operations.principal_components_dialog import PrincipalComponentsDialog

    repo = SqliteRepo(db_path=tmp_path / "m.dhub")
    repo.import_dataframe(frame, table_name="m", normalize_columns=False)
    try:
        dialog = MultivariateDialog(repo=repo, table="m")
        dialog.roles_widget.set_casting({"response": ["a", "b", "c"]})
        assert not dialog.problems()
        _run(qapp, dialog)
        assert len(repo.table_frame("_m_correlation_pairs")) == 3
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Pair Plot"]
        dialog.close()

        dialog = PrincipalComponentsDialog(repo=repo, table="m")
        dialog.roles_widget.set_casting({"response": ["a", "b", "c", "d"]})
        _run(qapp, dialog)
        rows = repo.table_frame("_m_pca")
        assert {"Prin1", "Prin4"} <= set(rows.columns)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        # scree, scores, a loading per column, the biplot with an arrow per column
        assert len(figure.axes) == 4 and len(figure.axes[2].series) == 4
        assert [a["type"] for a in figure.axes[3].options["annotations"]].count("arrow") == 4
        dialog.close()

        repo.import_dataframe(frame.assign(site=["north", "south", "east"] * 20), table_name="g",
                              normalize_columns=False)
        dialog = PrincipalComponentsDialog(repo=repo, table="g")
        dialog.roles_widget.set_casting({"response": ["a", "b", "c"], "by": ["site"]})
        assert not dialog.problems()
        _run(qapp, dialog)
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [series.name for series in figure.axes[1].series] == ["east", "north", "south"]
        dialog.roles_widget.set_casting({"response": ["a", "b", "site"], "by": ["site"]})
        assert dialog.problems()
        dialog.close()
    finally:
        repo.close()
