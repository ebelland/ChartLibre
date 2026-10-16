"""Discriminant analysis (AN-20): the numbers against scikit-learn and the known iris values; the dialog writes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from app.analysis.discriminant import QUADRATIC, discriminant
from app.data.sqlite_repo import SqliteRepo
from app.table_operations.discriminant_dialog import DiscriminantDialog


def _iris() -> tuple[pd.DataFrame, list[str]]:
    data = load_iris(as_frame=True)
    frame = data.frame.drop(columns="target")
    frame.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    frame["species"] = np.array(data.target_names)[data.target]
    return frame, list(frame.columns[:4])


def test_linear_matches_the_known_iris_values() -> None:
    frame, covariates = _iris()
    result = discriminant(frame, "species", covariates)
    assert result.summary["Misclassified"] == 3
    assert result.summary["Wilks' lambda"] == pytest.approx(0.023439, abs=1e-6)
    assert result.summary["Approx. F"] == pytest.approx(199.145, abs=1e-3)
    assert (result.summary["DF num"], result.summary["DF den"]) == (8.0, 288.0)
    np.testing.assert_allclose(result.canonical["Eigenvalue"], [32.1919, 0.2854], atol=1e-4)
    sklearn = LinearDiscriminantAnalysis().fit(frame[covariates], frame["species"])
    # The same canonical directions as scikit-learn; its scale moved in 1.9.1
    # (by sqrt((n - k) / n)), so the comparison is by correlation, and the
    # scale is checked on its own: unit pooled within-group variance.
    ours, theirs = result.scores.to_numpy(), sklearn.transform(frame[covariates])
    for column in range(ours.shape[1]):
        assert abs(np.corrcoef(ours[:, column], theirs[:, column])[0, 1]) == pytest.approx(1.0, abs=1e-9)
    groups = frame["species"].to_numpy()
    within = sum(((ours[groups == g] - ours[groups == g].mean(axis=0)) ** 2).sum(axis=0) for g in set(groups))
    np.testing.assert_allclose(within / (len(groups) - 3), 1.0, atol=1e-9)
    assert (result.predicted == sklearn.predict(frame[covariates])).all()


def test_quadratic_matches_scikit_learn() -> None:
    frame, covariates = _iris()
    result = discriminant(frame, "species", covariates, method=QUADRATIC)
    # Worked here rather than read from scikit-learn, whose QDA moved in 1.9.1:
    # each group's normal (its own covariance, n - 1) times its share of the rows.
    from scipy.stats import multivariate_normal

    x, groups = frame[covariates].to_numpy(), frame["species"].to_numpy()
    levels = sorted(set(groups))
    log = np.column_stack([
        np.log(np.mean(groups == g)) + multivariate_normal(x[groups == g].mean(axis=0),
                                                           np.cov(x[groups == g], rowvar=False)).logpdf(x)
        for g in levels
    ])
    expected = np.exp(log - log.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result.probabilities.to_numpy(), expected, atol=1e-9)
    assert (result.predicted == QuadraticDiscriminantAnalysis().fit(x, groups).predict(x)).all()


def test_rows_without_a_response_are_classified_and_without_a_covariate_are_not() -> None:
    frame, covariates = _iris()
    frame.loc[0, "species"] = None
    frame.loc[1, "petal_width"] = np.nan
    result = discriminant(frame, "species", covariates)
    assert result.summary["Rows"] == 148
    assert result.predicted[0] == "setosa"
    assert pd.isna(result.predicted[1])
    assert result.notes


def _wait(qapp, dialog: DiscriminantDialog) -> None:
    deadline = time.monotonic() + 60
    while dialog._task is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert dialog._task is None, "the computation did not finish"


def test_ok_writes_the_rows_and_draws_the_canonical_plot(qapp, repo: SqliteRepo) -> None:
    frame, covariates = _iris()
    repo.import_dataframe(frame, table_name="iris", normalize_columns=False)
    dialog = DiscriminantDialog(repo=repo, table="iris")
    try:
        dialog.roles_widget.set_casting({"response": ["species"], "covariate": covariates},
                                        {"species": "nominal", **{name: "continuous" for name in covariates}})
        assert not dialog.problems()
        dialog.ok()
        _wait(qapp, dialog)
        rows = repo.table_frame("iris_discriminant")
        assert {"Predicted species", "Prob(setosa)", "Canonical1", "Canonical2"} <= set(rows.columns)
        assert {"iris_discriminant_canonical", "iris_discriminant_coefficients",
                "iris_discriminant_means"} <= set(repo.list_user_tables()["Table"])
        figure = repo.load_figure_descriptor(dialog.created_figure_ids[0])
        assert [axis.name for axis in figure.axes] == ["Scatter Plot"]
        assert len(figure.axes[0].series) == 3  # a series per species
        from matplotlib.figure import Figure

        from app.plugin_api.charts.render_figure import render_figure_from_descriptor

        drawn = Figure()
        render_figure_from_descriptor(figure=drawn, descriptor=figure, repo=repo)
        assert drawn.axes[0].collections
    finally:
        dialog.close()
