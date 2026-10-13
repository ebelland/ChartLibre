"""Discriminant analysis: which group a row belongs to, from continuous measures (AN-20).

As JMP's Discriminant platform: each group of the response is a cloud of
points in the covariates' space, and a row goes to the group whose cloud it
most likely belongs to.

* Linear (LDA): the groups share one covariance - the pooled within-group
  one - so the boundaries between groups are planes.
* Quadratic (QDA): each group has its own covariance; the boundaries curve.

Priors are the groups' shares of the rows (proportional) or equal. The
canonical analysis - the directions that best separate the groups, as
eigenvectors of the between-group against the within-group spread - gives
the canonical plot and Wilks' lambda, the test that the groups differ at all.

Plain numpy and SciPy: the classifiers are a few lines each, and the numbers
are checked against scikit-learn's in dev/tests/test_discriminant.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, stats

LINEAR = "linear"
QUADRATIC = "quadratic"
METHODS: tuple[tuple[str, str], ...] = ((LINEAR, "Linear (common covariance)"), (QUADRATIC, "Quadratic (own covariance)"))

PROPORTIONAL = "proportional"
EQUAL = "equal"
PRIORS: tuple[tuple[str, str], ...] = ((PROPORTIONAL, "Proportional to the groups"), (EQUAL, "Equal"))


@dataclass(slots=True)
class DiscriminantResult:
    response: str
    covariates: list[str]
    levels: list[Any]
    method: str
    summary: dict[str, Any]
    #: Canonical, Eigenvalue, Percent, Cum Percent, Canonical Corr.
    canonical: pd.DataFrame
    #: Covariate, then a standardized coefficient per canonical variable.
    coefficients: pd.DataFrame
    #: Level, Count, then each covariate's mean.
    means: pd.DataFrame
    #: Actual levels by predicted levels, on the rows with a response.
    confusion: pd.DataFrame
    #: Per row of the input frame (NaN where a covariate is missing).
    predicted: pd.Series
    probabilities: pd.DataFrame
    scores: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _ordered_levels(values: pd.Series) -> list[Any]:
    found = list(pd.unique(values))
    try:
        return sorted(found, key=float)
    except (TypeError, ValueError):
        return sorted(found, key=str)


def discriminant(
    frame: pd.DataFrame,
    response: str,
    covariates: list[str],
    *,
    method: str = LINEAR,
    priors: str = PROPORTIONAL,
) -> DiscriminantResult:
    """Fit on the rows with a response and every covariate; classify every row with the covariates."""
    if method not in (LINEAR, QUADRATIC):
        raise ValueError(f"Unknown method {method!r}.")
    x_all = frame[covariates].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    complete = np.all(np.isfinite(x_all), axis=1)
    y_all = frame[response]
    fitted = complete & y_all.notna().to_numpy()
    levels = _ordered_levels(y_all[fitted])
    if len(levels) < 2:
        raise ValueError("The response needs at least two groups among the rows with every covariate.")
    codes = pd.Categorical(y_all[fitted], categories=levels).codes
    x = x_all[fitted]
    n, p = x.shape
    k = len(levels)
    counts = np.bincount(codes, minlength=k)
    if np.any(counts < 2):
        small = [str(level) for level, count in zip(levels, counts) if count < 2]
        raise ValueError(f"Every group needs at least two rows; {', '.join(small)} has fewer.")
    if n - k < p:
        raise ValueError(f"{p} covariates need more than {p + k - 1} rows; there are {n}.")

    means = np.vstack([x[codes == level].mean(axis=0) for level in range(k)])
    centred = x - means[codes]
    within = centred.T @ centred  # pooled within-group SSCP
    pooled = within / (n - k)
    prior = counts / n if priors == PROPORTIONAL else np.full(k, 1.0 / k)
    log_prior = np.log(prior)

    def scores_of(points: np.ndarray) -> np.ndarray:
        if method == LINEAR:
            weights = linalg.solve(pooled, means.T, assume_a="pos")  # p x k
            constant = -0.5 * np.sum(means.T * weights, axis=0) + log_prior
            return points @ weights + constant
        out = np.empty((points.shape[0], k))
        for level in range(k):
            members = x[codes == level]
            own = np.cov(members, rowvar=False).reshape(p, p)
            try:
                factor = linalg.cholesky(own, lower=True)
            except linalg.LinAlgError as error:
                raise ValueError(
                    f"Group {levels[level]} has too few rows, or collinear covariates, for a covariance of its "
                    "own: use Linear."
                ) from error
            z = linalg.solve_triangular(factor, (points - means[level]).T, lower=True)
            out[:, level] = -0.5 * np.sum(z * z, axis=0) - np.sum(np.log(np.diag(factor))) + log_prior[level]
        return out

    # Classify every row that has its covariates, with or without a response.
    discriminant_scores = scores_of(x_all[complete])
    shifted = np.exp(discriminant_scores - discriminant_scores.max(axis=1, keepdims=True))
    posterior = shifted / shifted.sum(axis=1, keepdims=True)
    chosen = np.asarray(levels, dtype=object)[np.argmax(posterior, axis=1)]
    predicted = pd.Series(np.nan, index=frame.index, dtype=object)
    predicted[complete] = chosen
    probabilities = pd.DataFrame(np.nan, index=frame.index, columns=[f"Prob({level})" for level in levels])
    probabilities.loc[complete, :] = posterior

    # The canonical analysis: B v = lambda W v.
    grand = x.mean(axis=0)
    between = ((means - grand).T * counts) @ (means - grand)
    eigenvalues, vectors = linalg.eigh(between, within)
    order = np.argsort(eigenvalues)[::-1][: min(p, k - 1)]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    # Scaled so the scores have unit variance within the groups.
    vectors = vectors[:, order] * np.sqrt(n - k)
    for column in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, column])), column] < 0:
            vectors[:, column] *= -1.0
    names = [f"Canonical{number + 1}" for number in range(len(order))]
    share = eigenvalues / eigenvalues.sum() * 100.0 if eigenvalues.sum() > 0 else np.zeros_like(eigenvalues)
    canonical = pd.DataFrame({
        "Canonical": names,
        "Eigenvalue": eigenvalues,
        "Percent": share,
        "Cum Percent": np.cumsum(share),
        "Canonical Corr": np.sqrt(eigenvalues / (1.0 + eigenvalues)),
    })
    standardized = vectors * np.sqrt(np.diag(pooled))[:, None]
    coefficients = pd.DataFrame(standardized, columns=names)
    coefficients.insert(0, "Covariate", covariates)
    scores = pd.DataFrame(np.nan, index=frame.index, columns=names)
    scores.loc[complete, :] = (x_all[complete] - grand) @ vectors

    # Wilks' lambda, with Rao's F.
    wilks = float(np.prod(1.0 / (1.0 + eigenvalues)))
    df1 = p * (k - 1)
    denominator = p * p + (k - 1) ** 2 - 5
    s = np.sqrt((p * p * (k - 1) ** 2 - 4) / denominator) if denominator > 0 else 1.0
    m = n - 1 - (p + k) / 2.0
    df2 = m * s - df1 / 2.0 + 1.0
    root = wilks ** (1.0 / s)
    f_value = (1.0 - root) / root * df2 / df1 if root > 0 else np.inf
    p_value = float(stats.f.sf(f_value, df1, df2)) if df2 > 0 else np.nan

    actual = pd.Categorical(y_all[fitted], categories=levels)
    guessed = pd.Categorical(chosen[fitted[complete]], categories=levels)
    confusion = pd.crosstab(actual, guessed, dropna=False)
    confusion.index.name, confusion.columns.name = "Actual", "Predicted"
    wrong = int(np.sum(np.asarray(actual) != np.asarray(guessed)))

    class_means = pd.DataFrame(means, columns=covariates)
    class_means.insert(0, "Count", counts)
    class_means.insert(0, "Level", levels)
    summary = {
        "Rows": n,
        "Misclassified": wrong,
        "Misclassification rate": wrong / n,
        "Wilks' lambda": wilks,
        "Approx. F": float(f_value),
        "DF num": float(df1),
        "DF den": float(df2),
        "Prob > F": p_value,
    }
    notes = []
    unclassified = int(np.count_nonzero(~complete))
    if unclassified:
        notes.append(f"{unclassified} row(s) with a missing covariate are neither fitted nor classified.")
    return DiscriminantResult(
        response=response, covariates=list(covariates), levels=levels, method=method, summary=summary,
        canonical=canonical, coefficients=coefficients, means=class_means, confusion=confusion,
        predicted=predicted, probabilities=probabilities, scores=scores, notes=notes,
    )
