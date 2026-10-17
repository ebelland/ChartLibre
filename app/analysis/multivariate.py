"""Multivariate and Principal Components: how continuous columns move together (AN-04).

As JMP's Multivariate platform, correlations for every pair of columns: the
matrix, and a row per pair with its confidence interval, p-value and count.
Each pair uses the rows where both columns have a value (pairwise), as JMP
does by default. Pearson's r has a Fisher-z interval; Spearman's rho the same
interval on the ranks, as an approximation; Kendall's tau none.

As JMP's Principal Components: the eigenvalues of the correlation matrix
(or of the covariance matrix), the share of the variance each explains, the
loadings - each column's correlation with a component, for the correlation
matrix - and every row's scores. Rows with a value in every column only.
Each component's sign is set so its largest loading is positive: an
eigenvector's sign is arbitrary, and a report should not flip between runs.

Plain numpy and SciPy; the numbers are checked against scikit-learn's PCA
and SciPy's own tests in dev/tests/test_multivariate.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PEARSON = "pearson"
SPEARMAN = "spearman"
KENDALL = "kendall"
METHODS: tuple[tuple[str, str], ...] = (
    (PEARSON, "Pearson (linear)"), (SPEARMAN, "Spearman's rho (ranks)"), (KENDALL, "Kendall's tau (ranks)"),
)

CORRELATIONS = "correlations"
COVARIANCES = "covariances"
PCA_ON: tuple[tuple[str, str], ...] = (
    (CORRELATIONS, "Correlations (columns standardized)"), (COVARIANCES, "Covariances (columns as they are)"),
)

#: The confidence level of the correlations' intervals.
CONFIDENCE = 0.95


@dataclass(slots=True)
class CorrelationResult:
    columns: list[str]
    method: str
    matrix: pd.DataFrame
    pairs: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PCAResult:
    columns: list[str]
    on: str
    eigenvalues: pd.DataFrame
    loadings: pd.DataFrame
    eigenvectors: pd.DataFrame
    scores: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _numbers(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)


def correlations(frame: pd.DataFrame, columns: list[str], *, method: str = PEARSON) -> CorrelationResult:
    """Every pair of *columns*, correlated on the rows where both have a value."""
    if len(columns) < 2:
        raise ValueError("choose two columns or more")
    if method not in dict(METHODS):
        raise ValueError(f"'{method}' is not one of {', '.join(key for key, _label in METHODS)}")
    data = _numbers(frame, columns)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    z = stats.norm.ppf(0.5 + CONFIDENCE / 2)
    for first, second in combinations(columns, 2):
        both = data[[first, second]].dropna()
        n = len(both)
        r = p = low = high = np.nan
        if n >= 3 and both[first].nunique() > 1 and both[second].nunique() > 1:
            x, y = both[first].to_numpy(), both[second].to_numpy()
            if method == PEARSON:
                r, p = stats.pearsonr(x, y)
            elif method == SPEARMAN:
                r, p = stats.spearmanr(x, y)
            else:
                r, p = stats.kendalltau(x, y)
            r, p = float(r), float(p)
            if method != KENDALL and n > 3 and abs(r) < 1:
                center, half = np.arctanh(r), z / np.sqrt(n - 3)
                low, high = float(np.tanh(center - half)), float(np.tanh(center + half))
        else:
            notes.append(f"{first} and {second}: fewer than three rows with both, or a constant column - no correlation.")
        matrix.loc[first, second] = matrix.loc[second, first] = r
        rows.append({"Variable": first, "by Variable": second, "Correlation": r,
                     f"Lower {CONFIDENCE:.0%}": low, f"Upper {CONFIDENCE:.0%}": high, "Prob>|t|": p, "N": n})
    pairs = pd.DataFrame(rows)
    complete = int(data.dropna().shape[0])
    summary = {"Columns": len(columns), "Rows with every column": complete, "Pairs": len(rows)}
    if complete < len(data):
        notes.append("Some rows miss a value: each pair uses the rows where both its columns have one.")
    return CorrelationResult(columns=list(columns), method=method, matrix=matrix.reset_index(names="Variable"),
                             pairs=pairs, summary=summary, notes=notes)


def principal_components(frame: pd.DataFrame, columns: list[str], *, on: str = CORRELATIONS) -> PCAResult:
    """The principal components of *columns*, on their correlations (standardized) or covariances."""
    if len(columns) < 2:
        raise ValueError("choose two columns or more")
    if on not in dict(PCA_ON):
        raise ValueError(f"'{on}' is not one of {', '.join(key for key, _label in PCA_ON)}")
    data = _numbers(frame, columns).dropna()
    if len(data) < 3:
        raise ValueError("fewer than three rows have a value in every column")
    centered = data - data.mean()
    spread = data.std(ddof=1)
    constant = [c for c in columns if not spread[c] > 0]
    if constant:
        raise ValueError(f"{', '.join(constant)} does not vary: it has no direction to give")
    standardized = centered / spread if on == CORRELATIONS else centered
    matrix = np.cov(standardized.to_numpy(), rowvar=False, ddof=1)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]
    for k in range(vectors.shape[1]):  # a fixed sign: the largest loading positive
        if vectors[np.argmax(np.abs(vectors[:, k])), k] < 0:
            vectors[:, k] = -vectors[:, k]
    names = [f"Prin{k + 1}" for k in range(len(values))]
    total = float(values.sum())
    percent = 100.0 * values / total if total > 0 else np.zeros_like(values)
    eigenvalues = pd.DataFrame({"Number": np.arange(1, len(values) + 1), "Eigenvalue": values,
                                "Percent": percent, "Cum Percent": np.cumsum(percent)})
    loadings = vectors * np.sqrt(values)
    if on == COVARIANCES:  # each column's correlation with a component, whatever the matrix
        loadings = loadings / spread.to_numpy()[:, None]
    scores = pd.DataFrame(standardized.to_numpy() @ vectors, index=data.index, columns=names)
    notes = []
    if len(data) < len(frame):
        notes.append(f"{len(frame) - len(data)} row(s) miss a value in some column and have no scores.")
    kaiser = int((values > 1.0).sum()) if on == CORRELATIONS else None
    summary: dict[str, Any] = {"Columns": len(columns), "Rows": len(data)}
    if kaiser is not None:
        summary["Components with eigenvalue > 1"] = kaiser
    return PCAResult(
        columns=list(columns), on=on, eigenvalues=eigenvalues,
        loadings=pd.DataFrame(loadings, columns=names).assign(Variable=list(columns))[["Variable", *names]],
        eigenvectors=pd.DataFrame(vectors, columns=names).assign(Variable=list(columns))[["Variable", *names]],
        scores=scores, summary=summary, notes=notes,
    )
