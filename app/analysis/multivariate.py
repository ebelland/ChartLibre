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


# ----------------------------------------------------------------------
# Factor Analysis (AN-04)
# ----------------------------------------------------------------------
ML = "ml"
PRINCIPAL_AXIS = "principal"
FA_METHODS: tuple[tuple[str, str], ...] = ((ML, "Maximum Likelihood"), (PRINCIPAL_AXIS, "Principal Axis"))
VARIMAX, QUARTIMAX, EQUAMAX, PROMAX, NO_ROTATION = "varimax", "quartimax", "equamax", "promax", "none"
ROTATIONS: tuple[tuple[str, str], ...] = (
    (VARIMAX, "Varimax"), (QUARTIMAX, "Quartimax"), (EQUAMAX, "Equamax"), (PROMAX, "Promax (oblique)"),
    (NO_ROTATION, "None"),
)


@dataclass(slots=True)
class FactorResult:
    columns: list[str]
    method: str
    rotation: str
    factors: int
    eigenvalues: pd.DataFrame        # of the correlation matrix, for the scree plot
    loadings: pd.DataFrame           # Variable, Factor1.. (rotated; the pattern when oblique)
    communalities: pd.DataFrame      # Variable, Communality, Uniqueness
    variance: pd.DataFrame           # Factor, Variance (SS loadings), Percent, Cum Percent
    factor_correlations: pd.DataFrame  # oblique rotations only; empty otherwise
    scores: pd.DataFrame             # each row's factor scores (regression method)
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def orthomax(loadings: np.ndarray, gamma: float, *, iterations: int = 1000,
             tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """The orthomax rotation of *loadings* (gamma 1 varimax, 0 quartimax, k/2 equamax): (rotated, rotation)."""
    p, k = loadings.shape
    rotation = np.eye(k)
    criterion = 0.0
    for _ in range(iterations):
        rotated = loadings @ rotation
        target = rotated ** 3 - (gamma / p) * rotated @ np.diag(np.sum(rotated ** 2, axis=0))
        u, s, vt = np.linalg.svd(loadings.T @ target)
        rotation = u @ vt
        previous, criterion = criterion, float(np.sum(s))
        if criterion - previous < tolerance * max(criterion, 1.0):
            break
    return loadings @ rotation, rotation


def promax(loadings: np.ndarray, power: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Promax: varimax, then an oblique fit to its loadings raised to *power*: (pattern, factor correlations)."""
    varimax, _rotation = orthomax(loadings, 1.0)
    target = varimax * np.abs(varimax) ** (power - 1)
    transform, *_rest = np.linalg.lstsq(varimax, target, rcond=None)
    transform = transform @ np.diag(np.sqrt(np.diag(np.linalg.inv(transform.T @ transform))))
    inverse = np.linalg.inv(transform)
    return varimax @ transform, inverse @ inverse.T


def _principal_axis(correlation: np.ndarray, factors: int, iterations: int = 200) -> tuple[np.ndarray, bool]:
    """Iterated principal-axis loadings, from the squared multiple correlations; and whether it converged."""
    try:
        communality = 1.0 - 1.0 / np.diag(np.linalg.inv(correlation))
    except np.linalg.LinAlgError:
        communality = np.full(len(correlation), 0.5)
    loadings = np.zeros((len(correlation), factors))
    for _ in range(iterations):
        reduced = correlation.copy()
        np.fill_diagonal(reduced, communality)
        values, vectors = np.linalg.eigh(reduced)
        order = np.argsort(values)[::-1][:factors]
        loadings = vectors[:, order] * np.sqrt(np.clip(values[order], 0.0, None))
        updated = np.clip(np.sum(loadings ** 2, axis=1), 0.0, 0.995)  # a Heywood case stops just short of 1
        if np.max(np.abs(updated - communality)) < 1e-6:
            return loadings, True
        communality = updated
    return loadings, False


def factor_analysis(frame: pd.DataFrame, columns: list[str], *, factors: int = 0, method: str = ML,
                    rotation: str = VARIMAX) -> FactorResult:
    """The common factors of *columns*: loadings, communalities, variance explained, each row's scores.

    On the correlations (the columns standardized), as JMP's Factor Analysis.
    *factors* 0 takes as many as the correlation matrix has eigenvalues above 1.
    """
    if len(columns) < 3:
        raise ValueError("choose three columns or more")
    if method not in dict(FA_METHODS):
        raise ValueError(f"'{method}' is not one of {', '.join(key for key, _label in FA_METHODS)}")
    if rotation not in dict(ROTATIONS):
        raise ValueError(f"'{rotation}' is not one of {', '.join(key for key, _label in ROTATIONS)}")
    data = _numbers(frame, columns).dropna()
    if len(data) <= len(columns):
        raise ValueError(f"{len(data)} complete rows for {len(columns)} columns: more rows than columns are needed")
    spread = data.std(ddof=1)
    constant = [c for c in columns if not spread[c] > 0]
    if constant:
        raise ValueError(f"{', '.join(constant)} does not vary")
    standardized = ((data - data.mean()) / spread).to_numpy(float)
    correlation = np.corrcoef(standardized, rowvar=False)
    values = np.sort(np.linalg.eigvalsh(correlation))[::-1]
    p = len(columns)
    notes: list[str] = []
    count = int(factors) if factors else max(1, int((values > 1.0).sum()))
    if count >= p:
        raise ValueError(f"{count} factors for {p} columns: fewer factors than columns are needed")
    if not factors:
        notes.append(f"{count} factor(s): the eigenvalues of the correlations above 1.")

    if method == ML:
        from sklearn.decomposition import FactorAnalysis

        model = FactorAnalysis(n_components=count, random_state=0).fit(standardized)
        loadings = model.components_.T
    else:
        loadings, converged = _principal_axis(correlation, count)
        if not converged:
            notes.append("Principal Axis did not converge in 200 iterations: the loadings are the last ones.")

    phi = np.eye(count)
    if rotation in (VARIMAX, QUARTIMAX, EQUAMAX) and count > 1:
        gamma = {VARIMAX: 1.0, QUARTIMAX: 0.0, EQUAMAX: count / 2.0}[rotation]
        loadings, _rotation = orthomax(loadings, gamma)
    elif rotation == PROMAX and count > 1:
        loadings, phi = promax(loadings)
    # A fixed sign (the largest loading positive) and the factors by the variance they carry.
    signs = np.sign(loadings[np.argmax(np.abs(loadings), axis=0), np.arange(count)])
    loadings = loadings * np.where(signs == 0, 1.0, signs)
    phi = phi * np.outer(np.where(signs == 0, 1.0, signs), np.where(signs == 0, 1.0, signs))
    variance = np.sum(loadings ** 2, axis=0)
    order = np.argsort(variance)[::-1]
    loadings, variance, phi = loadings[:, order], variance[order], phi[np.ix_(order, order)]

    structure = loadings @ phi
    communality = np.sum(loadings * structure, axis=1)
    names = [f"Factor{k + 1}" for k in range(count)]
    try:
        weights = np.linalg.solve(correlation, structure)  # Thurstone's regression scores
        scores = pd.DataFrame(standardized @ weights, index=data.index, columns=names)
    except np.linalg.LinAlgError:
        scores = pd.DataFrame(index=data.index, columns=names, dtype=float)
        notes.append("The correlation matrix is singular: no factor scores.")
    if np.any(communality > 0.99):
        notes.append("A communality near 1 (a Heywood case): a column is all common factor - fewer factors may fit better.")
    if len(data) < len(frame):
        notes.append(f"{len(frame) - len(data)} row(s) miss a value in some column and have no scores.")
    percent = 100.0 * variance / p
    oblique = rotation == PROMAX and count > 1
    if oblique:
        notes.append("Promax is oblique: the factors correlate, and their variances overlap (no cumulative share).")
    summary: dict[str, Any] = {"Columns": p, "Rows": len(data), "Factors": count,
                               "Common variance explained %": float(np.sum(communality) / p * 100.0)}
    return FactorResult(
        columns=list(columns), method=method, rotation=rotation, factors=count,
        eigenvalues=pd.DataFrame({"Number": np.arange(1, p + 1), "Eigenvalue": values}),
        loadings=pd.DataFrame(loadings, columns=names).assign(Variable=list(columns))[["Variable", *names]],
        communalities=pd.DataFrame({"Variable": list(columns), "Communality": communality,
                                    "Uniqueness": 1.0 - communality}),
        variance=pd.DataFrame({"Factor": names, "Variance": variance, "Percent": percent,
                               "Cum Percent": np.nan if oblique else np.cumsum(percent)}),
        factor_correlations=pd.DataFrame(phi, columns=names).assign(Factor=names)[["Factor", *names]]
        if oblique else pd.DataFrame(),
        scores=scores, summary=summary, notes=notes,
    )
