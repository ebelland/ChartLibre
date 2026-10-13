"""Partial least squares regression, for spectra and other wide, collinear X (AN-17).

As JMP's PLS platform: Y (one or more) from many X that move together -
hundreds of wavelengths of a spectrum - through a few latent components,
each a weighted sum of the X chosen to predict the Y. How many components
is decided by cross-validation, with Wold's criterion: a component is added
only while it cuts the prediction error sum of squares (PRESS) by at least
5% - past that point a component fits the noise of these rows, not
something the next sample will share.

What it reports, as JMP does:

* the cross-validation table (RMSECV and Q² by number of components);
* VIP, variable importance in projection, per X - above 0.8 matters (Wold);
* the X loadings of each component, the coefficients in the X's own units;
* per row, the scores on the first components and the prediction.

The X are centred; scaled to unit variance only when asked - right for X in
different units, wrong for a spectrum, where it gives the wavelengths that
carry only noise the same weight as the bands (JMP's Standardize X).
scikit-learn's PLSRegression (NIPALS) does the fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: The most components tried.
MAX_COMPONENTS = 15
#: VIP above this marks a variable that matters (Wold's rule of thumb).
VIP_THRESHOLD = 0.8
#: Wold's criterion: another component only while PRESS falls below this share of the previous one.
PRESS_RATIO = 0.95


@dataclass(slots=True)
class PLSResult:
    responses: list[str]
    predictors: list[str]
    components: int
    summary: dict[str, float]
    #: Components, RMSECV (per response, averaged), Q², the chosen one marked.
    cross_validation: pd.DataFrame
    #: Variable, VIP, Order (the X's position, for a spectrum's axis).
    vip: pd.DataFrame
    #: Variable, then a loading per component.
    loadings: pd.DataFrame
    #: Variable, then a coefficient per response, in the X's own units.
    coefficients: pd.DataFrame
    #: Per input row (NaN where an X or a Y is missing): scores T1.., predictions.
    scores: pd.DataFrame
    predicted: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _vip(model) -> np.ndarray:
    """Variable importance in projection: each X's share of the Y variance the components explain."""
    t, w, q = model.x_scores_, model.x_weights_, model.y_loadings_
    p = w.shape[0]
    explained = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)
    weights = (w / np.linalg.norm(w, axis=0)) ** 2
    total = explained.sum()
    return np.sqrt(p * (weights @ explained) / total) if total > 0 else np.zeros(p)


def pls(frame: pd.DataFrame, responses: list[str], predictors: list[str], *, max_components: int = MAX_COMPONENTS,
        folds: int = 7, components: int | None = None, scale: bool = False, seed: int = 0) -> PLSResult:
    """Fit PLS of *responses* on *predictors*; *components* None chooses by cross-validation."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    predictors = [name for name in predictors if name not in responses]
    if not responses or len(predictors) < 2:
        raise ValueError("PLS needs a response and at least two X columns.")
    data = frame[[*responses, *predictors]].apply(pd.to_numeric, errors="coerce")
    complete = data.notna().all(axis=1)
    x_all = data[predictors].to_numpy(float)
    y = data.loc[complete, responses].to_numpy(float)
    x = x_all[complete.to_numpy()]
    n = len(y)
    if n < 6:
        raise ValueError(f"{n} complete rows are too few for PLS.")
    folds = max(2, min(folds, n))
    limit = max(1, min(max_components, len(predictors), n - n // folds - 1))
    splitter = KFold(folds, shuffle=True, random_state=seed)

    # Cross-validation: every row predicted once, by a model that did not see it.
    press = np.zeros((limit, len(responses)))
    for train, test in splitter.split(x):
        for count in range(1, limit + 1):
            model = PLSRegression(n_components=count, scale=scale).fit(x[train], y[train])
            error = (y[test] - model.predict(x[test]).reshape(len(test), -1)) ** 2
            press[count - 1] += error.sum(axis=0)
    rmsecv = np.sqrt(press / n).mean(axis=1)
    total = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    q2 = (1.0 - press / np.where(total > 0, total, np.nan)).mean(axis=1)
    if components is None:
        total_press = press.sum(axis=1)
        components = 1
        while components < limit and total_press[components] < PRESS_RATIO * total_press[components - 1]:
            components += 1
    components = max(1, min(int(components), limit))

    model = PLSRegression(n_components=components, scale=scale).fit(x, y)
    fitted = model.predict(x).reshape(n, -1)
    r2 = 1.0 - np.sum((y - fitted) ** 2, axis=0) / np.where(total > 0, total, np.nan)
    names = [f"Component {k + 1}" for k in range(components)]

    cross_validation = pd.DataFrame({
        "Components": np.arange(1, limit + 1),
        "RMSECV": rmsecv,
        "Q²": q2,
        "Chosen": ["Yes" if k + 1 == components else "" for k in range(limit)],
    })
    vip = pd.DataFrame({"Variable": predictors, "VIP": _vip(model), "Order": np.arange(1, len(predictors) + 1)})
    loadings = pd.DataFrame(model.x_loadings_, columns=names)
    loadings.insert(0, "Variable", predictors)
    coefficients = pd.DataFrame(np.asarray(model.coef_).reshape(len(responses), -1).T, columns=responses)
    coefficients.insert(0, "Variable", predictors)

    index = frame.index
    usable = np.isfinite(x_all).all(axis=1)
    scores = pd.DataFrame(np.nan, index=index, columns=[f"T{k + 1}" for k in range(min(components, 3))])
    predicted = pd.DataFrame(np.nan, index=index, columns=[f"Predicted {name}" for name in responses])
    if usable.any():
        scores.loc[usable, :] = model.transform(x_all[usable])[:, : scores.shape[1]]
        predicted.loc[usable, :] = model.predict(x_all[usable]).reshape(int(usable.sum()), -1)

    summary = {
        "Components": float(components),
        "RSquare Y": float(np.nanmean(r2)),
        "Q² (cross-validated)": float(q2[components - 1]),
        "RMSECV": float(rmsecv[components - 1]),
        "Variables with VIP > 0.8": float(np.count_nonzero(vip["VIP"] > VIP_THRESHOLD)),
        "Observations": float(n),
        "Folds": float(folds),
    }
    notes = []
    if int(np.argmin(rmsecv)) + 1 != components:
        notes.append("Fewer components than the lowest RMSECV: the next one cut the prediction error by less "
                     "than 5%, so the simpler model is kept.")
    left_out = int(len(frame) - n)
    if left_out:
        notes.append(f"{left_out} row(s) with a missing value are left out of the fit.")
    return PLSResult(responses=list(responses), predictors=predictors, components=components, summary=summary,
                     cross_validation=cross_validation, vip=vip, loadings=loadings, coefficients=coefficients,
                     scores=scores, predicted=predicted, notes=notes)
