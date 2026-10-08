"""Least-squares models of a designed experiment, the way JMP's Fit Model reports them.

No Qt here: the Fit Model dialog runs :func:`fit_models` on a worker thread.

A model is a list of *terms* over the factors: ``("A",)`` a main effect,
``("A", "B")`` an interaction, ``("A", "A")`` a quadratic. Continuous factors
are coded to -1..+1 between their low and high (the design's, or else the
data's own range), so every estimate is half the effect over that range and
estimates of different factors compare directly - JMP's scaled estimates.
Nominal factors enter with sum-to-zero contrasts, one estimate per level but
the last, which is minus the sum of the others.

Each response gets, like JMP's report:

- the parameter estimates, with standard error, t ratio and p-value;
- the effect tests (type III, one F test per term, all its levels at once);
- the analysis of variance of the whole model, and the summary of fit
  (RSquare, RSquare Adj, root mean square error, mean, observations);
- the lack-of-fit test, when replicated runs give a pure error to test
  against;
- the predicted values and residuals, raw and studentized.

``reduce`` drops terms backwards, the least significant first, until every
term left has p <= alpha - never a term a higher-order one still contains
(A stays while A*B or A*A is in the model).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.analysis import Stopped

CONTINUOUS = "continuous"
NOMINAL = "nominal"

Term = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Factor:
    """A factor: its column, its modelling type, and (continuous) the range coded to -1..+1."""

    name: str
    kind: str = CONTINUOUS
    low: float | None = None
    high: float | None = None


@dataclass(slots=True)
class ModelSpec:
    factors: list[Factor]
    responses: list[str]
    terms: list[Term]
    reduce: bool = False
    alpha: float = 0.05


@dataclass(slots=True)
class ResponseFit:
    """One response's model and report."""

    response: str
    terms: list[Term]
    removed: list[Term]
    estimates: pd.DataFrame
    effect_tests: pd.DataFrame
    anova: pd.DataFrame
    summary: dict[str, float]
    lack_of_fit: pd.DataFrame | None
    rows: pd.Index
    predicted: pd.Series
    residuals: pd.Series
    studentized: pd.Series
    coding: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: The prediction profile: for each continuous factor in the model, the
    #: predicted response from -1 to +1 coded with the others at their centre
    #: (nominal ones at their most frequent level), and its 95% confidence
    #: band. Columns Factor, Coded, Value, Predicted, Lower, Upper.
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: A note for the report when the model cannot test anything (no error
    #: degrees of freedom, say).
    note: str = ""


# ----------------------------------------------------------------------
# Terms
# ----------------------------------------------------------------------


def term_label(term: Term) -> str:
    """How a term reads in a report: A, A*B, A*A."""
    return "*".join(term)


def main_effects(factors: Sequence[Factor]) -> list[Term]:
    return [(f.name,) for f in factors]


def factorial(factors: Sequence[Factor], degree: int = 2) -> list[Term]:
    """Main effects and every interaction up to *degree* factors."""
    from itertools import combinations

    names = [f.name for f in factors]
    terms: list[Term] = []
    for size in range(1, max(1, degree) + 1):
        terms.extend(combinations(names, size))
    return terms


def response_surface(factors: Sequence[Factor]) -> list[Term]:
    """Main effects, two-factor interactions, and the square of each continuous factor."""
    terms = factorial(factors, 2)
    terms.extend((f.name, f.name) for f in factors if f.kind == CONTINUOUS)
    return terms


def _contained(small: Term, big: Term) -> bool:
    """Whether *small* is a lower-order part of *big* (A in A*B, A in A*A)."""
    if len(small) >= len(big):
        return False
    remaining = list(big)
    for name in small:
        if name not in remaining:
            return False
        remaining.remove(name)
    return True


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def coding(factor: Factor, values: pd.Series) -> tuple[float, float]:
    """(centre, half range) of a continuous factor: from its low/high, else from the data."""
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    low = factor.low if factor.low is not None else (float(numbers.min()) if len(numbers) else 0.0)
    high = factor.high if factor.high is not None else (float(numbers.max()) if len(numbers) else 1.0)
    half = (high - low) / 2.0
    return (low + high) / 2.0, half if half else 1.0


def fit_models(
    frame: pd.DataFrame, spec: ModelSpec, *, should_stop: Callable[[], bool] | None = None
) -> list[ResponseFit]:
    """Fit *spec*'s model to each of its responses in *frame*."""
    if not spec.factors:
        raise ValueError("choose at least one factor")
    if not spec.responses:
        raise ValueError("choose at least one response")
    if not spec.terms:
        raise ValueError("the model has no terms")
    fits = []
    for response in spec.responses:
        if should_stop is not None and should_stop():
            raise Stopped()
        fits.append(_fit_one(frame, spec, response))
    return fits


def _fit_one(frame: pd.DataFrame, spec: ModelSpec, response: str) -> ResponseFit:
    import statsmodels.formula.api as smf

    factors = {f.name: f for f in spec.factors}
    safe = {f.name: f"x{index}" for index, f in enumerate(spec.factors)}
    data = pd.DataFrame(index=frame.index)
    data["y"] = pd.to_numeric(frame[response], errors="coerce")
    codes: dict[str, tuple[float, float]] = {}
    for f in spec.factors:
        if f.kind == CONTINUOUS:
            centre, half = coding(f, frame[f.name])
            codes[f.name] = (centre, half)
            data[safe[f.name]] = (pd.to_numeric(frame[f.name], errors="coerce") - centre) / half
        else:
            data[safe[f.name]] = frame[f.name].astype("string")
    data = data.dropna()
    if len(data) < 2:
        raise ValueError(f"{response}: fewer than two complete rows")

    def piece(term: Term) -> str:
        parts = []
        if len(term) == 2 and term[0] == term[1] and factors[term[0]].kind == CONTINUOUS:
            return f"I({safe[term[0]]}**2)"
        for name in term:
            parts.append(safe[name] if factors[name].kind == CONTINUOUS else f"C({safe[name]}, Sum)")
        return ":".join(parts)

    terms = [tuple(t) for t in spec.terms]
    removed: list[Term] = []
    while True:
        formula = "y ~ " + " + ".join(piece(t) for t in terms)
        model = smf.ols(formula, data=data).fit()
        tests = _effect_tests(model, terms, piece)
        if not spec.reduce or model.df_resid <= 0:
            break
        candidates = [
            (p, t) for t, p in zip(terms, tests["Prob > F"])
            if np.isfinite(p) and p > spec.alpha and not any(_contained(t, other) for other in terms if other != t)
        ]
        if not candidates or len(terms) == 1:
            break
        worst = max(candidates)[1]
        terms.remove(worst)
        removed.append(worst)

    estimates = _estimates(model, terms, piece)
    anova, summary = _whole_model(model, data["y"])
    lack = _lack_of_fit(model, data, [safe[f.name] for f in spec.factors])
    try:
        studentized = pd.Series(model.get_influence().resid_studentized_internal, index=data.index)
    except Exception:  # noqa: BLE001 - a saturated model has none
        studentized = pd.Series(np.nan, index=data.index)
    profile = _profile(model, data, spec.factors, safe, codes, terms)
    note = ""
    if model.df_resid <= 0:
        note = "No degrees of freedom are left for error: the model has as many terms as runs, so nothing can be tested. Remove terms, or add runs."
    return ResponseFit(
        response=response,
        terms=terms,
        removed=removed,
        estimates=estimates,
        effect_tests=tests,
        anova=anova,
        summary=summary,
        lack_of_fit=lack,
        rows=data.index,
        predicted=pd.Series(model.fittedvalues, index=data.index),
        residuals=pd.Series(model.resid, index=data.index),
        studentized=studentized,
        coding=codes,
        profile=profile,
        note=note,
    )


#: Points along each factor in the prediction profile.
PROFILE_POINTS = 21


def _profile(model, data: pd.DataFrame, factors: Sequence[Factor], safe: dict[str, str],
             codes: dict[str, tuple[float, float]], terms: list[Term]) -> pd.DataFrame:
    """The predicted response along each continuous factor still in the model, the others held."""
    used = {name for term in terms for name in term}
    held = {}
    for f in factors:
        column = safe[f.name]
        held[column] = 0.0 if f.kind == CONTINUOUS else data[column].mode().iloc[0]
    grid = np.linspace(-1.0, 1.0, PROFILE_POINTS)
    pieces = []
    for f in factors:
        if f.kind != CONTINUOUS or f.name not in used:
            continue
        new = pd.DataFrame({column: [value] * len(grid) for column, value in held.items()})
        new[safe[f.name]] = grid
        with np.errstate(all="ignore"):
            frame = model.get_prediction(new).summary_frame(alpha=0.05)
        centre, half = codes[f.name]
        pieces.append(pd.DataFrame({
            "Factor": f.name,
            "Coded": grid,
            "Value": centre + grid * half,
            "Predicted": frame["mean"].to_numpy(),
            "Lower": frame["mean_ci_lower"].to_numpy(),
            "Upper": frame["mean_ci_upper"].to_numpy(),
        }))
    columns = ["Factor", "Coded", "Value", "Predicted", "Lower", "Upper"]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def _patsy_name(piece_text: str) -> str:
    """The name patsy gives the term *piece_text* (it rewrites ``I(x**2)`` as ``I(x ** 2)``)."""
    import patsy

    return patsy.ModelDesc.from_formula("y ~ " + piece_text).rhs_termlist[-1].name()


def _effect_tests(model, terms: list[Term], piece) -> pd.DataFrame:
    """One F test per term, all its estimates at once (type III, sum contrasts)."""
    from statsmodels.stats.anova import anova_lm

    rows = []
    if model.df_resid > 0:
        table = anova_lm(model, typ=3)
    else:
        table = None
    for term in terms:
        name = _patsy_name(piece(term))
        if table is not None and name in table.index:
            line = table.loc[name]
            rows.append((term_label(term), int(line["df"]), float(line["sum_sq"]), float(line["F"]), float(line["PR(>F)"])))
        else:
            width = model.model.data.design_info.term_name_slices.get(name)
            df = (width.stop - width.start) if width is not None else 1
            rows.append((term_label(term), df, math.nan, math.nan, math.nan))
    tests = pd.DataFrame(rows, columns=["Term", "DF", "Sum of Squares", "F Ratio", "Prob > F"])
    tests["LogWorth"] = -np.log10(tests["Prob > F"].clip(lower=1e-300))
    return tests


def _estimates(model, terms: list[Term], piece) -> pd.DataFrame:
    """The parameter estimates, named by term (and level, for a nominal factor's)."""
    slices = model.model.data.design_info.term_name_slices
    names = list(model.params.index)
    labels = {"Intercept": "Intercept"}
    for term in terms:
        span = slices.get(_patsy_name(piece(term)))
        if span is None:
            continue
        for column in names[span]:
            label = term_label(term)
            # A nominal factor's level, as patsy names it: C(x2, Sum)[S.Blue]
            levels = [part.split("[S.", 1)[1].rstrip("]") for part in column.split(":") if "[S." in part]
            if levels:
                label += "[" + ",".join(levels) + "]"
            labels[column] = label
    with np.errstate(all="ignore"):
        table = pd.DataFrame(
            {
                "Term": [labels.get(n, n) for n in names],
                "Estimate": model.params.to_numpy(),
                "Std Error": model.bse.to_numpy() if model.df_resid > 0 else np.nan,
                "t Ratio": model.tvalues.to_numpy() if model.df_resid > 0 else np.nan,
                "Prob > |t|": model.pvalues.to_numpy() if model.df_resid > 0 else np.nan,
            }
        )
    return table


def _whole_model(model, y: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    total = float(((y - y.mean()) ** 2).sum())
    error = float(model.ssr)
    df_model, df_error = int(model.df_model), int(model.df_resid)
    model_ss = total - error
    rows = [("Model", df_model, model_ss), ("Error", df_error, error), ("C. Total", df_model + df_error, total)]
    anova = pd.DataFrame(rows, columns=["Source", "DF", "Sum of Squares"])
    with np.errstate(all="ignore"):
        anova["Mean Square"] = anova["Sum of Squares"] / anova["DF"].replace(0, np.nan)
        f = anova.loc[0, "Mean Square"] / anova.loc[1, "Mean Square"] if df_error > 0 else math.nan
    anova["F Ratio"] = [f, math.nan, math.nan]
    anova["Prob > F"] = [float(model.f_pvalue) if df_error > 0 else math.nan, math.nan, math.nan]
    anova.loc[2, "Mean Square"] = math.nan
    summary = {
        "RSquare": float(model.rsquared),
        "RSquare Adj": float(model.rsquared_adj) if df_error > 0 else math.nan,
        "Root Mean Square Error": math.sqrt(error / df_error) if df_error > 0 else math.nan,
        "Mean of Response": float(y.mean()),
        "Observations": float(len(y)),
    }
    return anova, summary


def _lack_of_fit(model, data: pd.DataFrame, factor_columns: list[str]) -> pd.DataFrame | None:
    """Lack of fit against the pure error of replicated runs; None without replicates."""
    groups = data.groupby(factor_columns, dropna=False, sort=False)["y"]
    pure = float(((data["y"] - groups.transform("mean")) ** 2).sum())
    df_pure = int(len(data) - groups.ngroups)
    df_lack = int(model.df_resid) - df_pure
    if df_pure <= 0 or df_lack <= 0:
        return None
    lack = float(model.ssr) - pure
    f = (lack / df_lack) / (pure / df_pure) if pure > 0 else math.inf
    from scipy import stats

    p = float(stats.f.sf(f, df_lack, df_pure)) if math.isfinite(f) else 0.0
    return pd.DataFrame(
        [
            ("Lack Of Fit", df_lack, lack, lack / df_lack, f, p),
            ("Pure Error", df_pure, pure, pure / df_pure, math.nan, math.nan),
            ("Total Error", df_lack + df_pure, float(model.ssr), float(model.ssr) / (df_lack + df_pure), math.nan, math.nan),
        ],
        columns=["Source", "DF", "Sum of Squares", "Mean Square", "F Ratio", "Prob > F"],
    )
