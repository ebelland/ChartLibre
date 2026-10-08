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
    #: How the responses are modelled - JMP's personalities: "standard"
    #: (least squares), "stepwise", "glm", "nominal" (logistic), "ordinal".
    personality: str = "standard"
    #: Stepwise: "backward" (drop) or "forward" (add) - see :data:`STEPWISE`.
    direction: str = "backward"
    #: Generalized linear model: the response's distribution - see :data:`FAMILIES`.
    family: str = "normal"


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
    #: The personality that fitted this, and how its whole-model table and
    #: its tests' p-values are titled (an ANOVA and Prob > F for least
    #: squares, a likelihood-ratio test and Prob > ChiSq for the others).
    personality: str = "standard"
    whole_title: str = "Analysis of variance"
    p_column: str = "Prob > F"
    #: A categorical response's levels, in the order modelled; the first is
    #: the one whose probability the predictions give.
    levels: list[str] = field(default_factory=list)


#: The personalities: key, name as JMP names it.
PERSONALITIES: tuple[tuple[str, str], ...] = (
    ("standard", "Standard Least Squares"),
    ("stepwise", "Stepwise"),
    ("glm", "Generalized Linear Model"),
    ("nominal", "Nominal Logistic"),
    ("ordinal", "Ordinal Logistic"),
)
#: Generalized linear model families, each with its canonical link (log for Gamma).
FAMILIES: tuple[tuple[str, str], ...] = (
    ("normal", "Normal"),
    ("binomial", "Binomial"),
    ("poisson", "Poisson"),
    ("gamma", "Gamma"),
)
STEPWISE: tuple[tuple[str, str], ...] = (("backward", "Backward"), ("forward", "Forward"))
#: The personalities whose response is a category rather than a number.
CATEGORICAL_RESPONSE = frozenset({"nominal", "ordinal"})


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
        if spec.personality == "glm":
            fits.append(_fit_glm(frame, spec, response))
        elif spec.personality in CATEGORICAL_RESPONSE:
            fits.append(_fit_logistic(frame, spec, response))
        else:
            fits.append(_fit_one(frame, spec, response))
    return fits


@dataclass(slots=True)
class _Prepared:
    """A response's rows, coded, and how each term is spelled in a formula."""

    data: pd.DataFrame
    safe: dict[str, str]
    codes: dict[str, tuple[float, float]]
    factors: dict[str, Factor]

    def piece(self, term: Term) -> str:
        if len(term) == 2 and term[0] == term[1] and self.factors[term[0]].kind == CONTINUOUS:
            return f"I({self.safe[term[0]]}**2)"
        return ":".join(
            self.safe[name] if self.factors[name].kind == CONTINUOUS else f"C({self.safe[name]}, Sum)"
            for name in term
        )

    def formula(self, terms: Sequence[Term]) -> str:
        return "y ~ " + (" + ".join(self.piece(t) for t in terms) if terms else "1")


def _prepare(frame: pd.DataFrame, spec: ModelSpec, response: str, *, categorical: bool) -> _Prepared:
    safe = {f.name: f"x{index}" for index, f in enumerate(spec.factors)}
    data = pd.DataFrame(index=frame.index)
    if categorical:
        data["y"] = frame[response].astype("string")
    else:
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
    return _Prepared(data, safe, codes, {f.name: f for f in spec.factors})


def _fit_one(frame: pd.DataFrame, spec: ModelSpec, response: str) -> ResponseFit:
    """Least squares - standard, or stepwise backward or forward."""
    import statsmodels.formula.api as smf

    prep = _prepare(frame, spec, response, categorical=False)
    data, piece = prep.data, prep.piece
    stepwise = spec.personality == "stepwise" or spec.reduce
    terms = [tuple(t) for t in spec.terms]
    removed: list[Term] = []

    if stepwise and spec.direction == "forward":
        # Forward: from the mean alone, add the most significant term whose
        # lower-order parts are already in, while it is significant.
        chosen: list[Term] = []
        while True:
            candidates = [
                t for t in terms if t not in chosen
                and all(other in chosen for other in terms if _contained(other, t))
            ]
            best: tuple[float, Term] | None = None
            for term in candidates:
                trial = chosen + [term]
                model = smf.ols(prep.formula(trial), data=data).fit()
                if model.df_resid <= 0:
                    continue
                p = float(_effect_tests(model, trial, piece).iloc[-1]["Prob > F"])
                if np.isfinite(p) and (best is None or p < best[0]):
                    best = (p, term)
            if best is None or best[0] > spec.alpha:
                break
            chosen.append(best[1])
        removed = [t for t in terms if t not in chosen]
        terms = chosen

    while True:
        model = smf.ols(prep.formula(terms), data=data).fit()
        tests = _effect_tests(model, terms, piece)
        if not stepwise or spec.direction == "forward" or model.df_resid <= 0:
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
    lack = _lack_of_fit(model, data, [prep.safe[f.name] for f in spec.factors])
    try:
        studentized = pd.Series(model.get_influence().resid_studentized_internal, index=data.index)
    except Exception:  # noqa: BLE001 - a saturated model has none
        studentized = pd.Series(np.nan, index=data.index)
    profile = _profile(model, data, spec.factors, prep.safe, prep.codes, terms)
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
        coding=prep.codes,
        profile=profile,
        note=note,
        personality=spec.personality,
    )


def _chi_square_table(rows: list[tuple[str, int, float, float]], statistic: str) -> pd.DataFrame:
    tests = pd.DataFrame(rows, columns=["Term", "DF", statistic, "Prob > ChiSq"])
    tests["LogWorth"] = -np.log10(tests["Prob > ChiSq"].clip(lower=1e-300))
    return tests


def _whole_model_test(llf: float, llnull: float, df: int) -> pd.DataFrame:
    """JMP's Whole Model Test: the likelihood ratio of the model against the intercept alone."""
    from scipy import stats

    chi2 = max(2.0 * (llf - llnull), 0.0)
    p = float(stats.chi2.sf(chi2, df)) if df > 0 else math.nan
    return pd.DataFrame(
        [("Difference", df, llf - llnull, chi2, p), ("Full", math.nan, -llf, math.nan, math.nan),
         ("Reduced", math.nan, -llnull, math.nan, math.nan)],
        columns=["Model", "DF", "-LogLikelihood", "Chi-Square", "Prob > ChiSq"],
    )


def _fit_glm(frame: pd.DataFrame, spec: ModelSpec, response: str) -> ResponseFit:
    """A generalized linear model: the response's family with its canonical link (log for Gamma)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    families = {
        "normal": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(link=sm.families.links.Log()),
    }
    prep = _prepare(frame, spec, response, categorical=False)
    terms = [tuple(t) for t in spec.terms]
    model = smf.glm(prep.formula(terms), data=prep.data, family=families.get(spec.family, families["normal"])).fit()
    wald = model.wald_test_terms(skip_single=False, scalar=True).table
    rows = []
    for term in terms:
        name = _patsy_name(prep.piece(term))
        line = wald.loc[name] if name in wald.index else None
        rows.append((term_label(term), int(line["df_constraint"]) if line is not None else 1,
                     float(line["statistic"]) if line is not None else math.nan,
                     float(line["pvalue"]) if line is not None else math.nan))
    estimates = _estimates(model, terms, prep.piece).rename(columns={"t Ratio": "z Ratio", "Prob > |t|": "Prob > |z|"})
    null = smf.glm("y ~ 1", data=prep.data, family=model.family).fit()
    summary = {
        "Observations": float(model.nobs),
        "Deviance": float(model.deviance),
        "Pearson Chi-Square": float(model.pearson_chi2),
        "AIC": float(model.aic),
        "BIC": float(model.bic_llf),
    }
    return ResponseFit(
        response=response, terms=terms, removed=[], estimates=estimates,
        effect_tests=_chi_square_table(rows, "Wald Chi-Square"),
        anova=_whole_model_test(float(model.llf), float(null.llf), int(model.df_model)),
        summary=summary, lack_of_fit=None, rows=prep.data.index,
        predicted=pd.Series(model.fittedvalues, index=prep.data.index),
        residuals=pd.Series(model.resid_deviance, index=prep.data.index),
        studentized=pd.Series(model.resid_pearson, index=prep.data.index),
        coding=prep.codes, profile=_profile(model, prep.data, spec.factors, prep.safe, prep.codes, terms),
        personality="glm", whole_title="Whole model test", p_column="Prob > ChiSq",
    )


def _levels(values: pd.Series) -> list[str]:
    """A categorical response's levels: numerically when they are numbers, else alphabetically."""
    distinct = list(dict.fromkeys(values.astype("string")))
    numbers = pd.to_numeric(pd.Series(distinct), errors="coerce")
    if numbers.notna().all():
        return [distinct[i] for i in np.argsort(numbers.to_numpy(), kind="stable")]
    return sorted(distinct)


def _fit_logistic(frame: pd.DataFrame, spec: ModelSpec, response: str) -> ResponseFit:
    """Nominal (binary or multinomial) or ordinal logistic regression, after JMP.

    Each term's test is a likelihood ratio: the model refitted without it.
    Predictions give the probability of the first level.
    """
    import patsy
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    prep = _prepare(frame, spec, response, categorical=True)
    levels = _levels(prep.data["y"])
    if len(levels) < 2:
        raise ValueError(f"{response}: a logistic model needs at least two levels")
    terms = [tuple(t) for t in spec.terms]
    ordinal = spec.personality == "ordinal"
    codes = prep.data["y"].map({level: i for i, level in enumerate(levels)}).astype(int)
    design = patsy.dmatrix(prep.formula(terms).split("~", 1)[1], prep.data, return_type="dataframe")
    slices = design.design_info.term_name_slices

    def fit(columns: list[str]):
        exog = design[columns]
        if ordinal:
            return OrderedModel(codes, exog.drop(columns=["Intercept"], errors="ignore"), distr="logit").fit(
                method="bfgs", disp=0, maxiter=500)
        if len(levels) == 2:
            return sm.Logit((codes == 0).astype(int), exog).fit(disp=0, maxiter=200)
        return sm.MNLogit(codes, exog).fit(disp=0, maxiter=200)

    columns = list(design.columns)
    full = fit(columns)
    counts = codes.value_counts().to_numpy(dtype=float)
    llnull = float((counts * np.log(counts / counts.sum())).sum())
    width = 1 if ordinal or len(levels) == 2 else len(levels) - 1
    rows = []
    for term in terms:
        span = slices.get(_patsy_name(prep.piece(term)))
        if span is None:
            continue
        dropped = columns[span]
        kept = [c for c in columns if c not in dropped]
        if ordinal and not [c for c in kept if c != "Intercept"]:
            reduced_llf = llnull
        else:
            reduced_llf = float(fit(kept).llf)
        chi2 = max(2.0 * (float(full.llf) - reduced_llf), 0.0)
        df = len(dropped) * width
        rows.append((term_label(term), df, chi2, float(stats.chi2.sf(chi2, df))))

    labels = _labels(columns, terms, prep.piece, slices)
    params, errors, z, p = full.params, full.bse, full.tvalues, full.pvalues
    if isinstance(params, pd.DataFrame):  # multinomial: one column per level after the first
        records = []
        for j, column in enumerate(params.columns):
            level = levels[j + 1]
            for name in params.index:
                records.append((f"{labels.get(name, name)} [{level}]", params.loc[name, column],
                                errors.loc[name, column], z.loc[name, column], p.loc[name, column]))
        estimates = pd.DataFrame(records, columns=["Term", "Estimate", "Std Error", "z Ratio", "Prob > |z|"])
    else:
        names = list(params.index)
        estimates = pd.DataFrame({
            "Term": [labels.get(n, n) for n in names],
            "Estimate": params.to_numpy(), "Std Error": np.asarray(errors), "z Ratio": np.asarray(z),
            "Prob > |z|": np.asarray(p),
        })

    probabilities = np.asarray(full.predict(design[columns].drop(columns=["Intercept"], errors="ignore")
                                            if ordinal else design[columns]))
    first = probabilities if probabilities.ndim == 1 else probabilities[:, 0]
    df_model = len(full.params) - (len(levels) - 1 if ordinal else width)
    llf = float(full.llf)
    summary = {
        "Observations": float(len(codes)),
        "-LogLikelihood": -llf,
        "RSquare (U)": 1 - llf / llnull if llnull else math.nan,
        "AICc": float(-2 * llf + 2 * len(np.ravel(full.params)) * len(codes) / max(len(codes) - len(np.ravel(full.params)) - 1, 1)),
    }
    profile = pd.DataFrame(columns=["Factor", "Coded", "Value", "Predicted", "Lower", "Upper"])
    if not ordinal and len(levels) == 2:
        profile = _probability_profile(full, design, prep, spec.factors, terms)
    return ResponseFit(
        response=response, terms=terms, removed=[], estimates=estimates,
        effect_tests=_chi_square_table(rows, "L-R Chi-Square"),
        anova=_whole_model_test(llf, llnull, int(df_model)),
        summary=summary, lack_of_fit=None, rows=prep.data.index,
        predicted=pd.Series(first, index=prep.data.index),
        residuals=pd.Series((codes == 0).astype(float).to_numpy() - first, index=prep.data.index),
        studentized=pd.Series(np.nan, index=prep.data.index),
        coding=prep.codes, profile=profile,
        personality=spec.personality, whole_title="Whole model test", p_column="Prob > ChiSq", levels=levels,
    )


def _labels(columns: list[str], terms: list[Term], piece, slices) -> dict[str, str]:
    """Design column names (patsy's) to the report's: A, A*B, Cat[level]."""
    labels = {"Intercept": "Intercept"}
    for term in terms:
        span = slices.get(_patsy_name(piece(term)))
        if span is None:
            continue
        for column in columns[span]:
            label = term_label(term)
            levels = [part.split("[S.", 1)[1].rstrip("]") for part in column.split(":") if "[S." in part]
            labels[column] = label + ("[" + ",".join(levels) + "]" if levels else "")
    return labels


def _probability_profile(model, design: pd.DataFrame, prep: _Prepared, factors, terms) -> pd.DataFrame:
    """A binary logistic model's probability of the first level along each continuous factor."""
    import patsy

    used = {name for term in terms for name in term}
    held = {prep.safe[f.name]: (0.0 if f.kind == CONTINUOUS else prep.data[prep.safe[f.name]].mode().iloc[0])
            for f in factors}
    grid = np.linspace(-1.0, 1.0, PROFILE_POINTS)
    pieces = []
    for f in factors:
        if f.kind != CONTINUOUS or f.name not in used:
            continue
        new = pd.DataFrame({column: [value] * len(grid) for column, value in held.items()})
        new[prep.safe[f.name]] = grid
        exog = patsy.build_design_matrices([design.design_info], new, return_type="dataframe")[0]
        p = np.asarray(model.predict(exog))
        centre, half = prep.codes[f.name]
        pieces.append(pd.DataFrame({"Factor": f.name, "Coded": grid, "Value": centre + grid * half,
                                    "Predicted": p, "Lower": np.nan, "Upper": np.nan}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["Factor", "Coded", "Value", "Predicted", "Lower", "Upper"])


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
