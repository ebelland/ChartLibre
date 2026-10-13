"""Quick answer: "What has the most impact?" - a response and many factors, ranked in words (Q-02).

For someone who wants to know which of many columns matters for one result,
without choosing a model. Two views, each covering the other's blind spot:

* a random forest says how much each factor matters, whatever the shape of
  its effect: the share of the forest's accuracy lost when the factor's
  values are shuffled (permutation importance), on rows the forest had not
  seen (3-fold cross-validation) - so a factor that only fits the noise
  scores nothing;
* a linear model says whether the effect is more than chance (its p-value,
  from a type II ANOVA) and which way it goes. For a categorical response,
  a test per factor: Kruskal-Wallis for a numeric factor, chi-squared for a
  categorical one.

The answer: the factor with the most impact and its share, the next ones,
how much they explain together, which hardly matter - and that impact is
not cause. Sentences are templates (translated by the dialog) with values,
as in app.analysis.answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.analysis.answers import ALPHA, NO, YES

#: A factor at or above this share of the impact, and not ruled out by its p-value,
#: matters; below it, or with a p-value at or above ALPHA, it has little or no impact.
LITTLE_SHARE = 5.0
FOLDS = 3


@dataclass(slots=True)
class ImpactAnswer:
    question: str
    verdict: str
    sentences: list[tuple[str, dict[str, Any]]]
    #: Factor, Impact % (sums to 100), p-value, Direction - most impact first.
    ranking: pd.DataFrame
    #: Cross-validated R² (numeric response) or accuracy (categorical), 0..1.
    explained: float
    numeric: bool
    rows: int


def _is_numeric(values: pd.Series) -> bool:
    numbers = pd.to_numeric(values, errors="coerce")
    return bool(numbers.notna().all()) and values.nunique() > 2


def _encode(data: pd.DataFrame, factors: list[str], numeric: dict[str, bool]) -> pd.DataFrame:
    parts = []
    for name in factors:
        if numeric[name]:
            parts.append(pd.to_numeric(data[name]).astype(float).rename(name))
        else:
            parts.append(pd.get_dummies(data[name].astype(str), prefix=name, prefix_sep="\x1f", dtype=float))
    return pd.concat(parts, axis=1)


def _forest(numeric_response: bool, seed: int):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    kind = RandomForestRegressor if numeric_response else RandomForestClassifier
    return kind(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=-1)


def _score(model: Any, x: pd.DataFrame, y: np.ndarray, numeric_response: bool) -> float:
    predicted = model.predict(x)
    if numeric_response:
        total = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - float(np.sum((y - predicted) ** 2)) / total if total > 0 else 0.0
    return float(np.mean(predicted == y))


def _forest_importance(data: pd.DataFrame, response: str, factors: list[str], numeric: dict[str, bool],
                       numeric_response: bool, seed: int) -> tuple[np.ndarray, float]:
    """Each factor's permutation importance on held-out rows, and the held-out score."""
    from sklearn.model_selection import KFold, StratifiedKFold

    y = pd.to_numeric(data[response]).to_numpy(float) if numeric_response else data[response].astype(str).to_numpy()
    encoded = _encode(data, factors, numeric)
    rng = np.random.default_rng(seed)
    splitter = (KFold(FOLDS, shuffle=True, random_state=seed) if numeric_response
                else StratifiedKFold(min(FOLDS, int(pd.Series(y).value_counts().min())) or 2, shuffle=True,
                                     random_state=seed))
    drops, scores = np.zeros(len(factors)), []
    for train, test in splitter.split(encoded, y):
        model = _forest(numeric_response, seed).fit(encoded.iloc[train], y[train])
        held_out = data.iloc[test]
        base = _score(model, encoded.iloc[test], y[test], numeric_response)
        scores.append(base)
        for index, name in enumerate(factors):
            shuffled = held_out.copy()
            shuffled[name] = rng.permutation(shuffled[name].to_numpy())
            columns = _encode(shuffled, factors, numeric).reindex(columns=encoded.columns, fill_value=0.0)
            drops[index] += base - _score(model, columns, y[test], numeric_response)
    return drops / max(len(scores), 1), float(np.mean(scores)) if scores else float("nan")


def _linear_pvalues(data: pd.DataFrame, response: str, factors: list[str],
                    numeric: dict[str, bool]) -> tuple[dict[str, float], dict[str, float]]:
    """Type II ANOVA p-value and slope (numeric factors) of each factor in one linear model."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    safe = {name: f"f{index}" for index, name in enumerate(factors)}
    frame = pd.DataFrame({"y": pd.to_numeric(data[response]).astype(float)})
    for name, alias in safe.items():
        frame[alias] = pd.to_numeric(data[name]).astype(float) if numeric[name] else data[name].astype(str)
    terms = [alias if numeric[name] else f"C({alias})" for name, alias in safe.items()]
    pvalues, slopes = {}, {}
    try:
        model = ols("y ~ " + " + ".join(terms), data=frame).fit()
        table = sm.stats.anova_lm(model, typ=2)
        for name, alias in safe.items():
            term = alias if numeric[name] else f"C({alias})"
            pvalues[name] = float(table.loc[term, "PR(>F)"]) if term in table.index else float("nan")
            if numeric[name]:
                slopes[name] = float(model.params.get(alias, float("nan")))
    except Exception:  # noqa: BLE001 - collinear or too few rows: the forest still ranks
        pass
    return pvalues, slopes


def _univariate_pvalues(data: pd.DataFrame, response: str, factors: list[str],
                        numeric: dict[str, bool]) -> dict[str, float]:
    """A categorical response: Kruskal-Wallis for a numeric factor, chi-squared for a categorical one."""
    y = data[response].astype(str)
    found = {}
    for name in factors:
        try:
            if numeric[name]:
                groups = [pd.to_numeric(part).to_numpy(float) for _level, part in data[name].groupby(y)]
                found[name] = float(stats.kruskal(*groups).pvalue)
            else:
                found[name] = float(stats.chi2_contingency(pd.crosstab(data[name].astype(str), y)).pvalue)
        except ValueError:
            found[name] = float("nan")
    return found


def impact(frame: pd.DataFrame, response: str, factors: list[str], *, seed: int = 1) -> ImpactAnswer:
    """Which of *factors* has the most impact on *response*, in words."""
    factors = [name for name in factors if name != response]
    if not factors:
        raise ValueError("Choose at least one factor other than the response.")
    data = frame[[response, *factors]].dropna().reset_index(drop=True)
    if len(data) < max(15, 3 * len(factors)):
        raise ValueError(f"{len(data)} complete rows are too few for {len(factors)} factor(s).")
    numeric_response = _is_numeric(data[response])
    if not numeric_response and data[response].nunique() < 2:
        raise ValueError(f"{response} has a single value.")
    numeric = {name: _is_numeric(data[name]) for name in factors}

    drops, explained = _forest_importance(data, response, factors, numeric, numeric_response, seed)
    positive = np.clip(drops, 0.0, None)
    shares = positive / positive.sum() * 100.0 if positive.sum() > 0 else np.zeros_like(positive)
    if numeric_response:
        pvalues, slopes = _linear_pvalues(data, response, factors, numeric)
    else:
        pvalues, slopes = _univariate_pvalues(data, response, factors, numeric), {}

    directions = {}
    for name in factors:
        p = pvalues.get(name, float("nan"))
        if not numeric_response or not (p < ALPHA):
            directions[name] = ""
        elif numeric[name]:
            directions[name] = "rises" if slopes.get(name, 0.0) > 0 else "falls"
        else:
            means = pd.to_numeric(data[response]).groupby(data[name].astype(str)).mean()
            directions[name] = f"highest at {means.idxmax()}"
    ranking = pd.DataFrame({
        "Factor": factors,
        "Impact %": np.round(shares, 1),
        "p-value": [pvalues.get(name, float("nan")) for name in factors],
        "Direction": [directions[name] for name in factors],
    }).sort_values("Impact %", ascending=False, kind="stable").reset_index(drop=True)

    def matters(row: pd.Series) -> bool:
        p = row["p-value"]
        return row["Impact %"] >= LITTLE_SHARE and not (p >= ALPHA)

    important = ranking[ranking.apply(matters, axis=1)]
    little = ranking[(ranking["Impact %"] < LITTLE_SHARE) | (ranking["p-value"] >= ALPHA)]
    sentences: list[tuple[str, dict[str, Any]]] = []
    if important.empty:
        verdict = NO
        sentences.append(("None of these factors shows a clear impact on “{response}”.", {"response": response}))
    else:
        verdict = YES
        first = important.iloc[0]
        sentences.append(("“{first}” has the most impact on “{response}”: {share:.0f}% of what the factors "
                          "explain.", {"first": first["Factor"], "response": response, "share": first["Impact %"]}))
        direction = first["Direction"]
        if direction in ("rises", "falls"):
            template = ("“{response}” rises as “{first}” rises." if direction == "rises"
                        else "“{response}” falls as “{first}” rises.")
            sentences.append((template, {"response": response, "first": first["Factor"]}))
        elif direction.startswith("highest at "):
            sentences.append(("“{response}” is highest when “{first}” is {level}.",
                              {"response": response, "first": first["Factor"],
                               "level": direction.removeprefix("highest at ")}))
        others = important.iloc[1:3]
        if len(others) == 1:
            sentences.append(("Then “{second}” ({second_share:.0f}%).",
                              {"second": others.iloc[0]["Factor"], "second_share": others.iloc[0]["Impact %"]}))
        elif len(others) == 2:
            sentences.append(("Then “{second}” ({second_share:.0f}%) and “{third}” ({third_share:.0f}%).",
                              {"second": others.iloc[0]["Factor"], "second_share": others.iloc[0]["Impact %"],
                               "third": others.iloc[1]["Factor"], "third_share": others.iloc[1]["Impact %"]}))
    values = {"response": response, "explained": max(explained, 0.0) * 100.0, "n": len(data)}
    if numeric_response:
        sentences.append(("Together the factors explain {explained:.0f}% of the variation of “{response}”, "
                          "judged on rows the model had not seen.", values))
        if explained < 0.1:
            sentences.append(("Most of the variation of “{response}” is not explained by these factors.", values))
    else:
        baseline = float(data[response].astype(str).value_counts(normalize=True).iloc[0]) * 100.0
        sentences.append(("From the factors, the group of “{response}” is guessed right {explained:.0f}% of the "
                          "time, against {baseline:.0f}% by always guessing the most common one.",
                          {**values, "baseline": baseline}))
    if not little.empty:
        sentences.append(("Little or no impact: {names}.", {"names": ", ".join(little["Factor"])}))
    sentences.append(("How it was decided: a random forest, each factor shuffled in turn on rows it had not seen "
                      "({n} rows); the p-values from a linear model.", values) if numeric_response else
                     ("How it was decided: a random forest, each factor shuffled in turn on rows it had not seen "
                      "({n} rows); the p-values from a test per factor.", values))
    sentences.append(("Impact is not proof of cause: a factor can stand in for another one that was not "
                      "measured.", values))
    return ImpactAnswer("What has the most impact?", verdict, sentences, ranking, float(explained),
                        numeric_response, len(data))
