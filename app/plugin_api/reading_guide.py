"""How to read a report: two sentences on one number of *this* result (H-04, the teaching mode).

Not a definition: what the number says here. "p = 0.003: if the groups had
the same mean of mass, a result at least this strong would come up by chance
about 3 times in 1000. Below 0.05, the usual line, so the result is called
real: the means differ." One explainer per statistic, so a p-value reads the
same in every report; a report's ``how_to_read`` picks the two or three
numbers it hinges on, and the box goes under its main result
(report_html.reading_guide).

Each explainer returns "" for a number that is not there (NaN, None), so a
report lists only what it has. The thresholds are said as the conventions
they are ("the usual line"), inside the two sentences.

Settings > Reports switches the box off (on for a new install: students
first); it is read when a report is made, so an exported report has it too.
"""
from __future__ import annotations

import math
from typing import Any

from app.utils.config import get_value
from app.utils.i18n import _

#: Settings > Reports, "Explain the results": the box in every report.
SETTING_KEY = "reports_how_to_read"
#: The usual line under which a p-value is called significant.
ALPHA = 0.05


def enabled() -> bool:
    """Whether reports carry their "How to read this" box."""
    return bool(get_value(SETTING_KEY, True))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def lookup(frame: Any, label: str, *, by: str, value: str = "Value") -> Any:
    """The *value* cell of the row of a report table whose *by* column is *label*; None when there is none."""
    try:
        rows = frame.loc[frame[by].astype(str) == label, value]
    except (AttributeError, KeyError, TypeError):
        return None
    return rows.iloc[0] if len(rows) else None


def _n(value: float) -> str:
    """A number in a sentence: three significant digits, no exponent below a million."""
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:.3g}" if 1e-3 <= abs(value) < 1e6 else f"{value:.2e}"


def _p(value: float) -> str:
    return "p < 0.0001" if value < 1e-4 else f"p = {value:.2g}"


def _by_chance(p: float) -> str:
    """How often, by chance: "about 3 times in 1000"."""
    if p >= 0.1:
        return _("about {count} times in 100").format(count=round(p * 100))
    if p >= 0.001:
        return _("about {count} times in 1000").format(count=max(1, round(p * 1000)))
    if p >= 1e-4:
        return _("less than once in 1000")
    return _("practically never")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def p_value(p: Any, *, null: str, shown: str, alpha: float = ALPHA) -> str:
    """*null*: what the test assumes ("the groups had the same mean of mass"); *shown*: what a small p
    says ("the means differ")."""
    value = _number(p)
    if value is None:
        return ""
    first = _("{p}: if {null}, a result at least this strong would come up by chance {how_often}.").format(
        p=_p(value), null=null, how_often=_by_chance(value))
    if value < alpha:
        second = _("Below {alpha}, the usual line, so the result is called real: {shown}.").format(alpha=alpha,
                                                                                                  shown=shown)
    else:
        second = _("Above {alpha}, the usual line: these data do not show that {shown}, which is not proof that "
                   "{null}.").format(alpha=alpha, shown=shown, null=null)
    return f"{first} {second}"


def normality(p: Any, *, test: str, column: str, alpha: float = ALPHA) -> str:
    value = _number(p)
    if value is None:
        return ""
    if value >= alpha:
        return _("{test}, {p}: values like these are common from a normal distribution, so nothing here says "
                 "{column} is not normal. Below {alpha}, the usual line, it would say so.").format(
            test=_(test), p=_p(value), column=column, alpha=alpha)
    return _("{test}, {p}: a normal distribution would rarely give values this far from its shape, so {column} "
             "is called not normal (below {alpha}, the usual line). Prefer the tests on ranks and medians to t and "
             "ANOVA, and look at the histogram for why: a long tail, two peaks.").format(
        test=_(test), p=_p(value), column=column, alpha=alpha)


# ----------------------------------------------------------------------
# One column
# ----------------------------------------------------------------------

def interval(low: Any, high: Any, *, what: str, level: int = 95) -> str:
    """A confidence interval for *what* ("the mean of mass")."""
    lo, hi = _number(low), _number(high)
    if lo is None or hi is None:
        return ""
    return _("{level}% interval for {what}: {low} to {high}. It is the range of values these rows leave "
             "plausible; more rows would narrow it.").format(level=level, what=what, low=_n(lo), high=_n(hi))


def spread(sd: Any, mean: Any, *, column: str) -> str:
    sd_value, mean_value = _number(sd), _number(mean)
    if sd_value is None or mean_value is None:
        return ""
    return _("Std Dev {sd}: a typical value of {column} lies about {sd} from its mean, {mean}. If the shape is "
             "normal, about two thirds of the values lie within one Std Dev of it, 95% within two.").format(
        sd=_n(sd_value), column=column, mean=_n(mean_value))


def effect_size(value: Any, *, name: str) -> str:
    """Cohen's d, eta² or Cramér's V, with the usual marks for small, medium and large."""
    number = _number(value)
    if number is None:
        return ""
    marks = {"Cohen's d": (0.2, 0.5, 0.8), "eta²": (0.01, 0.06, 0.14), "Cramér's V": (0.1, 0.3, 0.5)}.get(name)
    if marks is None:
        return ""
    meaning = {
        "Cohen's d": _("the two means are {value} standard deviations apart"),
        "eta²": _("the groups account for {percent}% of the variation"),
        "Cramér's V": _("the strength of the link between the two, from 0 (none) to 1 (complete)"),
    }[name].format(value=_n(abs(number)), percent=round(number * 100))
    size = abs(number)
    word = (_("large effect") if size >= marks[2] else _("medium effect") if size >= marks[1]
            else _("small effect") if size >= marks[0] else _("negligible effect"))
    return _("{name} = {value}: {meaning}. By the usual marks ({small} small, {medium} medium, {large} large) it is "
             "a {size}: how much the difference matters, where p only says whether there is one.").format(
        name=name, value=_n(number), meaning=meaning, size=word, small=marks[0], medium=marks[1], large=marks[2])


# ----------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------

def r_squared(value: Any, *, y: str) -> str:
    number = _number(value)
    if number is None:
        return ""
    percent = max(0, min(100, round(number * 100)))
    return _("R² = {value}: the fit accounts for {percent}% of how much {y} varies from row to row; the other "
             "{rest}% it leaves unexplained. Near 1 is a close fit - how close is enough depends on the field.").format(
        value=_n(number), percent=percent, y=y, rest=100 - percent)


def rmse(value: Any, *, y: str) -> str:
    number = _number(value)
    if number is None:
        return ""
    return _("RMSE = {value}: a value of {y} predicted by the fit is typically off by about {value}, in {y}'s own "
             "units.").format(value=_n(number), y=y)


def slope(estimate: Any, low: Any, high: Any, *, y: str, x: str) -> str:
    value, lo, hi = _number(estimate), _number(low), _number(high)
    if value is None or lo is None or hi is None:
        return ""
    first = _("Slope {value}: {y} changes by about {value} for each unit of {x} (95% interval {low} to {high}).").format(
        value=_n(value), y=y, x=x, low=_n(lo), high=_n(hi))
    if lo > 0 or hi < 0:
        second = _("The interval leaves out 0, so the change is called real.")
    else:
        second = _("The interval includes 0: these data do not show that {y} changes with {x}.").format(y=y, x=x)
    return f"{first} {second}"


def correlation(r: Any, *, x: str, y: str) -> str:
    value = _number(r)
    if value is None:
        return ""
    if value >= 0:
        template = _("r = {r}: {y} tends to rise as {x} rises; 1 or -1 would be a straight line, 0 no straight-line "
                     "link. They share {percent}% of their variation (r²) - a link, not proof that one causes the "
                     "other.")
    else:
        template = _("r = {r}: {y} tends to fall as {x} rises; 1 or -1 would be a straight line, 0 no straight-line "
                     "link. They share {percent}% of their variation (r²) - a link, not proof that one causes the "
                     "other.")
    return template.format(r=_n(value), y=y, x=x, percent=round(value * value * 100))


def auc(value: Any, *, level: str) -> str:
    number = _number(value)
    if number is None:
        return ""
    return _("AUC = {value}: of a row that is {level} and one that is not, the model gives the first the higher "
             "probability {percent}% of the time. 0.5 is a coin toss, 1 never wrong; above 0.8 is usually called "
             "good.").format(value=_n(number), level=level, percent=round(number * 100))


def logworth(value: Any, *, term: str) -> str:
    number = _number(value)
    if number is None:
        return ""
    return _("LogWorth {value} for {term}, the strongest effect: its p-value is about {p} (LogWorth is -log10 p). "
             "Above 2 - p below 0.01, the usual line for this ranking - an effect is called real.").format(
        value=_n(number), term=term, p=f"{10 ** -number:.1g}")


# ----------------------------------------------------------------------
# Processes
# ----------------------------------------------------------------------

def control_limits(center: Any, lower: Any, upper: Any, *, signals: int) -> str:
    c, lo, hi = _number(center), _number(lower), _number(upper)
    if c is None or lo is None or hi is None:
        return ""
    first = _("Limits {lower} to {upper} around {center}: a stable process keeps nearly every point (997 in 1000) "
              "inside them - its mean ± 3 sigma, not a specification.").format(lower=_n(lo), upper=_n(hi), center=_n(c))
    if signals:
        second = _("{count} signal(s) - a point outside, or a run on one side - point to a cause worth "
                   "finding.").format(count=int(signals))
    else:
        second = _("No signal: the variation looks like the process' own.")
    return f"{first} {second}"


# ----------------------------------------------------------------------
# Models that sort and predict
# ----------------------------------------------------------------------

def misclassification(rate: Any, *, count: Any, rows: Any) -> str:
    value, wrong, total = _number(rate), _number(count), _number(rows)
    if value is None or wrong is None or total is None:
        return ""
    return _("Misclassification {percent}%: the model puts {count} of {rows} rows in another group than their "
             "own. These are the rows it was fitted to: on new rows it usually does a little worse.").format(
        percent=_n(round(value * 100, 1)), count=int(wrong), rows=int(total))


def q_squared(value: Any, *, r2: Any, y: str) -> str:
    q, fitted = _number(value), _number(r2)
    if q is None:
        return ""
    first = _("Q² = {value}: on rows left out of each fit, the model accounts for {percent}% of how much {y} "
              "varies - its R² for prediction.").format(value=_n(q), percent=max(0, round(q * 100)), y=y)
    if fitted is not None and fitted - q > 0.2:
        second = _("Well below the R² of the fit ({r2}): part of what it learnt is noise; fewer components may "
                   "predict better.").format(r2=_n(fitted))
    else:
        second = _("Close to the R² of the fit ({r2}): what it learnt holds on rows it has not seen.").format(
            r2=_n(fitted)) if fitted is not None else ""
    return f"{first} {second}".strip()


def vip(value: Any, *, variable: str) -> str:
    number = _number(value)
    if number is None:
        return ""
    return _("VIP {value} for {variable}, the most important: how much it weighs in the components, against the "
             "average variable's 1. Above 0.8, the usual line, a variable is worth keeping.").format(
        value=_n(number), variable=variable)


def components(percent: Any, *, count: int, columns: int, kept: Any) -> str:
    value, above_one = _number(percent), _number(kept)
    if value is None:
        return ""
    return _("The first {count} components carry {percent}% of the variation of the {columns} columns: a scatter "
             "of them shows most of what the columns hold. An eigenvalue above 1 - more than one column's worth - "
             "is the usual line for a component worth keeping: {kept} here.").format(
        count=count, percent=round(value), columns=columns, kept=int(above_one) if above_one is not None else "-")


def silhouette(value: Any, *, clusters: Any) -> str:
    number, k = _number(value), _number(clusters)
    if number is None or k is None:
        return ""
    return _("Mean silhouette {value} with {k} clusters: from -1 to 1, how much nearer a row is to its own "
             "cluster than to the next one. Above 0.5 is usually called a clear structure, below 0.25 a weak "
             "one.").format(value=_n(number), k=int(k))


def share(portion: Any, *, factor: str) -> str:
    """A factor's part of what a model explains (a tree's splits)."""
    number = _number(portion)
    if number is None:
        return ""
    return _("{factor} accounts for {percent}% of what the splits explain: the factor the tree leans on most. "
             "A share is not a test - check that it holds on validation rows.").format(
        factor=factor, percent=round(number * 100))


# ----------------------------------------------------------------------
# Measurement, series, fields
# ----------------------------------------------------------------------

def gage(percent: Any, *, categories: Any) -> str:
    value, ndc = _number(percent), _number(categories)
    if value is None:
        return ""
    first = _("Gage R&R {percent}% of the study variation: the part the measuring system adds itself. Under 10% "
              "is usually called acceptable, 10 to 30% marginal, above 30% not.").format(percent=_n(round(value, 1)))
    if ndc is None:
        return first
    second = _("{ndc} distinct categories: the system tells that many groups of parts apart - 5 or more is the "
               "usual minimum.").format(ndc=int(ndc))
    return f"{first} {second}"


def aic(value: Any, *, model: str, runner_up: Any) -> str:
    best, second = _number(value), _number(runner_up)
    if best is None:
        return ""
    first = _("AIC {value} for {model}, the lowest of the models tried: AIC weighs the fit against the parameters "
              "it takes, and lower is better.").format(value=_n(best), model=model)
    if second is None:
        return first
    if second - best < 2:
        rest = _("The next is less than 2 above it: the two fit about as well.")
    else:
        rest = _("The next is {gap} above it: a clear choice (more than 2, the usual line).").format(
            gap=_n(second - best))
    return f"{first} {rest}"


def field_derivatives() -> str:
    return _("Divergence: where the field spreads out (above 0) or gathers in (below 0) - a source or a sink. "
             "Curl: where it turns, and which way - 0 where it flows without turning.")
