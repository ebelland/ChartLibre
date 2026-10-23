"""Quick answer: "Is it in control?" - a measure on a control chart, in words.

For someone who watches a process and wants to know whether it is stable:
whether its values vary only by the common causes that are always there, or
whether something special happened. The standard tool is a Shewhart chart
with the Nelson run rules (app.analysis.control_charts): an Individuals
chart, sigma from the average moving range - so a shift does not widen the
limits that are meant to catch it.

The order is a time or numeric column, or the rows' own order, as in
app.analysis.change. Sentences are templates (translated by the dialog) with
values, as in app.analysis.answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.analysis import control_charts as cc
from app.analysis.answers import NO, YES
from app.analysis.change import ROW_ORDER, _order

#: Fewer values than this give rough limits: the answer says so.
ROUGH_LIMITS_BELOW = 20
#: The signals listed one by one; the rest are counted.
LISTED_SIGNALS = 5
#: The rules that decide the answer: a point beyond the limits, nine in a row on one side.
#: All eight Nelson rules together flag half of all stable 60-value records; these two
#: about one in five, the rest are reported as patterns worth a look.
DECISIVE_RULES = frozenset({1, 2})


@dataclass(slots=True)
class ControlAnswer:
    question: str
    #: YES (in control) or NO (a special cause signals).
    verdict: str
    sentences: list[tuple[str, dict[str, Any]]]
    #: In order: Order (as given), Value, CL, UCL, LCL, Signal (the value where a rule is broken, else empty).
    series: pd.DataFrame
    center: float
    upper: float
    lower: float
    sigma: float
    signals: int
    rows: int
    temporal: bool


def is_in_control(frame: pd.DataFrame, measure: str, time: str = ROW_ORDER) -> ControlAnswer:
    """Is *measure* in statistical control along *time* (a column, or the rows' order when empty)?"""
    columns = [measure, time] if time else [measure]
    data = frame[columns].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    if len(data) < 5:
        raise ValueError(f"{len(data)} values are too few for a control chart: at least 5 are needed.")
    if time:
        order, temporal = _order(data, time)
        sort = np.argsort(order, kind="stable")
        data = data.iloc[sort].reset_index(drop=True)
    else:
        temporal = False
    values = data[measure].to_numpy(float)
    n = values.size
    positions = np.arange(n, dtype=float)
    chart = cc.variables_chart(cc.CHART_INDIVIDUALS, positions, values)
    when = data[time].astype(str).to_numpy() if time else np.arange(1, n + 1).astype(str)

    decisive = [v for v in chart.violations if set(v.rules) & DECISIVE_RULES]
    patterns = [v for v in chart.violations if not set(v.rules) & DECISIVE_RULES]
    signal = np.full(n, np.nan)
    for violation in decisive:
        signal[violation.index] = values[violation.index]
    series = pd.DataFrame({
        "Order": data[time] if time else np.arange(1, n + 1),
        "Value": values, "CL": chart.center, "UCL": chart.upper, "LCL": chart.lower, "Signal": signal,
    })

    common = {"measure": measure, "n": n, "center": chart.center, "upper": chart.upper, "lower": chart.lower,
              "sigma": chart.sigma, "count": len(decisive), "patterns": len(patterns)}
    sentences: list[tuple[str, dict[str, Any]]] = []
    if not decisive:
        verdict = YES
        sentences.append(("Yes: {measure} is in control - its {n} values stay between the limits "
                          "({lower:.4g} to {upper:.4g}) with no run of nine on one side.", common))
        if patterns:
            sentences.append(("{patterns} points show a pattern worth a look (the other run rules), "
                              "which a stable process also shows now and then.", common))
    else:
        verdict = NO
        sentences.append(("No: {measure} is not in control - one point signals a special cause." if len(decisive) == 1
                          else "No: {measure} is not in control - {count} points signal a special cause.", common))
        for violation in decisive[:LISTED_SIGNALS]:
            sentences.append(("At {when} ({value:.4g}): {rule}.",
                              {"when": when[violation.index], "value": violation.y,
                               "rule": "; ".join(violation.descriptions)}))
        if len(decisive) > LISTED_SIGNALS:
            sentences.append(("... and {more} more.", {"more": len(decisive) - LISTED_SIGNALS}))
    sentences.append(("Individuals chart: centre {center:.4g}, limits at 3 sigma, sigma {sigma:.4g} from the "
                      "average moving range. A special cause is a point beyond the limits or nine in a row on one side of the "
                      "centre.", common))
    if n < ROUGH_LIMITS_BELOW:
        sentences.append(("With {n} values the limits are rough: 20 or more make them trustworthy.", common))
    sentences.append(("In control means stable, not good: whether the values meet a specification is the "
                      "process capability's question.", {}))
    return ControlAnswer(
        question=f"Is {measure} in control?", verdict=verdict, sentences=sentences, series=series,
        center=chart.center, upper=chart.upper, lower=chart.lower, sigma=chart.sigma,
        signals=len(decisive), rows=n, temporal=temporal,
    )
