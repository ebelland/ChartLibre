"""Gage R&R (AN-06): how much of the variation a measuring system adds, by the ANOVA method.

A crossed study - every operator measures every part, each more than once - as
JMP's Variability / Gauge Chart and the AIAG MSA manual analyse it. A two-way
ANOVA with interaction gives the mean squares, and from them the variance
components:

* repeatability (equipment variation): the error mean square;
* operator: (MS operator - MS interaction) / (parts x replicates);
* operator x part: (MS interaction - MS error) / replicates;
* part to part: (MS part - MS interaction) / (operators x replicates).

A negative estimate is set to zero. When the interaction is not significant
(p > 0.25, AIAG's rule) it is pooled into the error and the model refitted
without it. Reproducibility is operator + interaction; Gage R&R is
repeatability + reproducibility. Reported, for each source: the variance, its
share of the total (% contribution), the standard deviation, the study
variation (6 sigma) and its share of the total's (% study variation) and, with
a tolerance, of the tolerance (% tolerance); and the number of distinct
categories, 1.41 x part sigma / gage sigma.

Without an operator column it is a repeatability study of the parts alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: AIAG: the interaction is pooled into the error when its p-value is above this.
POOL_INTERACTION_ABOVE = 0.25
#: Study variation is this many standard deviations (AIAG 4th edition; older used 5.15).
STUDY_SPREAD = 6.0


@dataclass(slots=True)
class GageStudy:
    measure: str
    part: str
    operator: str
    #: The ANOVA table: Source, DF, SS, MS, F, Prob > F.
    anova: pd.DataFrame
    #: Source, Variance, % Contribution, Std Dev, Study Var, % Study Var (and % Tolerance with a tolerance).
    components: pd.DataFrame
    ndc: int
    pooled: bool
    parts: int
    operators: int
    replicates: float
    tolerance: float | None
    #: (template, values) sentences, as in app.analysis.answers.
    sentences: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def grr_percent(self) -> float:
        """Gage R&R as a % of the total study variation."""
        row = self.components.set_index("Source").loc["Gage R&R"]
        return float(row["% Study Var"])


def _anova(data: pd.DataFrame, factors: list[str], interaction: bool) -> pd.DataFrame:
    """Sums of squares of a balanced-or-not crossed design by statsmodels' OLS (type II)."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    terms = [f"C({name})" for name in factors]
    if interaction and len(factors) == 2:
        terms.append(f"C({factors[0]}):C({factors[1]})")
    model = smf.ols("y ~ " + " + ".join(terms), data=data).fit()
    table = anova_lm(model, typ=2)
    table = table.rename(columns={"sum_sq": "SS", "df": "DF", "F": "F", "PR(>F)": "Prob > F"})
    table["MS"] = table["SS"] / table["DF"]
    names = {f"C({factors[0]})": "part", "Residual": "error"}
    if len(factors) == 2:
        names[f"C({factors[1]})"] = "operator"
        names[f"C({factors[0]}):C({factors[1]})"] = "operator x part"
    table.index = [names.get(str(index), str(index)) for index in table.index]
    return table[["DF", "SS", "MS", "F", "Prob > F"]]


def gage_rr(frame: pd.DataFrame, measure: str, part: str, operator: str = "",
            tolerance: float | None = None) -> GageStudy:
    """The Gage R&R of *measure*, the parts in *part* and (optionally) the operators in *operator*."""
    columns = [measure, part] + ([operator] if operator else [])
    data = frame[columns].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    data = data.dropna()
    data = pd.DataFrame({"y": data[measure].to_numpy(float), "P": data[part].astype(str).to_numpy(),
                         **({"O": data[operator].astype(str).to_numpy()} if operator else {})})
    parts = int(data["P"].nunique())
    operators = int(data["O"].nunique()) if operator else 1
    if parts < 2:
        raise ValueError("A gage study needs at least two parts.")
    cells = data.groupby(["P", "O"] if operator else ["P"]).size()
    if int(cells.min()) < 2 and not operator:
        raise ValueError("Each part has to be measured at least twice: the repeats are what repeatability is.")
    replicates = float(cells.mean())
    if operator and int(cells.min()) < 1:
        raise ValueError("Every operator has to measure every part.")
    if tolerance is not None and not tolerance > 0:
        raise ValueError("The tolerance must be a positive width (USL - LSL).")

    factors = ["P", "O"] if operator else ["P"]
    pooled = False
    table = _anova(data, factors, interaction=bool(operator) and replicates > 1)
    if operator and "operator x part" in table.index:
        if float(table.loc["operator x part", "Prob > F"]) > POOL_INTERACTION_ABOVE:
            table = _anova(data, factors, interaction=False)
            pooled = True
    ms = table["MS"]
    error = float(ms["error"])
    if operator:
        interaction = float(ms["operator x part"]) if "operator x part" in ms else error
        var_inter = max(0.0, (interaction - error) / replicates) if "operator x part" in ms else 0.0
        var_operator = max(0.0, (float(ms["operator"]) - interaction) / (parts * replicates))
        var_part = max(0.0, (float(ms["part"]) - interaction) / (operators * replicates))
    else:
        var_inter = var_operator = 0.0
        var_part = max(0.0, (float(ms["part"]) - error) / replicates)
    var_repeat = error
    var_reprod = var_operator + var_inter
    var_grr = var_repeat + var_reprod
    var_total = var_grr + var_part

    rows = [("Gage R&R", var_grr), ("  Repeatability", var_repeat), ("  Reproducibility", var_reprod)]
    if operator:
        rows += [("    Operator", var_operator), ("    Operator x Part", var_inter)]
    rows += [("Part-to-Part", var_part), ("Total Variation", var_total)]
    components = pd.DataFrame(rows, columns=["Source", "Variance"])
    components["% Contribution"] = 100.0 * components["Variance"] / var_total if var_total > 0 else np.nan
    components["Std Dev"] = np.sqrt(components["Variance"])
    components["Study Var"] = STUDY_SPREAD * components["Std Dev"]
    total_sd = float(np.sqrt(var_total))
    components["% Study Var"] = 100.0 * components["Std Dev"] / total_sd if total_sd > 0 else np.nan
    if tolerance is not None:
        components["% Tolerance"] = 100.0 * components["Study Var"] / float(tolerance)
    components["Source"] = components["Source"].str.strip()
    ndc = int(max(1, np.floor(1.41 * np.sqrt(var_part) / np.sqrt(var_grr)))) if var_grr > 0 else 0

    study = GageStudy(measure=measure, part=part, operator=operator, anova=table.reset_index(names="Source"),
                      components=components, ndc=ndc, pooled=pooled, parts=parts, operators=operators,
                      replicates=replicates, tolerance=tolerance)
    study.sentences = _sentences(study)
    return study


def _sentences(study: GageStudy) -> list[tuple[str, dict[str, Any]]]:
    grr = study.grr_percent
    values = {"measure": study.measure, "grr": grr, "ndc": study.ndc}
    if grr < 10:
        verdict = "The measuring system is acceptable: Gage R&R is {grr:.1f}% of the study variation (under 10%)."
    elif grr <= 30:
        verdict = ("The measuring system may be acceptable, depending on its use and cost: Gage R&R is {grr:.1f}% of "
                   "the study variation (10% to 30%).")
    else:
        verdict = "The measuring system needs improving: Gage R&R is {grr:.1f}% of the study variation (over 30%)."
    out = [(verdict, values),
           ("It tells {ndc} distinct categories of part apart; 5 or more is adequate.", values)]
    if study.operator:
        comps = study.components.set_index("Source")
        larger = ("repeatability (the instrument)" if comps.loc["Repeatability", "Variance"]
                  >= comps.loc["Reproducibility", "Variance"] else "reproducibility (the operators)")
        out.append(("Most of the measuring error is {larger}.", {"larger": larger}))
        if study.pooled:
            out.append(("The operator x part interaction was not significant (p > 0.25) and was pooled into the "
                        "error.", {}))
    if study.tolerance is not None:
        tol = float(study.components.set_index("Source").loc["Gage R&R", "% Tolerance"])
        out.append(("Against the tolerance of {tolerance:g}, Gage R&R takes {tol:.1f}% of it.",
                    {"tolerance": study.tolerance, "tol": tol}))
    out.append(("Study variation is {spread:g} standard deviations (AIAG); variance components by the ANOVA method.",
                {"spread": STUDY_SPREAD}))
    return out


