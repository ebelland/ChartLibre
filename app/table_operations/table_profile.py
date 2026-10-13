"""What a table holds, for the guided analysis: which goals and suggestions it can answer.

The guided analysis used to ignore the data. With the table selected in the
Tables panel it can say what the table contains (measures, columns that
group the rows, dates), which suggestions those columns can feed - and why
not, for the others - and ask concrete questions of it: "Does weight change
between the lines?", which opens "Are there differences?" with the two
columns already cast.

Pure: a frame and its declared SQL types in, plain data out; the panel draws it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

from app.analysis import catalog
from app.table_operations.column_roles import CONTINUOUS, NOMINAL, ORDINAL, declared_kind, guess_kind

#: A column with at most this many levels groups the rows; an ordinal one with more is a measure.
MAX_GROUP_LEVELS = 20
#: The concrete questions offered at most.
MAX_QUESTIONS = 3

_TIME_TYPES = ("DATE", "TIME")


@dataclass(slots=True)
class TableProfile:
    table: str
    rows: int
    #: Each column's modelling type, as the analysis windows would cast it.
    kinds: dict[str, str]
    #: Numbers to measure or predict.
    measures: list[str] = field(default_factory=list)
    #: Columns whose few levels split the rows into groups.
    groups: list[str] = field(default_factory=list)
    #: Dates and times.
    times: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"measures": len(self.measures), "groups": len(self.groups), "times": len(self.times)}


def _is_time(values: pd.Series, declared: str) -> bool:
    if any(word in declared.upper() for word in _TIME_TYPES) or pd.api.types.is_datetime64_any_dtype(values):
        return True
    present = values.dropna()
    if present.empty or pd.api.types.is_numeric_dtype(present):
        return False
    sample = present.astype(str).head(200)
    if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)


def profile(table: str, frame: pd.DataFrame, declared: dict[str, str] | None = None) -> TableProfile:
    """Sort *frame*'s columns into measures, groups and times."""
    declared = declared or {}
    result = TableProfile(table=table, rows=len(frame), kinds={})
    for name in frame.columns:
        column = str(name)
        values = frame[name]
        kind = declared_kind(declared.get(column)) or guess_kind(values)
        result.kinds[column] = kind
        if _is_time(values, str(declared.get(column) or "")):
            result.times.append(column)
            continue
        levels = int(values.nunique(dropna=True))
        if kind == CONTINUOUS:
            result.measures.append(column)
        elif kind == ORDINAL and levels > MAX_GROUP_LEVELS:
            result.measures.append(column)
        elif 2 <= levels <= MAX_GROUP_LEVELS and levels < max(len(frame), 2):
            result.groups.append(column)
    return result


# ----------------------------------------------------------------------
# Can the table answer it?
# ----------------------------------------------------------------------

#: Untranslated reasons, shown through _().
NEEDS_MEASURE = "Needs a numeric column."
NEEDS_TWO_MEASURES = "Needs two numeric columns."
NEEDS_GROUPS = "Needs a column that groups the rows (a few categories)."
NEEDS_TWO_GROUPS = "Needs two columns of categories."
NEEDS_TWO_COLUMNS = "Needs a response and at least one other column."
NEEDS_ORDER = "Needs a date column, or a numeric one to order the values by."

#: Suggestions with needs of their own, beyond what their goal and response kind ask.
_ENTRY_NEEDS: dict[str, str] = {
    "chart:Mosaic Plot": NEEDS_TWO_GROUPS,
    "chart:Scatter Plot": NEEDS_TWO_MEASURES,
    "series:Fit": NEEDS_TWO_MEASURES,
    "chart:Pair Plot": NEEDS_TWO_MEASURES,
    "series:Clustering": NEEDS_TWO_MEASURES,
    "series:Decomposition": NEEDS_TWO_MEASURES,
    "chart:Interaction Plot": NEEDS_TWO_GROUPS,
    "chart:Pareto Chart": NEEDS_GROUPS,
    "analysis:Discriminant": NEEDS_GROUPS,
}

#: What a goal asks of any table, whatever is suggested for it.
_GOAL_NEEDS: dict[str, str] = {
    "compare": NEEDS_GROUPS,
    "relate": NEEDS_TWO_COLUMNS,
    "explain": NEEDS_TWO_COLUMNS,
    "predict": NEEDS_TWO_COLUMNS,
    "classify": NEEDS_GROUPS,
    "signal": NEEDS_ORDER,
    "anomalies": NEEDS_MEASURE,
    "groups": NEEDS_TWO_MEASURES,
}


def _met(table: TableProfile, need: str) -> bool:
    measures, groups = len(table.measures), len(table.groups)
    columns = measures + groups + len(table.times)
    return {
        NEEDS_MEASURE: measures >= 1,
        NEEDS_TWO_MEASURES: measures >= 2,
        NEEDS_GROUPS: groups >= 1 and columns >= 2,
        NEEDS_TWO_GROUPS: groups >= 2,
        NEEDS_TWO_COLUMNS: columns >= 2,
        NEEDS_ORDER: measures >= 1 and (bool(table.times) or measures >= 2),
    }.get(need, True)


def goal_missing(table: TableProfile, goal: str) -> str | None:
    """Why *table* cannot answer *goal*, or None when it can."""
    need = _GOAL_NEEDS.get(goal)
    return need if need is not None and not _met(table, need) else None


def suggestion_missing(table: TableProfile, key: str, suits: str) -> str | None:
    """Why *table* cannot feed the suggestion *key* (for a response of kind *suits*), or None."""
    needs = [_ENTRY_NEEDS[key]] if key in _ENTRY_NEEDS else []
    if suits == catalog.NUMERIC:
        needs.append(NEEDS_MEASURE)
    elif suits == catalog.CATEGORICAL:
        needs.append(NEEDS_GROUPS)
    return next((need for need in needs if not _met(table, need)), None)


# ----------------------------------------------------------------------
# Concrete questions
# ----------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Question:
    #: Untranslated, with {measure} and {group}.
    template: str
    #: The catalogue key of what answers it.
    key: str
    #: Role -> columns, as the analysis window casts them.
    roles: dict[str, list[str]]
    kinds: dict[str, str]
    values: dict[str, str]


def questions(table: TableProfile) -> list[Question]:
    """Up to MAX_QUESTIONS concrete questions of the table.

    First "what has the most impact on" the first measure, when there are
    two or more other columns to rank; then "does this measure change
    between these groups?" - every measure with the first group, then the
    next group: the measures are what the table was made to record.
    """
    found: list[Question] = []
    if table.measures:
        result = table.measures[0]
        factors = [name for name in (*table.measures, *table.groups) if name != result]
        if len(factors) >= 2:
            found.append(Question(
                template="What has the most impact on “{measure}”?",
                key="analysis:What has the most impact?",
                roles={"response": [result], "factor": factors},
                kinds={},
                values={"measure": result},
            ))
    for group in table.groups:
        for measure in table.measures:
            if len(found) >= MAX_QUESTIONS:
                return found
            found.append(Question(
                template="Does “{measure}” change between the groups of “{group}”?",
                key="analysis:Are there differences?",
                roles={"response": [measure], "group": [group]},
                kinds={group: table.kinds[group] if table.kinds[group] in (NOMINAL, ORDINAL) else NOMINAL},
                values={"measure": measure, "group": group},
            ))
    return found
