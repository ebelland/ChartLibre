"""What the guided analysis wizard offers for the columns someone picked (G-02).

Columns first: a person picks one to three columns of a table, and this
says what can be done with them - charts to see them ("See it") and
questions an analysis answers ("Ask a question"), phrased with the columns'
own names. Only what the columns allow is offered; nothing greyed.

Each column is a measure, a group or a time (table_profile). The result -
the column the others might explain - is asked only when two measures could
be it; otherwise it follows from the types. Pure: a profile and names in,
plain data out; the wizard draws it and the main window opens it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.plugin_api.table_operations.column_roles import NOMINAL, ORDINAL
from app.plugin_api.table_operations.table_profile import TableProfile
from app.utils.i18n import _

#: The most charts and questions offered for a choice of columns.
MAX_IDEAS = 3

#: An aggregate a chart reads instead of the rows: the count of each group.
COUNT = "COUNT(*)"


@dataclass(frozen=True, slots=True)
class ChartIdea:
    """A chart of the columns: its type, its title, and the role -> column (or aggregate) it reads."""

    chart_type: str
    #: Untranslated, with the columns' names as {placeholders}.
    title: str
    values: dict[str, str]
    roles: dict[str, str]
    #: Rows grouped by this column (the chart reads an aggregate, as a count).
    group_by: str | None = None
    #: An aggregate of a column, as ("AVG", "mass"), read for a role.
    aggregates: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuestionIdea:
    """A question an analysis answers: its catalogue key, the question and why, the columns cast."""

    key: str
    #: Untranslated, with the columns' names as {placeholders}.
    question: str
    why: str
    values: dict[str, str]
    roles: dict[str, list[str]]
    kinds: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Plan:
    measures: list[str]
    groups: list[str]
    times: list[str]
    #: The measures that could be the result: the wizard asks which when there are two or more.
    results: list[str]
    result: str | None
    charts: list[ChartIdea]
    questions: list[QuestionIdea]


def _group_kinds(profile: TableProfile, groups: list[str]) -> dict[str, str]:
    """A group as an analysis takes it: its own type when categorical, nominal otherwise."""
    return {g: profile.kinds.get(g) if profile.kinds.get(g) in (NOMINAL, ORDINAL) else NOMINAL for g in groups}


def plan(profile: TableProfile, columns: list[str], result: str | None = None) -> Plan:
    """The charts and questions for *columns* of the table *profile* describes.

    *result* is the measure the others might explain, when the person chose
    one among several; the last measure picked otherwise.
    """
    measures = [c for c in columns if c in profile.measures]
    groups = [c for c in columns if c in profile.groups]
    times = [c for c in columns if c in profile.times]
    results = measures if len(measures) >= 2 else []
    y = result if result in measures else (measures[-1] if measures else None)
    others = [m for m in measures if m != y]
    charts: list[ChartIdea] = []
    questions: list[QuestionIdea] = []
    kinds = _group_kinds(profile, groups)

    if len(measures) == 1 and not groups and not times:
        m = measures[0]
        charts += [
            ChartIdea("Histogram", "How “{m}” is spread", {"m": m}, {"value": m}),
            ChartIdea("Box Plot", "“{m}”: median, quartiles and outliers", {"m": m}, {"value": m}),
            ChartIdea("Q-Q Plot", "Is “{m}” normal?", {"m": m}, {"value": m}),
        ]
        questions.append(QuestionIdea(
            "analysis:Distribution", "How is “{m}” distributed?",
            "Its quantiles, mean and spread, and whether it looks normal.", {"m": m}, {"response": [m]}))
    if len(groups) == 1 and not measures and not times:
        g = groups[0]
        charts += [
            ChartIdea("Pareto Chart", "How often each “{g}” occurs, most frequent first", {"g": g},
                      {"X": g, "Y": COUNT}, group_by=g),
            ChartIdea("Bar Chart", "Rows per “{g}”", {"g": g}, {"X": g, "Y": COUNT}, group_by=g),
        ]
    if len(groups) == 2 and not measures:
        a, b = groups
        charts.append(ChartIdea("Mosaic Plot", "“{a}” against “{b}”", {"a": a, "b": b}, {"x": b, "y": a}))
        questions.append(QuestionIdea(
            "analysis:Fit Y by X", "Does “{a}” depend on “{b}”?",
            "A contingency table, with the chi-squared test and Fisher's.", {"a": a, "b": b},
            {"response": [a], "factor": [b]}, kinds))
    if len(measures) == 2 and not groups and not times:
        x = others[0]
        charts += [
            ChartIdea("Scatter Plot", "“{y}” against “{x}”", {"x": x, "y": y}, {"x": x, "y": y}),
            ChartIdea("Hexbin", "Where the points of “{y}” against “{x}” crowd", {"x": x, "y": y},
                      {"x": x, "y": y}),
        ]
        questions += [
            QuestionIdea("analysis:Fit Y by X", "How does “{y}” change with “{x}”?",
                         "The fitted line, how well it fits, and the correlation.", {"x": x, "y": y},
                         {"response": [y], "factor": [x]}),
            QuestionIdea("analysis:Multivariate", "Are “{x}” and “{y}” related?",
                         "Their correlation, with its confidence interval and p-value.", {"x": x, "y": y},
                         {"response": [x, y]}),
        ]
    if len(measures) == 1 and len(groups) == 1 and not times:
        m, g = measures[0], groups[0]
        charts += [
            ChartIdea("Box Plot", "“{m}” by “{g}”", {"m": m, "g": g}, {"value": m, "group": g}),
            ChartIdea("Violin Plot", "The shape of “{m}” in each “{g}”", {"m": m, "g": g},
                      {"value": m, "group": g}),
            ChartIdea("Bar Chart", "Mean “{m}” per “{g}”", {"m": m, "g": g}, {"X": g, "Y": "Y"},
                      group_by=g, aggregates={"Y": ("AVG", m)}),
        ]
        questions += [
            QuestionIdea("analysis:Are there differences?", "Does “{m}” change between the groups of “{g}”?",
                         "The right test is chosen for you, and the answer given in words.", {"m": m, "g": g},
                         {"response": [m], "group": [g]}, kinds),
            QuestionIdea("analysis:Fit Y by X", "Which groups of “{g}” differ in “{m}”?",
                         "The means, the ANOVA and every pair of groups compared.", {"m": m, "g": g},
                         {"response": [m], "factor": [g]}, kinds),
        ]
    if times and measures:
        t, m = times[0], y or measures[0]
        idea = ChartIdea("Time Series", "“{m}” over “{t}”", {"m": m, "t": t}, {"x": t, "y": m})
        if groups:
            idea = ChartIdea("Time Series", "“{m}” over “{t}”, by “{g}”", {"m": m, "t": t, "g": groups[0]},
                             {"x": t, "y": m, "color": groups[0]})
        charts.append(idea)
        questions.append(QuestionIdea(
            "analysis:Has it changed over time?", "Has “{m}” changed over time?",
            "A steady trend or a sudden change: the tests are chosen for you.", {"m": m},
            {"response": [m], "time": [t]}))
    if len(measures) >= 2 and (len(measures) >= 3 or groups) and not times:
        pair = {f"column_{i + 1}": c for i, c in enumerate(measures[:6])}
        if groups:
            pair["group"] = groups[0]
        charts.append(ChartIdea("Pair Plot", "Every pair of the measures", {}, pair))
        if len(measures) >= 3:
            charts.append(ChartIdea("Scatter Plot", "“{y}” against “{x}”", {"x": others[0], "y": y},
                                    {"x": others[0], "y": y}))
        if groups and len(measures) == 2:
            charts.insert(0, ChartIdea("Scatter Plot", "“{y}” against “{x}”, coloured by “{g}”",
                                       {"x": others[0], "y": y, "g": groups[0]},
                                       {"x": others[0], "y": y, "color": groups[0]}))
    if len(columns) >= 3 and y is not None:
        factors = [c for c in columns if c != y and c not in times]
        if len(factors) >= 2:
            questions.append(QuestionIdea(
                "analysis:What has the most impact?", "What has the most impact on “{y}”?",
                "The other columns ranked by how much they move it, said in words.", {"y": y},
                {"response": [y], "factor": factors}, kinds))
    if len(measures) >= 3 or (len(measures) >= 2 and len(columns) >= 3):
        if len(measures) >= 2:
            questions.append(QuestionIdea(
                "analysis:Multivariate", "How do {names} move together?",
                "Every pair's correlation with its p-value, and the scatterplot matrix.",
                {"names": ", ".join(f"“{c}”" for c in measures)}, {"response": list(measures)}))
        if len(measures) >= 3:
            questions.append(QuestionIdea(
                "analysis:Principal Components", "Which few directions carry most of {names}?",
                "Principal components: what the measures share, and the groups the rows form.",
                {"names": ", ".join(f"“{c}”" for c in measures)}, {"response": list(measures)}))
        if groups and len(measures) >= 2:
            questions.append(QuestionIdea(
                "analysis:Discriminant", "Do the measures tell the groups of “{g}” apart?",
                "Which group a row belongs to, from its measures, and how often that is right.",
                {"g": groups[0]}, {"response": [groups[0]], "covariate": list(measures)}, kinds))
    if not charts and y is not None and groups:  # a measure and several groups: by the first
        charts.append(ChartIdea("Box Plot", "“{m}” by “{g}”", {"m": y, "g": groups[0]},
                                {"value": y, "group": groups[0]}))
    return Plan(measures=measures, groups=groups, times=times, results=results, result=y,
                charts=charts[:MAX_IDEAS], questions=questions[:MAX_IDEAS])


def chart_sql(table: str, idea: ChartIdea) -> tuple[str, dict[str, str]]:
    """The series query of *idea* on *table*: each role's column as the role; (sql, roles)."""
    def q(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    items = []
    for role, column in idea.roles.items():
        if column == COUNT:
            items.append(f"COUNT(*) AS {q(role)}")
        elif role in idea.aggregates:
            function, of = idea.aggregates[role]
            items.append(f"{function}({q(of)}) AS {q(role)}")
        else:
            items.append(f"{q(column)} AS {q(role)}")
    sql = f"SELECT {', '.join(items)} FROM {q(table)}"
    if idea.group_by:
        sql += f" WHERE {q(idea.group_by)} IS NOT NULL GROUP BY {q(idea.group_by)}"
        if idea.chart_type == "Pareto Chart":
            sql += f" ORDER BY {q('Y')} DESC"
    return sql, {role: role for role in idea.roles}


def make_chart(repo: Any, table: str, idea: ChartIdea) -> int:
    """Make *idea* on *table* as a new figure of one axis, undoably; its id.

    What Create chart does in the wizard - and the demos, which ask as a
    person would.
    """
    sql, roles = chart_sql(table, idea)
    title = _(idea.title).format(**idea.values)
    x_label = next((idea.roles[r] for r in ("x", "X", "group") if r in idea.roles), "")
    y_label = next((idea.roles[r] for r in ("y", "value") if r in idea.roles), "")
    if idea.roles.get("Y") == COUNT:
        y_label = _("Count")
    elif "Y" in idea.aggregates:
        y_label = _("Mean {column}").format(column=idea.aggregates["Y"][1])
    names = [c for c in idea.roles.values() if c not in (COUNT, "Y")]
    repo.snapshot_for_undo(repo.DESCRIPTOR_TABLES, label=f"New chart: {title}")
    figure_id = int(repo.create_figure_descriptor(name=title))
    axis_id = int(repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type=idea.chart_type, title=title,
        x_label=str(x_label), y_label=str(y_label), options={},
    ))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=", ".join(names) or table,
                                  sql_query=sql, roles=roles, style={})
    return figure_id
