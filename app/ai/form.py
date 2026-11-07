"""Small models: the application decides, the model fills in a form (dev/plans/small_models_guided_plan.txt, C).

Choosing a tool, its arguments and the next step is too much for a 3B model; filling one form is not. A request
then takes three steps:

1. the model fills a JSON form - an action and its fields - whose tables, columns, chart types and analyses are
   enums built from the project, so it cannot name what is not there (``schema``, the provider's ``fill``);
2. the application checks it and runs the action with the ordinary tools, no model involved (``execute``);
3. the model writes the answer from that short result (``answer_prompt``, the provider's ``say``).

The permission level limits the actions offered, as it limits the tools.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.ai.tools import AssistantTools

#: action -> (what it is for, said to the model)
ACTIONS: dict[str, str] = {
    "answer": "answer from the table summary alone: a question about the data's columns or the application",
    "describe_table": "statistics of every column of the table: counts, missing values, min, max, mean",
    "query": "a SQL SELECT on the project, for counts, averages, the largest or smallest rows",
    "compare_groups": "is a measure different between groups (species, island, sex...)",
    "find_impact": "which factors affect a result the most",
    "create_chart": "draw a chart",
    "run_analysis": "run one of the listed analyses (statistics, correlation, fit...)",
    "list_charts": "which chart types there are",
    "list_analyses": "which analyses there are",
}
#: The tool each action needs (answer needs none).
_TOOL = {action: action for action in ACTIONS if action != "answer"}

INSTRUCTIONS = """Fill in this form for the user's request; the application then does it. Choose the action that \
does what the user asked, and fill only the fields that action needs:
{actions}
Fields: table; measure (the column measured), groups (the column of the groups), result and factors (for \
find_impact); chart_type, x, y (a list), group (colours) for create_chart; analysis and columns for run_analysis; \
sql for query (SQLite, double quotes around names).

Tables and their columns:
{tables}"""


def _columns(tools: AssistantTools, table: str) -> list[str]:
    return [str(name) for name in tools._repo.get_columns(table) if name not in ("Hide", "Selected", "_id")]


def schema(tools: AssistantTools, level: str) -> dict[str, Any]:
    """The form's JSON schema, its choices taken from the project and the permission level."""
    from app.scanners.axis_renderer_scanner import renderers

    allowed = {spec.name for spec in tools.specs(level)}
    actions = [action for action in ACTIONS if action == "answer" or _TOOL[action] in allowed]
    tables = [str(name) for name in tools._repo.list_user_tables()["Table"]]
    columns = sorted({column for table in tables for column in _columns(tools, table)}) or [""]
    column = {"type": "string", "enum": ["", *columns]}
    some_columns = {"type": "array", "items": {"type": "string", "enum": columns}}
    properties: dict[str, Any] = {
        "action": {"type": "string", "enum": actions},
        "table": {"type": "string", "enum": tables or [""]},
        "measure": column, "groups": column, "result": column, "factors": some_columns,
        # Required with the action, never empty: left optional, a chart came without its type (a scatter).
        "chart_type": {"type": "string", "enum": sorted(str(entry["value"]) for entry in renderers)},
        "x": column, "y": some_columns, "group": column,
        "analysis": {"type": "string", "enum": ["", *sorted(tools.catalogue())]},
        "columns": some_columns,
        "sql": {"type": "string"},
    }
    required = ["action", "table", *(["chart_type"] if "create_chart" in actions else [])]
    return {"type": "object", "properties": properties, "required": required}


def instructions(tools: AssistantTools, level: str) -> str:
    """What the model is told with the form: the actions it may choose, and the tables with their columns."""
    allowed = {spec.name for spec in tools.specs(level)}
    actions = "\n".join(f"- {action}: {purpose}" for action, purpose in ACTIONS.items()
                        if action == "answer" or _TOOL[action] in allowed)
    tables = "\n".join(f"- {table}: {', '.join(_columns(tools, table))}"
                       for table in (str(name) for name in tools._repo.list_user_tables()["Table"]))
    return INSTRUCTIONS.format(actions=actions, tables=tables or "(none)")


def _one(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _many(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def _roles(tools: AssistantTools, analysis: str, form: Mapping[str, Any]) -> dict[str, list[str]]:
    """The analysis' roles cast from the form: the measured column as its response, groups as its groups..."""
    from app.scanners.data_operation_scanner import import_class_from_file

    entry = tools.catalogue().get(analysis)
    cls = import_class_from_file(entry) if entry is not None else None
    keys = [str(role.key) for role in getattr(cls, "ROLES", ())]
    named = [*_many(form.get("measure")), *_many(form.get("y")), *_many(form.get("columns"))]
    roles: dict[str, list[str]] = {}
    for key in keys:
        if key in ("response", "y", "input") and (form.get("measure") or form.get("y") or named):
            roles[key] = _many(form.get("measure")) or _many(form.get("y")) or named[:1]
        elif key in ("factor", "predictor", "x") and (form.get("factors") or form.get("x")):
            roles[key] = _many(form.get("factors")) or _many(form.get("x"))
        elif key in ("group", "by") and (form.get("groups") or form.get("group")):
            roles[key] = _many(form.get("groups")) or _many(form.get("group"))
    if not roles and keys and named:
        roles[keys[0]] = named
    return roles


def execute(tools: AssistantTools, form: Mapping[str, Any], level: str) -> tuple[str, str, bool]:
    """Run what the form asks: (the action, its result as text, whether it failed). No model involved."""
    action = str(form.get("action") or "answer")
    table = str(form.get("table") or "")
    if action not in ACTIONS:
        return action, f"Unknown action '{action}'.", True
    if action == "answer":
        return action, tools.run("describe_table", {"table": table}, level)[0] if table else "", False
    arguments: dict[str, Any] = {"table": table}
    if action == "query":
        arguments = {"sql": str(form.get("sql") or "")}
    elif action == "compare_groups":
        arguments |= {"measure": _one(form.get("measure")) or _one(form.get("y")),
                      "groups": _one(form.get("groups")) or _one(form.get("group")) or _one(form.get("x"))}
    elif action == "find_impact":
        result = _one(form.get("result")) or _one(form.get("measure"))
        arguments |= {"result": result,
                      "factors": [name for name in _many(form.get("factors")) or _many(form.get("columns"))
                                  if name != result]}
    elif action == "create_chart":
        chart_type = str(form.get("chart_type") or "Scatter Plot")
        cls = tools._chart_class(chart_type)
        known = {*getattr(cls, "RequiredRoles", ()), *getattr(cls, "OptionalRoles", ())}
        x, group = _one(form.get("x")), _one(form.get("group"))
        if x and "x" not in known:
            # A box plot has no x: the column across it is its groups ("body mass by species").
            group, x = group or x, ""
        # The colours' column: the chart's group role, else its color role (a scatter has no group).
        role = "group" if "group" in known else "color" if "color" in known else ""
        roles_text = f"{role}={group}" if group and role else ""
        arguments |= {"chart_type": chart_type,
                      "y": _many(form.get("y")) or _many(form.get("measure")) or _many(form.get("columns")),
                      "x": x, "roles_text": roles_text}
    elif action == "run_analysis":
        analysis = str(form.get("analysis") or "")
        arguments |= {"analysis": analysis, "roles": _roles(tools, analysis, form)}
    elif action in ("list_charts", "list_analyses"):
        arguments = {}
    content, failed = tools.run(_TOOL[action], arguments, level)
    return action, content, failed


def answer_prompt(request: str, action: str, result: str, failed: bool) -> str:
    """The message the model answers from: the request, what was done, and what it gave."""
    if failed:
        done = f"The application tried {action} and it failed: {result}\nSay so briefly and what to try instead."
    elif action == "answer":
        done = f"The table, summarised:\n{result}" if result else "Nothing was run."
    else:
        done = f"The application ran {action}. Its result:\n{result}"
    return (f"The user asked: {request}\n\n{done}\n\nAnswer the user in their language, in a few sentences, with "
            "the key numbers from the result. Do not invent numbers.")
