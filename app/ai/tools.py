"""What the assistant can do in the project: the tools it is given, and how the application runs them.

The tools are the application's own functions, run here, on the GUI thread,
by the application - never code the model writes. Which ones it is given
follows Settings > Assistant:

* every level - read: the tables, a table's columns and summary (with sample
  rows when allowed), a read-only query, the figures, the analyses there are;
* Analyses - run an analysis as its window's OK would (tables, figure, report,
  history, one Undo step), and draw a chart;
* Everything - change the project's data: a computed column, a statement that
  writes. Each confirmed by the user first, each an Undo step.

Every result goes back to the model as text, kept short.
"""
from __future__ import annotations

import inspect
import re
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from app.ai import settings
from app.data.repo._common import is_read_only_select
from app.plugin_api.ai.provider import ToolSpec
from app.plugin_api.data_operations.column_roles import declared_kind, guess_kind
from app.utils.coercion import to_numbers

#: The longest text a tool gives back; the rest is said to be cut.
MAX_RESULT = 8000
#: The rows a query gives back at most.
MAX_QUERY_ROWS = 200
#: How long an analysis may run before the tool gives up waiting.
ANALYSIS_TIMEOUT_S = 600


def _cut(text: str, limit: int = MAX_RESULT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} characters cut)"


def _assignments(text: str) -> dict[str, str]:
    """"a=1; b=x,y" as {"a": "1", "b": "x,y"}: what any model can write, whatever its schema can say."""
    pairs = (part.split("=", 1) for part in str(text or "").split(";") if "=" in part)
    return {key.strip(): value.strip() for key, value in pairs}


def _closest(name: str, known: Iterable[str]) -> str:
    """The one of *known* a small model meant - "box_plot" for Box Plot, "body mass" for body_mass_g; else *name*."""
    def words(text: str) -> list[str]:
        return [word for word in re.split(r"[^a-z0-9]+", str(text).lower()) if word]

    wanted = words(name)
    choices = list(known)
    for test in (lambda have: have == wanted,
                 lambda have: [w for w in have if w not in ("plot", "chart")] == [w for w in wanted
                                                                                  if w not in ("plot", "chart")],
                 lambda have: bool(wanted) and all(word in have for word in wanted)):
        found = [choice for choice in choices if test(words(choice))]
        if len(found) == 1:
            return found[0]
    return name


def _column_list(value: Any) -> list[str]:
    """Columns as a list, also when a model wrote the list as text: "['a', 'b']" or "a, b"."""
    if isinstance(value, str):
        return [part.strip(" '\"") for part in value.strip("[] ").split(",") if part.strip(" '\"")]
    return [str(item) for item in value]


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _plain(html: str) -> str:
    """A report's HTML as text, for the model."""
    from PySide6.QtGui import QTextDocument

    document = QTextDocument()
    document.setHtml(html)
    return document.toPlainText().strip()


class AssistantTools:
    """The tools of one conversation, on *repo*.

    *confirm(question)* asks the user before a change and says whether to go ahead; *changed(figure_ids)* tells
    the window the project changed (the figures made, to show); *current_figure* is the figure the window shows.
    """

    def __init__(self, repo: Any, *, confirm: Callable[[str], bool], changed: Callable[[list[int]], None],
                 current_figure: Callable[[], int | None], keep_waiting: Callable[[], bool] = lambda: True) -> None:
        self._repo = repo
        self._confirm = confirm
        self._changed = changed
        self._current_figure = current_figure
        self._keep_waiting = keep_waiting
        self._catalogue: dict[str, Any] | None = None
        #: The longest text given back; a provider with a small model asks for less (AIProvider.MAX_RESULT).
        self.max_result = MAX_RESULT

    # ------------------------------------------------------------------
    # What the model is told
    # ------------------------------------------------------------------

    def specs(self, level: str) -> list[ToolSpec]:
        """The tools for *level* (settings.READ_ONLY, ANALYSES, EVERYTHING)."""
        def spec(name: str, description: str, properties: Mapping[str, Any] | None = None,
                 required: tuple[str, ...] = ()) -> ToolSpec:
            return ToolSpec(name, description, {"type": "object", "properties": dict(properties or {}),
                                                "required": list(required), "additionalProperties": False})

        text = {"type": "string"}
        tools = [
            spec("list_tables", "The project's data tables: name, rows, columns."),
            spec("describe_table", "A table's columns - SQL type, modelling type (continuous, ordinal, nominal), "
                 "missing values and a summary - and, when the user allows it, sample rows.",
                 {"table": text}, ("table",)),
            spec("query", "Run a read-only SQL SELECT on the project (SQLite) and get its rows as CSV.",
                 {"sql": text, "limit": {"type": "integer", "minimum": 1, "maximum": MAX_QUERY_ROWS}}, ("sql",)),
            spec("list_figures", "The project's figures: their axes (chart type, title) and series."),
        ]
        if level in (settings.ANALYSES, settings.EVERYTHING):
            tools += [
                spec("compare_groups", "Is a measure different between groups? The right test is chosen for you; "
                     "draws a figure and returns the answer in words.",
                     {"table": text, "measure": text, "groups": text}, ("table", "measure", "groups")),
                spec("find_impact", "Which columns have the most impact on a result? Draws a figure and returns "
                     "the answer in words.",
                     {"table": text, "result": text, "factors": {"type": "array", "items": text}},
                     ("table", "result", "factors")),
                spec("list_analyses", "The other analyses, one line each. describe_analysis gives one's details."),
                spec("describe_analysis", "One analysis: its column roles, parameters and models.",
                     {"analysis": text}, ("analysis",)),
                spec("run_analysis", "Run an analysis as its window's OK does: it writes its result tables, draws its "
                     "figure (unless draw_on is none), keeps its report and is one Undo step. Returns the report.",
                     {"analysis": text, "table": text,
                      "roles_text": {"type": "string", "description": "Role key = columns: response=mass; x=a,b"},
                      "parameters_text": {"type": "string", "description": "Parameter = value: alpha=0.05; n=3"},
                      "roles": {"type": "object", "additionalProperties": {"type": "array", "items": text},
                                "description": "The same as roles_text, as an object."},
                      "parameters": {"type": "object", "description": "The same as parameters_text, as an object."},
                      "model": {"type": "string", "description": "One of the analysis' models, when it has any."},
                      "draw_on": {"type": "string", "enum": ["new_figure", "new_axis", "same_axis", "none"]}},
                     ("analysis", "table")),
                spec("list_charts", "The chart types by family. With a chart_type: when to use it and its roles; "
                     "with a family: its charts and when to use each.",
                     {"family": text, "chart_type": text}),
                spec("create_chart", "Draw a table's columns as a new figure. y: the columns to draw, one series "
                     "each. x, and roles_text for other roles (group=species; color=sex), as list_charts gives them.",
                     {"table": text, "chart_type": text, "x": text, "y": {"type": "array", "items": text},
                      "roles_text": text, "title": text}, ("table", "chart_type", "y")),
            ]
        if level == settings.EVERYTHING:
            tools += [
                spec("add_column", "Add a column to a table computed by a SQLite expression on each row (the user "
                     "confirms first; one Undo step).",
                     {"table": text, "name": text, "expression": text,
                      "type": {"type": "string", "enum": ["REAL", "INTEGER", "TEXT", ""]}},
                     ("table", "name", "expression")),
                spec("change_data", "Run one SQL statement that changes the data - UPDATE, INSERT, DELETE, CREATE "
                     "TABLE ... AS SELECT. The user confirms first; one Undo step. Say what it does in description.",
                     {"sql": text, "description": text}, ("sql", "description")),
            ]
        return tools

    # ------------------------------------------------------------------
    # Running a call
    # ------------------------------------------------------------------

    def run(self, name: str, arguments: Mapping[str, Any], level: str) -> tuple[str, bool]:
        """Run the tool *name*; (its result as text, whether it failed)."""
        if name not in {spec.name for spec in self.specs(level)}:
            return f"'{name}' is not allowed at this permission level.", True
        handler = getattr(self, f"_tool_{name}")
        parameters = inspect.signature(handler).parameters
        # Names a small model mangles (" roles") trimmed, and arguments the tool does not have left out.
        arguments = {str(key).strip(): value for key, value in dict(arguments).items()}
        arguments = {key: value for key, value in arguments.items() if key in parameters}
        if "table" in parameters:
            # Only one choice: the project's only table, when the model left it out or named a column instead.
            names = [str(table) for table in self._repo.list_user_tables()["Table"]]
            given = str(arguments.get("table") or "")
            if len(names) == 1 and _closest(given, names) not in names:
                arguments["table"] = names[0]
        try:
            inspect.signature(handler).bind(**dict(arguments))
        except TypeError as error:  # an argument missing, or not one of the tool's
            return f"Wrong arguments for {name}: {error}", True
        try:
            return _cut(str(handler(**dict(arguments))), self.max_result), False
        except Exception as error:
            return f"{type(error).__name__}: {error}", True

    # -- read ------------------------------------------------------------

    def _tool_list_tables(self) -> str:
        frame = self._repo.list_user_tables()
        lines = []
        for table in frame["Table"].astype(str):
            columns = self._repo.get_columns(table)
            lines.append(f"{table}: {self._repo.row_count(table)} rows; columns: {', '.join(columns)}")
        return "\n".join(lines) or "The project has no tables."

    def _tool_describe_table(self, table: str) -> str:
        frame = self._repo.query_df(f"SELECT * FROM {_quote(table)}")
        frame = frame[[c for c in frame.columns if not str(c).startswith("_") and c not in ("Hide", "Selected")]]
        declared = {str(row[1]): str(row[2] or "") for row in self._repo.table_info(table)}
        lines = [f"{table}: {len(frame)} rows"]
        for column in frame.columns:
            values = frame[column]
            kind = declared_kind(declared.get(str(column))) or guess_kind(values)
            line = f"- {column} ({declared.get(str(column)) or 'no type'}, {kind}): {int(values.isna().sum())} missing"
            numbers = to_numbers(values).dropna()
            if len(numbers) and len(numbers) >= 0.9 * values.notna().sum():
                line += (f"; min {numbers.min():.6g}, max {numbers.max():.6g}, mean {numbers.mean():.6g}, "
                         f"sd {numbers.std():.6g}")
            else:
                counts = values.astype(str).value_counts()
                line += f"; {len(counts)} distinct; most common: " + ", ".join(
                    f"{value} ({count})" for value, count in counts.head(5).items())
            lines.append(line)
        rows = settings.sample_rows()
        if rows:
            lines.append(f"\nFirst {min(rows, len(frame))} rows (CSV):")
            lines.append(frame.head(rows).to_csv(index=False))
        return "\n".join(lines)

    def _tool_query(self, sql: str, limit: int = 50) -> str:
        # Read-only always, whatever the SQL guard says: the assistant's writes go through change_data.
        ok, reason = is_read_only_select(sql)
        if not ok:
            raise ValueError(f"Only a read-only SELECT: {reason}")
        frame = self._repo.query_df(sql)
        shown = frame.head(max(1, min(int(limit), MAX_QUERY_ROWS)))
        more = f"\n({len(frame) - len(shown)} more rows not shown)" if len(frame) > len(shown) else ""
        return shown.to_csv(index=False) + more

    def _tool_list_figures(self) -> str:
        lines = []
        for figure_id, figure_name in self._repo.load_figures_from_db():
            descriptor = self._repo.load_figure_descriptor(figure_id)
            lines.append(f"Figure {figure_id} '{figure_name}'")
            for axis in (descriptor.axes if descriptor else []):
                series = ", ".join(str(item.name) for item in axis.series)
                lines.append(f"  axis '{axis.title or ''}' ({axis.name}): {series}")
        return "\n".join(lines) or "The project has no figures."

    # -- analyses --------------------------------------------------------

    def catalogue(self) -> dict[str, Any]:
        """The table analyses by name: their discovery entries."""
        if self._catalogue is None:
            from app.scanners.data_operation_scanner import data_operations, reads_roles

            self._catalogue = {str(entry["value"]): entry for entry in data_operations if reads_roles(entry)}
        return self._catalogue

    def _tool_list_analyses(self) -> str:
        return "\n".join(f"{name}: {str(entry.get('description') or '')[:90]}"
                         for name, entry in sorted(self.catalogue().items()))

    def _tool_describe_analysis(self, analysis: str) -> str:
        from app.plugin_api.data_operations.parameter_spec import (
            BoolParam,
            ChoiceParam,
            FloatParam,
            IntParam,
        )
        from app.scanners.data_operation_scanner import import_class_from_file

        if analysis not in self.catalogue():
            raise ValueError(f"No analysis named '{analysis}'. They are: {', '.join(sorted(self.catalogue()))}.")
        lines = []
        entry = self.catalogue()[analysis]
        cls = import_class_from_file(entry)
        lines.append(f"## {analysis} - {entry.get('description', '')}")
        for role in getattr(cls, "ROLES", ()):
            most = f"at most {role.maximum}" if role.maximum else "any number"
            lines.append(f"  role {role.key} ({role.label}): {', '.join(role.kinds)}; at least {role.minimum}, "
                         f"{most} - {role.tooltip or role.label}")
        for param in getattr(cls, "PARAMS", ()):
            if isinstance(param, ChoiceParam):
                kind = "one of " + ", ".join(str(value) for _label, value in param.labelled_choices())
            elif isinstance(param, (IntParam, FloatParam)):
                kind = f"number {param.minimum}..{param.maximum}"
            elif isinstance(param, BoolParam):
                kind = "true/false"
            else:
                kind = "text"
            lines.append(f"  parameter {param.name}: {kind}, default {param.default!r}")
        models = list(getattr(cls, "MODELS", {}) or {})
        if models:
            lines.append(f"  models: {', '.join(models)}")
        return "\n".join(lines)

    def _table(self, table: str) -> str:
        """The table a model named, or the error that names the real ones."""
        names = [str(name) for name in self._repo.list_user_tables()["Table"]]
        found = _closest(table, names)
        if found not in names:
            raise ValueError(f"No table named '{table}'. Tables: {', '.join(names) or 'none'}.")
        return found

    def _columns(self, table: str, names: Iterable[str]) -> list[str]:
        """The columns a model named, or the error that names the real ones."""
        known = [str(name) for name in self._repo.get_columns(table) if name not in ("Hide", "Selected")]
        found = [_closest(name, known) for name in names]
        wrong = [name for name in found if name not in known]
        if wrong:
            raise ValueError(f"Not columns of {table}: {', '.join(wrong)}. Columns: {', '.join(known)}.")
        return found

    def _tool_compare_groups(self, table: str, measure: str, groups: str) -> str:
        table = self._table(table)
        measure, groups = self._columns(table, [measure, groups])
        return self._tool_run_analysis("Are there differences?", table, {"response": [measure], "group": [groups]})

    def _tool_find_impact(self, table: str, result: str, factors: list[str]) -> str:
        table = self._table(table)
        result = self._columns(table, [result])[0]
        return self._tool_run_analysis("What has the most impact?", table,
                                       {"response": [result],
                                        "factor": [name for name in self._columns(table, _column_list(factors))
                                                   if name != result]})

    def _tool_run_analysis(self, analysis: str, table: str, roles: Mapping[str, list[str]] | None = None,
                           parameters: Mapping[str, Any] | None = None, model: str = "",
                           draw_on: str = "new_figure", roles_text: str = "", parameters_text: str = "") -> str:
        from PySide6.QtWidgets import QApplication

        from app.scanners.data_operation_scanner import import_class_from_file

        analysis = _closest(analysis, self.catalogue())
        entry = self.catalogue().get(analysis)
        if entry is None:
            raise ValueError(f"No analysis named '{analysis}'. They are: {', '.join(sorted(self.catalogue()))}.")
        casting = {**{key: value.split(",") for key, value in _assignments(roles_text).items()}, **dict(roles or {})}
        values = {**_assignments(parameters_text), **dict(parameters or {})}
        table = self._table(table)
        casting = {key: self._columns(table, _column_list(columns)) for key, columns in casting.items()}
        before = set(self._repo.list_user_tables()["Table"].astype(str))
        cls = import_class_from_file(entry)
        if cls is None:  # its file went, or did not load: the log says why
            raise ValueError(f"The analysis '{analysis}' could not be loaded; the log says why.")
        dialog = cls(repo=self._repo, table=table)
        try:
            if dialog.reads_chart():  # the dialog remembers it was last used on a chart: the assistant names a table
                dialog.source_table_radio.setChecked(True)
                dialog._load_tables(table)
            if dialog.table != table:
                raise ValueError(f"No table named '{table}'.")
            dialog.current_figure_id = self._current_figure()
            dialog.roles_widget.set_casting({key: [str(column).strip() for column in columns] for key, columns in casting.items()})
            for form in dialog.parameter_forms():
                form.set_values(values)
            if model and getattr(dialog, "model_combo", None) is not None:
                index = dialog.model_combo.findData(model)
                if index < 0:
                    raise ValueError(f"'{model}' is not one of {analysis}'s models.")
                dialog.model_combo.setCurrentIndex(index)
            combo = getattr(dialog, "destination_combo", None)
            if combo is not None and combo.findData(draw_on) >= 0:
                combo.setCurrentIndex(combo.findData(draw_on))
            problems = dialog.problems()
            if problems:
                raise ValueError("; ".join(problems))
            dialog.ok()
            deadline = time.monotonic() + ANALYSIS_TIMEOUT_S
            while dialog.job_running and time.monotonic() < deadline and self._keep_waiting():
                QApplication.processEvents()
                time.sleep(0.01)
            if dialog.job_running:
                dialog.stop()
                raise RuntimeError("The analysis was stopped before it finished.")
            report = _plain(getattr(dialog.results_widget(), "_html", "") or "")
            figures = list(dialog.created_figure_ids)
        finally:
            dialog.close()
        written = sorted(set(self._repo.list_user_tables()["Table"].astype(str)) - before)
        self._changed(figures)
        return (f"{analysis} on {table} done. Tables written: {', '.join(written) or 'none'}. "
                f"Figures: {', '.join(map(str, figures)) or 'none'}.\nReport:\n{report}")

    def _chart_class(self, chart_type: str) -> Any:
        from app.scanners.axis_renderer_scanner import get_renderer, renderers
        from app.scanners.data_operation_scanner import import_class_from_file

        entry = get_renderer(_closest(chart_type, [str(item["value"]) for item in renderers]))
        if entry is None:
            raise ValueError(f"No chart type '{chart_type}': list_charts gives them.")
        return import_class_from_file(entry)

    def _tool_list_charts(self, family: str = "", chart_type: str = "") -> str:
        from app.scanners.axis_renderer_scanner import renderers

        if chart_type:
            cls = self._chart_class(chart_type)
            return (f"{chart_type}: {cls.Description}\n{cls.WhenToUse}\nRequired roles: "
                    f"{', '.join(cls.RequiredRoles)}. Optional: {', '.join(cls.OptionalRoles)}.")
        families: dict[str, list[str]] = {}
        for entry in renderers:
            families.setdefault(str(entry.get("category") or ""), []).append(str(entry["value"]))
        if family:
            if family not in families:
                raise ValueError(f"No family '{family}'. They are: {', '.join(sorted(families))}.")
            return "\n".join(f"{name}: {self._chart_class(name).WhenToUse}" for name in families[family])
        return "\n".join(f"{name}: {', '.join(charts)}" for name, charts in sorted(families.items()))

    def _tool_create_chart(self, table: str, chart_type: str, y: list[str], x: str = "", title: str = "",
                           roles_text: str = "") -> str:
        cls = self._chart_class(chart_type)
        chart_type = str(cls.Name)
        table = self._table(table)
        y = self._columns(table, _column_list(y))
        required = list(getattr(cls, "RequiredRoles", ()) or ())
        known = {*required, *(getattr(cls, "OptionalRoles", ()) or ())}
        shared = _assignments(roles_text)
        if x:
            shared.setdefault("x", x)
        unknown = sorted(set(shared) - known)
        if unknown:
            raise ValueError(f"{chart_type} has no role {', '.join(unknown)}. Its roles: {', '.join(sorted(known))}.")
        shared = dict(zip(shared, self._columns(table, shared.values()), strict=True))
        # What y's columns fill: y when the chart has it, else its first required role still free (a histogram's x).
        free = [role for role in required if role not in shared]
        drawn = "y" if "y" in known else (free[0] if free else "")
        if not drawn:
            raise ValueError(f"{chart_type}'s roles are all given: {', '.join(sorted(known))}.")
        absent = [role for role in free if role not in (drawn, "x")]
        if absent:
            raise ValueError(f"{chart_type} also needs: {', '.join(absent)} (in roles_text).")
        use_row = "x" in required and "x" not in shared and drawn != "x"
        repo = self._repo
        repo.snapshot_for_undo(list(repo.DESCRIPTOR_TABLES), label=f"Assistant: chart of '{table}'")
        figure_id = int(repo.create_figure_descriptor(name=title or f"{table}: {', '.join(y)}"))
        axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type=chart_type,
                                                  title=title, x_label=shared.get("x", ""),
                                                  y_label=y[0] if len(y) == 1 else "", options={"grid": True}))
        for index, column in enumerate(y):
            chosen = {**shared, drawn: column}
            select = [f"{_quote(name)} AS {_quote(role)}" for role, name in chosen.items()]
            if use_row:
                select.insert(0, 'rowid AS "x"')
            roles = {role: role for role in chosen} | ({"x": "x"} if use_row else {})
            repo.create_series_descriptor(axis_id=axis_id, series_index=index, name=column,
                                          sql_query=f"SELECT {', '.join(select)} FROM {_quote(table)}",
                                          roles=roles, style={})
        self._changed([figure_id])
        return f"Figure {figure_id} drawn: {chart_type} of {', '.join(y)}" + (f" against {x}" if x else "") + "."

    # -- changes ---------------------------------------------------------

    def _tool_add_column(self, table: str, name: str, expression: str, type: str = "") -> str:
        if not self._confirm(f"Add the column '{name}' to '{table}', computed as:\n\n{expression}"):
            return "The user declined the change."
        self._repo.add_column_from_expression(table, name, expression, type or None)
        self._changed([])
        return f"Column '{name}' added to '{table}'."

    #: Statements change_data refuses: the database's structure and its other files.
    _REFUSED = re.compile(r"\b(ATTACH|DETACH|PRAGMA|VACUUM|DROP|ALTER)\b", re.IGNORECASE)

    def _tool_change_data(self, sql: str, description: str) -> str:
        statement = str(sql).strip().rstrip(";").strip()
        if ";" in statement:
            raise ValueError("One statement at a time.")
        if self._REFUSED.search(statement):
            raise ValueError("Not this kind of statement: dropping or altering tables, attaching databases and "
                             "pragmas are not done by the assistant.")
        names = set(re.findall(r'\b(?:UPDATE|INTO|TABLE|FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_ ]*?)"?(?:\s|$|\()',
                               statement, flags=re.IGNORECASE))
        if any(name.startswith("__") for name in names):
            raise ValueError("The application's own tables are not changed by the assistant.")
        if not self._confirm(f"{description}\n\n{statement}"):
            return "The user declined the change."
        rows = self._repo.execute_change(statement, tables=sorted(names), label=f"Assistant: {description[:60]}")
        self._changed([])
        return f"Done: {rows if rows >= 0 else 'some'} rows changed."

