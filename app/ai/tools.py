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
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from app.ai import settings
from app.data.repo._common import is_read_only_select
from app.plugin_api.ai.provider import ToolSpec
from app.plugin_api.table_operations.column_roles import declared_kind, guess_kind

#: The longest text a tool gives back; the rest is said to be cut.
MAX_RESULT = 8000
#: The rows a query gives back at most.
MAX_QUERY_ROWS = 200
#: How long an analysis may run before the tool gives up waiting.
ANALYSIS_TIMEOUT_S = 600


def _cut(text: str) -> str:
    return text if len(text) <= MAX_RESULT else text[:MAX_RESULT] + f"\n... ({len(text) - MAX_RESULT} characters cut)"


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
            names = sorted(self.catalogue())
            tools += [
                spec("list_analyses", "The analyses that can be run: each one's column roles (with the modelling "
                     "types they take), its parameters and their choices, its models."),
                spec("run_analysis", "Run an analysis as its window's OK does: it writes its result tables, draws its "
                     "figure (unless draw_on is none), keeps its report and is one Undo step. Returns the report.",
                     {"analysis": {"type": "string", "enum": names}, "table": text,
                      "roles": {"type": "object", "additionalProperties": {"type": "array", "items": text},
                                "description": "Role key -> column names, as list_analyses gives them."},
                      "parameters": {"type": "object", "description": "Parameter name -> value; defaults otherwise."},
                      "model": {"type": "string", "description": "One of the analysis' models, when it has any."},
                      "draw_on": {"type": "string", "enum": ["new_figure", "new_axis", "same_axis", "none"]}},
                     ("analysis", "table", "roles")),
                spec("create_chart", "Draw a table's columns as a new figure: chart_type as the application names "
                     "them (Scatter Plot, Histogram, Box Plot, Time Series, Bar Chart...).",
                     {"table": text, "chart_type": text, "x": text, "y": {"type": "array", "items": text},
                      "title": text}, ("table", "chart_type", "y")),
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
        try:
            inspect.signature(handler).bind(**dict(arguments))
        except TypeError as error:  # an argument missing, or not one of the tool's
            return f"Wrong arguments for {name}: {error}", True
        try:
            return _cut(str(handler(**dict(arguments)))), False
        except Exception as error:  # noqa: BLE001 - told to the model, which can try otherwise
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
            numbers = pd.to_numeric(values, errors="coerce").dropna()
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
            from app.scanners.table_operation_scanner import table_operations

            self._catalogue = {str(entry["value"]): entry for entry in table_operations}
        return self._catalogue

    def _tool_list_analyses(self) -> str:
        from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam, IntParam
        from app.scanners.table_operation_scanner import import_class_from_file

        lines = []
        for name, entry in sorted(self.catalogue().items()):
            cls = import_class_from_file(entry)
            lines.append(f"## {name} - {entry.get('description', '')}")
            for role in getattr(cls, "ROLES", ()):
                most = f"at most {role.maximum}" if role.maximum else "any number"
                lines.append(f"  role {role.key} ({role.label}): {', '.join(role.kinds)}; at least {role.minimum}, "
                             f"{most} - {role.description}")
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

    def _tool_run_analysis(self, analysis: str, table: str, roles: Mapping[str, list[str]],
                           parameters: Mapping[str, Any] | None = None, model: str = "",
                           draw_on: str = "new_figure") -> str:
        from PySide6.QtWidgets import QApplication

        from app.scanners.table_operation_scanner import import_class_from_file

        entry = self.catalogue().get(analysis)
        if entry is None:
            raise ValueError(f"No analysis named '{analysis}'.")
        before = set(self._repo.list_user_tables()["Table"].astype(str))
        dialog = import_class_from_file(entry)(repo=self._repo, table=table)
        try:
            if dialog.table != table:
                raise ValueError(f"No table named '{table}'.")
            dialog.current_figure_id = self._current_figure()
            dialog.roles_widget.set_casting({key: list(columns) for key, columns in dict(roles).items()})
            for form in dialog.parameter_forms():
                form.set_values(dict(parameters or {}))
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

    def _tool_create_chart(self, table: str, chart_type: str, y: list[str], x: str = "", title: str = "") -> str:
        from app.scanners.axis_renderer_scanner import get_renderer

        if get_renderer(chart_type) is None:
            raise ValueError(f"No chart type '{chart_type}'.")
        columns = set(self._repo.get_columns(table))
        missing = [name for name in [*y, *([x] if x else [])] if name not in columns]
        if missing:
            raise ValueError(f"Not columns of {table}: {', '.join(missing)}")
        repo = self._repo
        repo.snapshot_for_undo(list(repo.DESCRIPTOR_TABLES), label=f"Assistant: chart of '{table}'")
        figure_id = int(repo.create_figure_descriptor(name=title or f"{table}: {', '.join(y)}"))
        axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type=chart_type,
                                                  title=title, x_label=x, y_label=y[0] if len(y) == 1 else "",
                                                  options={"grid": True}))
        for index, column in enumerate(y):
            select = f"{_quote(x)} AS x, {_quote(column)} AS y" if x else f"rowid AS x, {_quote(column)} AS y"
            repo.create_series_descriptor(axis_id=axis_id, series_index=index, name=column,
                                          sql_query=f"SELECT {select} FROM {_quote(table)}",
                                          roles={"x": "x", "y": "y"}, style={})
        self._changed([figure_id])
        return f"Figure {figure_id} drawn: {chart_type} of {', '.join(y)}" + (f" against {x}" if x else "") + "."

    # -- changes ---------------------------------------------------------

    def _tool_add_column(self, table: str, name: str, expression: str, type: str = "") -> str:  # noqa: A002
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

