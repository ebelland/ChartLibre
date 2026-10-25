"""Demo: the assistant tried without a key or a connection - a fixed script, not an AI.

It goes through what a real provider does, with the application's own tools:
it lists the tables, describes the one the question names (else the first),
then - as far as Settings > Assistant allows - runs an analysis on it and
adds a standardized column, which the user confirms and Undo removes. Read
only, it asks the database a summary instead. The answer is streamed as a
model's is. Nothing leaves the computer: no consent and no key are needed.
"""
from __future__ import annotations

import csv
import io
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from app.plugin_api.ai.provider import ACCURATE, FAST, AIProvider, ChatSession, ToolCall, ToolResult, ToolSpec, Turn
from app.utils.i18n import _

#: A column of describe_table's answer: "- mass (REAL, continuous): 0 missing; min 7.6, max 14.1, mean 11, sd 1.4".
_COLUMN = re.compile(r"^- (?P<name>.+) \((?P<type>[^()]*), (?P<kind>[a-z]+)\): \d+ missing; (?P<rest>.*)$")
_TABLE = re.compile(r"^(?P<name>[^:]+): (?P<rows>\d+) rows")
#: A pause between the words of the answer: it streams as a model's does.
_WORD_DELAY_S = 0.012


@dataclass
class _Column:
    name: str
    kind: str
    summary: str
    numeric: bool
    distinct: int = 0
    mean: float | None = None
    sd: float | None = None


@dataclass
class _Script:
    """Where the script is, and what it found so far."""

    question: str
    step: str = "tables"
    tables: list[str] = field(default_factory=list)
    table: str = ""
    rows: int = 0
    columns: list[_Column] = field(default_factory=list)
    found: list[str] = field(default_factory=list)


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value == value else None  # not NaN


def _markdown_table(csv_text: str) -> str:
    rows = [row for row in csv.reader(io.StringIO(csv_text)) if row]
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:11]]
    return "\n".join(lines)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class DemoSession(ChatSession):
    def __init__(self, tools: Sequence[ToolSpec]) -> None:
        self._tools = {spec.name for spec in tools}
        self._script: _Script | None = None
        self._calls = 0

    def send(self, *, model: str, user_text: str | None = None, tool_results: Sequence[ToolResult] = (),
             on_text: Callable[[str], None], cancel: threading.Event) -> Turn:
        if user_text is not None:
            self._script = _Script(question=user_text)
            return self._call(model, "list_tables", {})
        script = self._script
        assert script is not None
        result = tool_results[0] if tool_results else ToolResult("", "", True)
        if script.step == "tables":
            return self._after_tables(model, script, result, on_text, cancel)
        if script.step == "describe":
            return self._after_describe(model, script, result, on_text, cancel)
        if script.step == "analysis":
            self._after_analysis(script, result)
            return self._change_or_answer(model, script, on_text, cancel)
        if script.step == "change":
            script.found.append(_("The change was declined: nothing was changed.")
                                if "declined" in result.content else
                                _("Column **{name}** added - Edit > Undo removes it.").format(
                                    name=result.content.split("'")[1] if "'" in result.content else "")
                                if not result.is_error else _("The column could not be added: {error}").format(
                                    error=result.content))
        return self._answer(model, script, on_text, cancel)

    # -- the script ------------------------------------------------------

    def _after_tables(self, model: str, script: _Script, result: ToolResult, on_text: Callable[[str], None],
                      cancel: threading.Event) -> Turn:
        tables = [(match["name"], int(match["rows"])) for line in result.content.splitlines()
                  if (match := _TABLE.match(line))]
        script.tables = [name for name, _rows in tables]
        # The table the question names, else the first of the user's own (not a "_" result).
        named = [item for item in tables if item[0].lower() in script.question.lower()]
        own = [item for item in tables if not item[0].startswith("_")] or tables
        if not (named or own):
            script.found.append(_("The project has no tables yet: import one from File, then ask again."))
            return self._answer(model, script, on_text, cancel)
        script.table, script.rows = (named or own)[0]
        script.step = "describe"
        return self._call(model, "describe_table", {"table": script.table})

    def _after_describe(self, model: str, script: _Script, result: ToolResult, on_text: Callable[[str], None],
                        cancel: threading.Event) -> Turn:
        for line in result.content.splitlines():
            match = _COLUMN.match(line)
            if match is None:
                continue
            rest = match["rest"]
            column = _Column(match["name"], match["kind"], rest.split("; most common")[0], rest.startswith("min "))
            if column.numeric:
                stats = dict(re.findall(r"(mean|sd) (\S+?)(?:,|$)", rest))
                column.mean, column.sd = _number(stats.get("mean", "")), _number(stats.get("sd", ""))
            elif distinct := re.match(r"(\d+) distinct", rest):
                column.distinct = int(distinct[1])
            script.columns.append(column)
        numeric = [c for c in script.columns if c.numeric and c.kind == "continuous"] or \
                  [c for c in script.columns if c.numeric]
        groups = [c for c in script.columns if not c.numeric and 2 <= c.distinct <= 12]
        if "run_analysis" in self._tools and numeric:
            script.step = "analysis"
            if groups:
                return self._call(model, "run_analysis", {"analysis": "Are there differences?",
                                                          "table": script.table, "roles": {
                                                              "response": [numeric[0].name],
                                                              "group": [groups[0].name]}})
            return self._call(model, "run_analysis", {"analysis": "Distribution", "table": script.table,
                                                      "roles": {"response": [numeric[0].name]}})
        if numeric or groups:
            # Read only: a summary asked of the database instead.
            script.step = "analysis"
            table = _quote(script.table)
            if groups and numeric:
                group, value = _quote(groups[0].name), _quote(numeric[0].name)
                sql = (f"SELECT {group}, COUNT(*) AS n, ROUND(AVG({value}), 3) AS mean FROM {table} "
                       f"GROUP BY 1 ORDER BY 1")
            elif numeric:
                value = _quote(numeric[0].name)
                sql = (f"SELECT COUNT({value}) AS n, MIN({value}) AS min, ROUND(AVG({value}), 3) AS mean, "
                       f"MAX({value}) AS max FROM {table}")
            else:
                group = _quote(groups[0].name)
                sql = f"SELECT {group}, COUNT(*) AS n FROM {table} GROUP BY 1 ORDER BY 2 DESC"
            return self._call(model, "query", {"sql": sql})
        return self._answer(model, script, on_text, cancel)

    def _after_analysis(self, script: _Script, result: ToolResult) -> None:
        if result.is_error:
            script.found.append(_("The analysis could not run: {error}").format(error=result.content))
            return
        head, _sep, report = result.content.partition("\nReport:\n")
        if not report and "," in head:  # a query's answer: CSV
            script.found.append(_("What the database answers:") + "\n\n" + _markdown_table(result.content))
            return
        # Its first sentences - the title repeated and the tables' headings after them left out.
        lines = [line.strip() for line in report.splitlines() if line.strip()][1:7]
        script.found.append(head.split(" done.")[0].join(("**", "**")) + " - "
                            + _("written to the project; the report begins:") + "\n\n"
                            + "\n".join(f"> {line}  " for line in lines))

    def _change_or_answer(self, model: str, script: _Script, on_text: Callable[[str], None],
                          cancel: threading.Event) -> Turn:
        names = {column.name for column in script.columns}
        candidates = [c for c in script.columns if c.numeric and c.mean is not None and c.sd]
        if "add_column" in self._tools and candidates:
            column = candidates[0]
            name = re.sub(r"\W+", "_", column.name).strip("_") + "_z"
            if name not in names:
                script.step = "change"
                return self._call(model, "add_column", {
                    "table": script.table, "name": name, "type": "REAL",
                    "expression": f"({_quote(column.name)} - {column.mean!r}) / {column.sd!r}"})
        return self._answer(model, script, on_text, cancel)

    # -- turns -----------------------------------------------------------

    def _call(self, model: str, name: str, arguments: Mapping[str, object]) -> Turn:
        self._calls += 1
        return Turn(stop="tools", tool_calls=[ToolCall(f"demo-{self._calls}", name, dict(arguments))], model=model)

    def _answer(self, model: str, script: _Script, on_text: Callable[[str], None],
                cancel: threading.Event) -> Turn:
        parts = [_("**Demo** - a fixed script, not an AI: it shows what the assistant does with the project's "
                   "tools. With a real provider the answer is written for your question.")]
        if script.table:
            parts.append(_("The project has {count} tables; I looked at **{table}** ({rows} rows).").format(
                count=len(script.tables), table=script.table, rows=script.rows))
        if script.columns:
            rows = [f"| {column.name} | {_(column.kind.capitalize())} | {column.summary} |" for column in script.columns[:10]]
            parts.append("\n".join([f"| {_('Column')} | {_('Kind')} | {_('Summary')} |", "|---|---|---|", *rows]))
        parts += script.found
        text = "\n\n".join(parts)
        turn = Turn(model=model, input_tokens=len(script.question) // 4, output_tokens=len(text) // 4)
        for word in re.split(r"(?<=\s)", text):
            if cancel.is_set():
                turn.stop = "stopped"
                return turn
            turn.text += word
            on_text(word)
            time.sleep(_WORD_DELAY_S)
        return turn


class DemoProvider(AIProvider):
    Name = "Demo"
    Description = ("Tries the assistant without a key or a connection: a fixed script that uses the real tools. "
                   "Nothing leaves the computer.")
    SENDS_DATA = False

    def models(self) -> Mapping[str, str]:
        return {ACCURATE: "demo", FAST: "demo-fast"}

    def classify(self, text: str, *, api_key: str) -> str:
        # As a router would: a short question is a simple one.
        del api_key
        return FAST if len(text.split()) <= 6 else ACCURATE

    def start(self, *, api_key: str, system: str, tools: Sequence[ToolSpec]) -> ChatSession:
        del api_key, system
        return DemoSession(tools)
