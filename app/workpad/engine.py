"""The SymPy workpad's session: commands run one after another on one namespace, beside the project's tables.

Everything SymPy offers is there (``from sympy import *``), with a few words for the project:

    get(table, column)   a column - numbers as a numpy array, else a list; get(table) the whole table as a dict
    tables(), columns(table)
    put(table, {column: values})   a new table (Undo takes it back); also save, store, write
    plot(chart_type, x=..., y=..., group=..., title=...)   a chart; each role a column's values, or with
                         table="name" the names of its columns
    mean, std, median, minimum, maximum, count   on lists and arrays

With the filter on (the default) a command is read before it runs, and imports, names starting with "_", eval, exec,
open and the like are refused. It stops a mistake or a line pasted from the web, not someone set on harm: a project's
workpad is the user's own.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from app.ai.tools import AssistantTools

#: Builtins a workpad may call with the filter on.
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip", "True", "False", "None",
    "ArithmeticError", "Exception", "ValueError", "TypeError", "ZeroDivisionError",
)

#: Names refused with the filter on, wherever they appear.
UNSAFE_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint", "help", "exit", "quit", "memoryview", "classmethod", "staticmethod", "super", "object",
    # SymPy's own doors to Python: they run text as code.
    "sympify", "S", "parse_expr", "lambdastr", "init_session", "preview", "var",
})


class UnsafeCommand(ValueError):
    """A command the filter refuses: what, and where."""


def check(source: str) -> ast.Module:
    """*source* parsed, or UnsafeCommand naming the first construct the filter refuses."""
    tree = ast.parse(source, mode="exec")
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise UnsafeCommand(f"Line {line}: import is not allowed in the workpad.")
        if isinstance(node, ast.Global | ast.Nonlocal | ast.AsyncFunctionDef | ast.Await | ast.ClassDef):
            raise UnsafeCommand(f"Line {line}: {type(node).__name__.lower()} is not allowed in the workpad.")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in UNSAFE_NAMES):
            raise UnsafeCommand(f"Line {line}: '{node.id}' is not allowed in the workpad.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise UnsafeCommand(f"Line {line}: '.{node.attr}' is not allowed in the workpad.")
        if isinstance(node, ast.arg) and node.arg.startswith("_"):
            raise UnsafeCommand(f"Line {line}: '{node.arg}' is not allowed in the workpad.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise UnsafeCommand(f"Line {line}: text with '__' is not allowed in the workpad.")
    return tree


def _values(data: Any) -> list[Any]:
    """*data* - a list, an array, a SymPy matrix, a pandas column - as a flat list of plain values."""
    if hasattr(data, "tolist") and not isinstance(data, np.ndarray):
        data = data.tolist()
    array = np.asarray(data, dtype=object).ravel()
    plain = []
    for value in array:
        if hasattr(value, "is_number") and value.is_number:
            value = complex(value) if not value.is_real else float(value)
        plain.append(value)
    return plain


def _numbers(data: Any) -> np.ndarray:
    return np.asarray([float(value) for value in _values(data)], dtype=float)


class Workpad:
    """One session on *repo()*; *changed(figure_ids)* tells the window the project changed."""

    def __init__(self, repo: Callable[[], Any], *, changed: Callable[[list[int]], None] = lambda figures: None,
                 safe: bool = True) -> None:
        self._repo = repo
        self._changed = changed
        self.safe = safe
        self.namespace: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        """Forget every name: a new session."""
        import sympy

        namespace: dict[str, Any] = {name: getattr(sympy, name) for name in sympy.__all__}
        namespace.update(
            sympy=sympy, np=np,
            get=self.get, tables=self.tables, columns=self.columns, plot=self.plot,
            put=self.put, save=self.put, store=self.put, write=self.put,
            mean=lambda data: float(np.nanmean(_numbers(data))),
            std=lambda data: float(np.nanstd(_numbers(data), ddof=1)),
            median=lambda data: float(np.nanmedian(_numbers(data))),
            minimum=lambda data: float(np.nanmin(_numbers(data))),
            maximum=lambda data: float(np.nanmax(_numbers(data))),
            count=lambda data: len(_values(data)),
        )
        self.namespace = namespace
        self._builtins_namespace()

    def _builtins_namespace(self) -> None:
        if self.safe:
            self.namespace["__builtins__"] = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
        else:
            self.namespace["__builtins__"] = builtins.__dict__

    def names(self) -> dict[str, str]:
        """The names the user defined, each with a short view of its value."""
        defaults = self._defaults()
        shown = {}
        for name, value in self.namespace.items():
            if name.startswith("_") or (name in defaults and defaults[name] is value):
                continue
            text = " ".join(str(value).split())
            shown[name] = text if len(text) <= 60 else text[:57] + "..."
        return shown

    def _defaults(self) -> dict[str, Any]:
        if not hasattr(self, "_default_names"):
            self._default_names = dict(self.namespace)
        return self._default_names

    def run(self, source: str) -> tuple[str, Any]:
        """Run *source*; return what it printed and the value of its last line when that is an expression.

        Raises UnsafeCommand when the filter refuses it, or the command's own error.
        """
        self._defaults()
        self._builtins_namespace()
        tree = check(source) if self.safe else ast.parse(source, mode="exec")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        printed = io.StringIO()
        value = None
        with contextlib.redirect_stdout(printed):
            if tree.body:
                exec(compile(tree, "<workpad>", "exec"), self.namespace)  # noqa: S102 - the workpad's purpose
            if last is not None:
                expression = ast.Expression(last.value)
                value = eval(compile(expression, "<workpad>", "eval"), self.namespace)  # noqa: S307
        if value is not None:
            self.namespace["ans"] = value
        return printed.getvalue(), value

    # ------------------------------------------------------------------
    # The project's tables
    # ------------------------------------------------------------------

    def _tools(self) -> AssistantTools:
        return AssistantTools(self._repo(), confirm=lambda question: True, changed=self._changed,
                              current_figure=lambda: None)

    def tables(self) -> list[str]:
        """The project's tables."""
        return [str(name) for name in self._repo().list_user_tables()["Table"]]

    def columns(self, table: str) -> list[str]:
        """*table*'s columns."""
        table = self._tools()._table(table)
        return [str(name) for name in self._repo().get_columns(table) if name not in ("Hide", "Selected")]

    def get(self, table: str, column: str | None = None) -> Any:
        """*column* of *table*: numbers as a numpy array, else a list. Without *column*, every column, as a dict."""
        tools = self._tools()
        table = tools._table(table)
        names = [column] if column is not None else self.columns(table)
        names = tools._columns(table, names)
        frame = self._repo().query_df(f"SELECT {', '.join(_quote(name) for name in names)} FROM {_quote(table)}")
        result = {name: _column_values(frame[name]) for name in names}
        return result[names[0]] if column is not None else result

    def put(self, table: str, data: Mapping[str, Any]) -> str:
        """A new table *table* from *data* (column -> values); the name made unique. Its name back."""
        repo = self._repo()
        columns = {str(name): _values(values) for name, values in data.items()}
        lengths = {len(values) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"The columns have different lengths: {sorted(lengths)}.")
        name = _unique(table, self.tables())
        repo.snapshot_for_undo([], label=f"Workpad: table '{name}'")
        repo.import_dataframe(pd.DataFrame(columns), table_name=name, normalize_columns=False)
        self._changed([])
        return name

    def plot(self, chart_type: str, *, table: str = "", title: str = "", **roles: Any) -> str:
        """A chart of *chart_type*: each role (x=, y=, group=, color=...) a column's values or, with *table*,
        a column's name. y may be several (a list of names, or a dict name -> values)."""
        if "y" not in roles and not roles:
            raise ValueError("plot needs at least one role, e.g. plot('Scatter', x=xs, y=ys).")
        if not table:
            data: dict[str, Any] = {}
            names: dict[str, Any] = {}
            for role, values in roles.items():
                if role == "y" and isinstance(values, Mapping):
                    names["y"] = [str(key) for key in values]
                    data.update({str(key): column for key, column in values.items()})
                else:
                    data[role] = values
                    names[role] = role
            table = self.put(f"Workpad {chart_type}", data)
            roles = names
        y = roles.pop("y", None)
        y_columns = [y] if isinstance(y, str) else list(y or [])
        if not y_columns:
            # A chart without y (a histogram): its first role is the one drawn.
            first = next(iter(roles))
            y_columns = [roles.pop(first)]
        x = roles.pop("x", "")
        roles_text = ", ".join(f"{role}={name}" for role, name in roles.items())
        tools = self._tools()
        return tools._tool_create_chart(table, chart_type, y_columns, x=str(x or ""), title=title,
                                        roles_text=roles_text)


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _column_values(values: pd.Series) -> Any:
    numbers = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() and numbers.notna().sum() == values.notna().sum():
        return numbers.to_numpy(dtype=float)
    return values.tolist()


def _unique(name: str, taken: Iterable[str]) -> str:
    taken_lower = {item.lower() for item in taken}
    candidate, number = name, 2
    while candidate.lower() in taken_lower:
        candidate, number = f"{name} {number}", number + 1
    return candidate


__all__ = ["UnsafeCommand", "Workpad", "check"]
