"""Stored functions as SQL: a chart of a function, evaluated by SQLite at each read.

Two SQL functions are registered on every project connection (register):

    chartlibre_grid(n, from, to, points, spacing)      the n-th point of a range, linear or log
    chartlibre_function(name, parameters, x [, y])     the stored function there, its parameters as JSON

so a function source is a query - a recursive CTE counting the grid's points, the function at each one
(function_sql) - and nothing is stored but the formula and its numbers. Save the formula again and every
chart of it draws the new one at its next read; each chart keeps its own parameters and range.

The formula is read from __sympy__ at each call - a primary-key read, cheap - and its compiled model kept by
fingerprint, so a saved change is seen at once. A function this computer does not trust (todo ID47) is NULL
everywhere: nothing of it runs.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.logs.logger import applogger

FUNCTION = "chartlibre_function"
GRID = "chartlibre_grid"
LINEAR, LOG = "linear", "log"


@dataclass(frozen=True, slots=True)
class GridRange:
    """Where a function is evaluated: *points* values from *start* to *stop*, evenly or logarithmically."""

    start: float = 0.0
    stop: float = 10.0
    points: int = 200
    spacing: str = LINEAR


def grid_point(n: Any, start: Any, stop: Any, points: Any, spacing: Any) -> float | None:
    """The n-th of *points* values from *start* to *stop* - numpy's linspace / logspace, one value at a time.

    In Python, not SQL: SQLite's power() and log10() are not always built in.
    """
    try:
        index, first, last, count = int(n), float(start), float(stop), max(2, int(points))
    except (TypeError, ValueError):
        return None
    fraction = index / (count - 1)
    if str(spacing) == LOG:
        if first <= 0.0 or last <= 0.0:
            return None
        low, high = math.log10(first), math.log10(last)
        return 10.0 ** (low + (high - low) * fraction)
    if index == count - 1:
        return last
    return first + (last - first) * fraction


def _literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _number(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"A range needs finite numbers, not {value!r}.")
    return repr(number)


def function_sql(name: str, parameters: Mapping[str, float], x: GridRange, y: GridRange | None = None) -> str:
    """The query of a function source: the grid and the function at each point - x, y; or x, y, z for f(x, y).

    For f(x, y) the rows go x fastest, y after (numpy's meshgrid "xy", ravelled).
    """
    values = json.dumps({str(key): float(value) for key, value in parameters.items()}, sort_keys=True)
    call_head = f"{FUNCTION}({_literal(str(name))}, {_literal(values)}"

    def counter(alias: str, grid: GridRange) -> str:
        last = max(2, int(grid.points)) - 1
        return f'"{alias}"(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM "{alias}" WHERE n < {last})'

    def point(alias: str, grid: GridRange) -> str:
        return (f'{GRID}("{alias}".n, {_number(grid.start)}, {_number(grid.stop)}, {max(2, int(grid.points))}, '
                f"{_literal(LOG if grid.spacing == LOG else LINEAR)})")

    if y is None:
        return (f'WITH RECURSIVE {counter("_i", x)} '
                f'SELECT "x", {call_head}, "x") AS "y" '
                f'FROM (SELECT {point("_i", x)} AS "x" FROM "_i")')
    return (f'WITH RECURSIVE {counter("_i", x)}, {counter("_j", y)} '
            f'SELECT "x", "y", {call_head}, "x", "y") AS "z" '
            f'FROM (SELECT {point("_i", x)} AS "x", {point("_j", y)} AS "y" FROM "_j" CROSS JOIN "_i")')


_CALL = re.compile(FUNCTION + r"\(\s*'((?:[^']|'')*)'")


def function_names(queries: Iterable[str]) -> set[str]:
    """The stored functions these queries draw - read from their text, nothing run."""
    return {match.replace("''", "'") for query in queries for match in _CALL.findall(str(query or ""))}


@lru_cache(maxsize=64)
def _parameters(text: str) -> dict[str, float]:
    try:
        decoded = json.loads(text or "{}")
        if not isinstance(decoded, dict):
            return {}
        return {str(key): float(value) for key, value in decoded.items()}
    except (TypeError, ValueError):
        return {}


def register(con: sqlite3.Connection) -> None:
    """chartlibre_function and chartlibre_grid, on *con*."""
    #: (fingerprint, parameter names) -> (model, parameter names, ndim): compiled once, while trusted.
    models: dict[tuple[str, str], tuple[Any, list[str], int]] = {}
    #: The functions whose failure was logged: an error at every point of a grid is one line in the log.
    failed: set[str] = set()

    def value(name: Any, parameters: Any, x: Any, y: Any = None) -> float | None:
        try:
            return _value(con, models, str(name), str(parameters or ""), x, y)
        except Exception:  # noqa: BLE001 - a user's formula: an error at a point is no value there
            if str(name) not in failed:
                failed.add(str(name))
                applogger.exception("The stored function '%s' could not be evaluated", name)
            return None

    con.create_function(FUNCTION, 3, lambda name, parameters, x: value(name, parameters, x))
    con.create_function(FUNCTION, 4, value)
    con.create_function(GRID, 5, grid_point, deterministic=True)


def _value(con: sqlite3.Connection, models: dict[tuple[str, str], tuple[Any, list[str], int]],
           name: str, parameters: str, x: Any, y: Any) -> float | None:
    if x is None:
        return None
    row = con.execute('SELECT function_name, source_type, symbols, code_string, "Values" FROM "__sympy__" '
                      "WHERE function_name = ?", (name,)).fetchone()
    if row is None:
        return None
    # Here, not at the top: SymPy is loaded when a function is first drawn, not when a project opens.
    import numpy as np

    from app.data_operations import sympy_store

    data = {"function_name": row[0], "source_type": row[1], "symbols": row[2], "code_string": row[3]}
    key = (sympy_store.fingerprint(data), str(data["symbols"]))
    entry = models.get(key)
    if entry is None:
        if not sympy_store.is_trusted(data):
            return None
        payload = sympy_store.payload(data)
        entry = (sympy_store.make_model(payload), list(payload["params"]), int(payload["ndim"]))
        models[key] = entry
    model, names, ndim = entry
    # A parameter the query does not give: its stored value, read each time as the formula is.
    given, stored = _parameters(parameters), _parameters(str(row[4] or ""))
    theta = np.array([given.get(param, stored.get(param, 1.0)) for param in names], dtype=float)
    if ndim == 2:
        if y is None:
            return None
        independent = np.array([[float(x), float(y)]], dtype=float)
    else:
        independent = np.array([float(x)], dtype=float)
    with np.errstate(all="ignore"):  # outside its domain a function has no value there: NULL, not a warning
        result = np.asarray(model(independent, theta), dtype=float).reshape(-1)
    number = float(result[0]) if result.size else math.nan
    return number if math.isfinite(number) else None
