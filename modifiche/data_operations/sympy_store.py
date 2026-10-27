"""Stored SymPy/Python functions shared by Function and Fit dialogs."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import sympy as sp


def _scientific_import(name: str, globals: Any = None, locals: Any = None, fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    """Allow imports only from NumPy and SymPy inside stored Code models."""
    root = str(name).split(".", 1)[0]
    if root not in {"numpy", "sympy"}:
        raise ImportError("Stored function code may import only numpy and sympy.")
    return __import__(name, globals, locals, fromlist, level)

TABLE = "__sympy__"
IMPLICIT = ("x", "y", "z")


def ensure_table(repo: Any) -> None:
    repo.create_sympy_table()


def rows(repo: Any) -> list[dict[str, str]]:
    return list(repo.list_sympy_functions())


def get(repo: Any, name: str) -> dict[str, str] | None:
    return repo.get_sympy_function(name)


def save(repo: Any, data: Mapping[str, str]) -> None:
    repo.save_sympy_function(
        str(data["function_name"]),
        str(data["source_type"]),
        str(data.get("symbols", "")),
        str(data["code_string"]),
        float(data.get("from", "0")), float(data.get("to", "10")),
        int(data.get("points", "200")), str(data.get("description", "")),
    )


def parameter_names(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    # Accept old rows that explicitly stored x/y/z, but never expose them as fit parameters.
    return [name for name in names if name not in IMPLICIT]


def _code_info(code: str, wanted: str) -> tuple[ast.FunctionDef, list[str], int]:
    tree = ast.parse(code, mode="exec")
    node = next((item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == wanted), None)
    if node is None:
        raise ValueError(f"Code must define def {wanted}(...).")
    params = [arg.arg for arg in node.args.args if arg.arg not in IMPLICIT]
    names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    ndim = 2 if "y" in names else 1
    return node, params, ndim


def payload(data: Mapping[str, str]) -> dict[str, Any]:
    name = str(data["function_name"])
    kind = str(data["source_type"]).lower()
    code = str(data["code_string"])
    params = parameter_names(str(data.get("symbols", "")))
    ndim = 1
    if kind == "expression":
        symbols = {coord: sp.Symbol(coord) for coord in IMPLICIT}
        symbols.update({param: sp.Symbol(param) for param in params})
        expr = sp.sympify(code, locals=symbols)
        ndim = 2 if symbols["y"] in expr.free_symbols else 1
    elif kind == "code":
        _node, signature_params, ndim = _code_info(code, name)
        if not params:
            params = signature_params
    else:
        raise ValueError(f"Unsupported stored source type: {kind}")
    return {
        "name": name, "description": str(data.get("description", "")) or "Stored SymPy function", "expression": code,
        "params": params, "p0": [1.0] * len(params), "ndim": ndim,
        "_sympy_stored": True, "source_type": kind, "symbols": ",".join(params), "code_string": code,
        "keywords": f"stored sympy {name}",
    }


def catalog(repo: Any) -> list[dict[str, Any]]:
    return [payload(row) for row in rows(repo)]


def make_model(data: Mapping[str, Any]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    name = str(data["name"])
    kind = str(data["source_type"])
    params = parameter_names(str(data.get("symbols", "")))
    code = str(data["code_string"])
    ndim = int(data.get("ndim", 1))

    if kind == "expression":
        x, y, z = sp.symbols("x y z")
        p_symbols = [sp.Symbol(param) for param in params]
        local = {"x": x, "y": y, "z": z, **dict(zip(params, p_symbols))}
        expr = sp.sympify(code, locals=local)
        args = (x, y, *p_symbols) if ndim == 2 else (x, *p_symbols)
        numeric = sp.lambdify(args, expr, modules="numpy")

        def model(independent: np.ndarray, theta: np.ndarray) -> np.ndarray:
            p = np.asarray(theta, dtype=float)
            arr = np.asarray(independent, dtype=float)
            value = numeric(arr[:, 0], arr[:, 1], *p) if ndim == 2 else numeric(arr, *p)
            return np.asarray(value, dtype=float)
        return model

    _node, signature_params, _ndim = _code_info(code, name)
    if not params:
        params = signature_params
    env: dict[str, Any] = {"np": np, "numpy": np, "sp": sp, "sympy": sp, "__builtins__": {"__import__": _scientific_import}}
    exec(compile(code, "<stored-sympy>", "exec"), env, env)
    func = env[name]

    def model(independent: np.ndarray, theta: np.ndarray) -> np.ndarray:
        arr = np.asarray(independent, dtype=float)
        if ndim == 2:
            env["x"], env["y"] = arr[:, 0], arr[:, 1]
        else:
            env["x"] = arr
        p = np.asarray(theta, dtype=float)
        return np.asarray(func(*p), dtype=float)
    return model
