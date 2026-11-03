"""Derivatives and integrals of a stored function's expression, exact - SymPy's, not a finite difference.

A series drawn from a stored function (app.data_operations.sympy_store) has a formula, not only points: its
derivative is a formula too, d/dx (a*x**2) = 2*a*x, with no window, no noise and no edge effect. The
expression is read with the parameters as symbols, so the result is a function of the same parameters -
stored beside the first one, it draws for any of their values.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import sympy as sp

#: The project's functions by name, for the calls in an expression (set by the caller for one calculation).
Functions = Mapping[str, Mapping]


def _parse(expression: str, parameters: Sequence[str], functions: Functions | None = None,
           ) -> tuple[sp.Expr, sp.Symbol, list[sp.Symbol]]:
    from app.functions import library

    x = sp.Symbol("x")
    symbols = [sp.Symbol(name) for name in parameters]
    local = library.sympy_locals(list(parameters), code=str(expression), functions=functions)
    return sp.sympify(str(expression).replace("^", "**"), locals=local), x, symbols


def _exact(result: sp.Expr) -> sp.Expr:
    """*result*, when it is a formula; ValueError when part of it is not (a code function's derivative)."""
    if result.has(sp.Derivative) or any(isinstance(f, sp.core.function.AppliedUndef)
                                        for f in result.atoms(sp.Function)):
        raise ValueError("Part of the function is code, not a formula: no exact result.")
    return result


def derivative(expression: str, parameters: Sequence[str], order: int = 1, *,
               functions: Functions | None = None) -> str:
    """The *order*-th derivative of the expression in x, simplified, as text sympify reads back."""
    expr, x, _symbols = _parse(expression, parameters, functions)
    return str(_exact(sp.simplify(sp.diff(expr, x, int(order)))))


def antiderivative(expression: str, parameters: Sequence[str], start: float, *,
                   functions: Functions | None = None) -> str:
    """The integral from *start* to x, as text: F(x) - F(start). ValueError when SymPy finds no closed form."""
    expr, x, _symbols = _parse(expression, parameters, functions)
    primitive = _exact(sp.integrate(expr, x))
    if primitive.has(sp.Integral):
        raise ValueError(f"No closed-form integral of {expression}.")
    return str(sp.simplify(primitive - primitive.subs(x, sp.Float(start))))


def definite(expression: str, parameters: Sequence[str], values: Mapping[str, float], start: float,
             stop: float, *, functions: Functions | None = None) -> tuple[float, str]:
    """The area from *start* to *stop* for these parameter values: (the number, the integral's exact form)."""
    expr, x, symbols = _parse(expression, parameters, functions)
    exact = _exact(sp.integrate(expr, (x, sp.Float(start), sp.Float(stop))))
    if exact.has(sp.Integral):
        raise ValueError(f"No closed-form integral of {expression}.")
    number = exact.subs({symbol: float(values.get(str(symbol), 1.0)) for symbol in symbols})
    return float(sp.N(number)), str(sp.simplify(sp.integrate(expr, (x, sp.Symbol("a_"), sp.Symbol("b_")))))


def evaluate(expression: str, parameters: Sequence[str], values: Mapping[str, float], x: np.ndarray, *,
             functions: Functions | None = None) -> np.ndarray:
    """The expression at each x, for these parameter values."""
    from app.functions import library

    expr, x_symbol, symbols = _parse(expression, parameters, functions)
    numeric = sp.lambdify((x_symbol, *symbols), expr, modules=library.NUMPY_MODULES)
    result = numeric(np.asarray(x, dtype=float), *(float(values.get(str(s), 1.0)) for s in symbols))
    return np.broadcast_to(np.asarray(result, dtype=float), np.shape(x)).copy()
