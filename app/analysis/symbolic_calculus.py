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


def _parse(expression: str, parameters: Sequence[str]) -> tuple[sp.Expr, sp.Symbol, list[sp.Symbol]]:
    x = sp.Symbol("x")
    symbols = [sp.Symbol(name) for name in parameters]
    local = {"x": x, **dict(zip(parameters, symbols, strict=True))}
    return sp.sympify(str(expression).replace("^", "**"), locals=local), x, symbols


def derivative(expression: str, parameters: Sequence[str], order: int = 1) -> str:
    """The *order*-th derivative of the expression in x, simplified, as text sympify reads back."""
    expr, x, _symbols = _parse(expression, parameters)
    return str(sp.simplify(sp.diff(expr, x, int(order))))


def antiderivative(expression: str, parameters: Sequence[str], start: float) -> str:
    """The integral from *start* to x, as text: F(x) - F(start). ValueError when SymPy finds no closed form."""
    expr, x, _symbols = _parse(expression, parameters)
    primitive = sp.integrate(expr, x)
    if primitive.has(sp.Integral):
        raise ValueError(f"No closed-form integral of {expression}.")
    return str(sp.simplify(primitive - primitive.subs(x, sp.Float(start))))


def definite(expression: str, parameters: Sequence[str], values: Mapping[str, float], start: float,
             stop: float) -> tuple[float, str]:
    """The area from *start* to *stop* for these parameter values: (the number, the integral's exact form)."""
    expr, x, symbols = _parse(expression, parameters)
    exact = sp.integrate(expr, (x, sp.Float(start), sp.Float(stop)))
    if exact.has(sp.Integral):
        raise ValueError(f"No closed-form integral of {expression}.")
    number = exact.subs({symbol: float(values.get(str(symbol), 1.0)) for symbol in symbols})
    return float(sp.N(number)), str(sp.simplify(sp.integrate(expr, (x, sp.Symbol("a_"), sp.Symbol("b_")))))


def evaluate(expression: str, parameters: Sequence[str], values: Mapping[str, float], x: np.ndarray) -> np.ndarray:
    """The expression at each x, for these parameter values."""
    expr, x_symbol, symbols = _parse(expression, parameters)
    numeric = sp.lambdify((x_symbol, *symbols), expr, modules="numpy")
    result = numeric(np.asarray(x, dtype=float), *(float(values.get(str(s), 1.0)) for s in symbols))
    return np.broadcast_to(np.asarray(result, dtype=float), np.shape(x)).copy()
