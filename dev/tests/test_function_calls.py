"""A SymPy function calling others - the library's and the project's (step 3 of the library plan)."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from app.analysis import symbolic_calculus as symbolic
from app.data.repo.function_sql import GridRange, function_sql
from app.data.sqlite_repo import SqliteRepo
from app.data_operations import sympy_store
from app.functions import library


def _save(repo: SqliteRepo, name: str, code: str, symbols: str, values: dict | None = None) -> None:
    sympy_store.save(repo, {"function_name": name, "source_type": "expression", "symbols": symbols,
                            "code_string": code, "values": values or {}})


def test_a_peak_on_a_line_from_the_librarys_functions(repo: SqliteRepo) -> None:
    _save(repo, "PeakOnLine", "Gaussian_peak(x, A, mu, s, 0) + Linear(x, b, m)", "A, mu, s, b, m")
    payload = sympy_store.payload(sympy_store.get(repo, "PeakOnLine"))
    model = sympy_store.make_model(payload, sympy_store.functions_by_name(repo))
    x = np.linspace(0, 10, 11)
    expected = 3.0 * np.exp(-(x - 5.0) ** 2 / (2 * 0.8 ** 2)) + 1.0 + 0.5 * x
    np.testing.assert_allclose(model(x, np.array([3.0, 5.0, 0.8, 1.0, 0.5])), expected)


def test_a_project_function_calls_another_and_draws_as_a_series(repo: SqliteRepo) -> None:
    _save(repo, "Ramp", "a*x", "a", {"a": 2.0})
    _save(repo, "RampPlusOne", "Ramp(x, a) + 1", "a", {"a": 2.0})
    rows = repo.query_df(function_sql("RampPlusOne", {"a": 3.0}, GridRange(0.0, 2.0, 3)))
    assert rows["y"].tolist() == [1.0, 4.0, 7.0]


def test_a_function_that_calls_itself_is_refused(repo: SqliteRepo) -> None:
    _save(repo, "F", "G(x) + 1", "")
    with pytest.raises(library.CallError, match="G -> F"):
        _save(repo, "G", "F(x) * 2", "")
    with pytest.raises(library.CallError, match="itself"):
        _save(repo, "H", "H(x)", "")


def test_calculus_is_exact_through_calls_and_says_so_when_code_is_inside(repo: SqliteRepo) -> None:
    _save(repo, "Ramp", "a*x**2", "a")
    functions = sympy_store.functions_by_name(repo)
    assert symbolic.derivative("Ramp(x, a) + Linear(x, b, m)", ["a", "b", "m"], functions=functions) == "2*a*x + m"
    with pytest.raises(ValueError, match="code"):
        symbolic.derivative("Voigt_peak(x, A, x0, s, g, 0)", ["A", "x0", "s", "g"], functions=functions)


def test_a_library_function_reads_back_through_its_identifier() -> None:
    assert library.identifier("Logistic 4P") == "Logistic_4P"
    expression = sp.sympify("Logistic_4P(x, 0, 1, 2, 3)", locals=library.sympy_locals([], code="Logistic_4P(x, 0, 1, 2, 3)"))
    assert not expression.atoms(sp.core.function.AppliedUndef)
