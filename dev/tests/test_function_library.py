"""The function library as SymPy (app.functions.library): each entry gives its class's numbers, guards included.

Step 1 of dev/plans/sympy_library_plan.txt: nothing reads the library yet; this is what makes the switch safe.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from app.functions import library
from dev.tests.legacy_functions import basic, nist, surfaces


def _classes() -> dict[str, type]:
    found = {}
    for module in (basic, surfaces, nist):
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and "execute" in cls.__dict__ and getattr(cls, "name", ""):
                found[name] = cls
    return found


CLASSES = _classes()


def test_every_class_is_in_the_library_once() -> None:
    assert sorted(entry["class_name"] for entry in library.ENTRIES) == sorted(CLASSES)
    assert [entry["kind"] for entry in library.ENTRIES].count("code") == 1  # Voigt: wofz has no SymPy form
    for entry in library.ENTRIES:
        assert len(entry["params"]) == len(entry["values"]) == len(entry["labels"])
        assert all(name.isidentifier() for name in entry["params"])


def _parameter_sets(values: list[float], rng: np.random.Generator) -> list[np.ndarray]:
    """The starting values, three draws around them, and one with a parameter at 0 (the guards' case)."""
    base = np.asarray(values, dtype=float)
    scale = np.where(np.abs(base) > 0, np.abs(base), 1.0)
    draws = [base + rng.normal(0.0, 0.2, base.size) * scale for _ in range(3)]
    zeroed = base.copy()
    zeroed[min(1, base.size - 1)] = 0.0
    return [base, *draws, zeroed]


@pytest.mark.parametrize("entry", library.ENTRIES, ids=lambda entry: entry["class_name"])
def test_the_expression_gives_the_classs_numbers(entry: dict) -> None:
    rng = np.random.default_rng(abs(hash(entry["class_name"])) % (2 ** 32))
    old = CLASSES[entry["class_name"]].execute
    new = library.numeric(entry)
    x = np.concatenate([np.linspace(-3.0, 10.0, 41), rng.uniform(0.1, 20.0, 20)])
    y = rng.uniform(-5.0, 5.0, x.size)
    compared = 0
    for p in _parameter_sets(entry["values"], rng):
        with np.errstate(all="ignore"):
            expected = np.asarray(old(x, p) if entry["ndim"] == 1 else old(x, y, p), dtype=float)
            got = np.asarray(new(x, p) if entry["ndim"] == 1 else new(x, y, p), dtype=float)
        expected = np.broadcast_to(expected, x.shape)
        finite = np.isfinite(expected) & (np.abs(expected) < 1e12)
        np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-7, atol=1e-9,
                                   err_msg=f"{entry['name']} with {p}")
        compared += int(finite.sum())
    assert compared > 0


def test_the_guards_keep_a_width_off_zero_as_the_classes_do() -> None:
    gaussian = library.get("Gaussian peak")
    assert gaussian is not None and "pos(" in gaussian["code"]
    values = library.numeric(gaussian)(np.array([0.0, 1.0]), np.zeros(len(gaussian["params"])))
    assert np.all(np.isfinite(values) | np.isnan(values))
    sp = library.sp
    formula = library.expression(gaussian)
    assert formula.has(library.pos)
    # Differentiated through the guard: d/dx stays a formula, no unevaluated Derivative left.
    assert not sp.diff(formula, sp.Symbol("x")).has(sp.Derivative)


@pytest.mark.parametrize("entry", [entry for entry in library.ENTRIES if entry["guess"]],
                         ids=lambda entry: entry["class_name"])
def test_the_estimator_gives_the_classs_starting_values(entry: dict) -> None:
    rng = np.random.default_rng(7)
    old = CLASSES[entry["class_name"]]
    new = library.function_class(entry["name"])
    x = np.linspace(0.5, 10.0, 60)
    p = np.asarray(entry["values"], dtype=float) * (1.0 + 0.1 * rng.standard_normal(len(entry["values"])))
    if entry["ndim"] == 1:
        y = old.execute(x, p) + 0.01 * rng.standard_normal(x.size)
        expected, got = old.initial_guess(x, y), new.initial_guess(x, y)
    else:
        y = rng.uniform(0.5, 10.0, x.size)
        z = old.execute(x, y, p) + 0.01 * rng.standard_normal(x.size)
        expected, got = old.initial_guess(x, y, z), new.initial_guess(x, y, z)
    assert (expected is None) == (got is None)
    if expected is not None:
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_the_fit_catalogue_offers_every_library_function_under_its_old_name() -> None:
    from app.scanners.functions_scanner import FunctionScanner, SurfaceFunctionScanner

    offered = {model["function_class"]: model for scanner in (FunctionScanner(extra_roots=()), SurfaceFunctionScanner(extra_roots=()))
               for models in scanner.catalog().values() for model in models}
    assert set(CLASSES) <= set(offered)
    gaussian = offered["gaussian_peak"]
    assert gaussian["params"] == list(CLASSES["gaussian_peak"].params)
    model = FunctionScanner().make_model(gaussian)
    np.testing.assert_allclose(model(np.array([4.0]), np.array(gaussian["p0"])),
                               CLASSES["gaussian_peak"].execute(np.array([4.0]), np.array(gaussian["p0"])))
    assert callable(getattr(FunctionScanner().load_class(gaussian), "initial_guess", None))
