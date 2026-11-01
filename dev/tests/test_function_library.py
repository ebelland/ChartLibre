"""The function library as SymPy (app.functions.library): each entry gives its class's numbers, guards included.

Step 1 of dev/plans/sympy_library_plan.txt: nothing reads the library yet; this is what makes the switch safe.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import app.functions.functions as basic
import app.functions.nist_functions as nist
import app.functions.surface_functions as surfaces
from app.functions import library


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
