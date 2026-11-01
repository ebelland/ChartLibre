"""Starting values for any SymPy function (app.analysis.initial_guess) - step 2 of the library plan."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import initial_guess as ig
from app.data.sqlite_repo import SqliteRepo

X = np.linspace(0.0, 20.0, 120)


@pytest.mark.parametrize(("code", "params", "stored", "truth"), [
    ("A*exp(-(x - x0)**2/(2*s**2)) + c", ["A", "x0", "s", "c"], [1, 0, 1, 0], [5.0, 12.0, 1.5, 2.0]),
    ("L/(1 + exp(-k*(x - x0)))", ["L", "k", "x0"], [1, 1, 0], [80.0, 0.9, 7.0]),
    ("A*exp(-x/tau) + c", ["A", "tau", "c"], [1, 1, 0], [30.0, 4.0, 5.0]),
])
def test_shapes_it_has_never_seen_are_found(code: str, params: list[str], stored: list[float],
                                            truth: list[float]) -> None:
    from app.functions import library

    entry = {"name": "t", "kind": "expression", "code": code, "params": params, "ndim": 1}
    rng = np.random.default_rng(1)
    exact = library.numeric(entry)(X, truth)
    y = exact + rng.normal(0, 0.005 * np.ptp(exact), X.size)
    guess = ig.generic_guess(code, params, stored, X, y)
    np.testing.assert_allclose(guess, truth, rtol=0.05)


def test_a_function_linear_in_its_parameters_is_solved_exactly() -> None:
    y = 1.0 + 2.0 * X - 0.1 * X ** 2
    assert ig.generic_guess("a + b*x + c2*x**2", ["a", "b", "c2"], [0, 0, 0], X, y) == pytest.approx([1, 2, -0.1])


def test_the_names_say_the_roles() -> None:
    assert [ig.role(name) for name in ("A", "x0", "sigma", "offset", "k", "tau", "m", "b", "foo")] == [
        "amplitude", "centre", "width", "offset", "rate", "time", "slope", "intercept", ""]


def test_fit_starts_a_stored_function_from_the_data(qapp, repo: SqliteRepo) -> None:
    """Chosen in Fit, a stored function's Initial column is read off the data, not its stored 1, 0, 1, 0."""
    from app.data_operations import sympy_store
    from app.data_operations.fit_dialog import SeriesFitDialog

    y = 5.0 * np.exp(-(X - 12.0) ** 2 / (2 * 1.5 ** 2)) + 2.0
    repo.import_dataframe(pd.DataFrame({"x": X, "y": y}), table_name="peak", normalize_columns=False)
    sympy_store.save(repo, {"function_name": "Bump", "source_type": "expression", "symbols": "A, x0, s, c",
                            "code_string": "A*exp(-(x - x0)**2/(2*s**2)) + c",
                            "values": {"A": 1.0, "x0": 0.0, "s": 1.0, "c": 0.0}})
    figure_id = repo.create_figure_descriptor(name="Fit")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="peak", x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="peak", sql_query='SELECT "x", "y" FROM "peak"',
                                  roles={"x": "x", "y": "y"}, style={})
    dialog = SeriesFitDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.series_selector.reload(select_all_series=True)
        assert dialog.select_model("Bump")
        start = dialog._collect_params_from_table()[0]
        np.testing.assert_allclose(start, [5.0, 12.0, 1.5, 2.0], rtol=0.02)
    finally:
        dialog.close()
