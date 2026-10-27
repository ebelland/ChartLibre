"""Stored functions: a SymPy expression or a Python function kept in the project (__sympy__), drawn by Function and
offered to Fit curve under "Stored functions"."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data_operations import sympy_store


@pytest.fixture
def repo(tmp_path: Path):
    repo = SqliteRepo(db_path=tmp_path / "s.dhub")
    yield repo
    repo.close()


def _expression(name: str, code: str, symbols: str) -> dict[str, str]:
    return {"function_name": name, "source_type": "expression", "symbols": symbols, "code_string": code,
            "from": "0", "to": "5", "points": "50", "description": "a test"}


def test_a_function_is_kept_in_the_project_and_replaced_by_name(repo: SqliteRepo) -> None:
    sympy_store.save(repo, _expression("Line", "a*x + b", "a, b"))
    sympy_store.save(repo, _expression("Line", "a*x + b + c", "a, b, c"))
    assert [row["function_name"] for row in sympy_store.rows(repo)] == ["Line"]
    assert sympy_store.get(repo, "Line")["code_string"] == "a*x + b + c"
    assert repo.delete_sympy_function("Line") and sympy_store.get(repo, "Line") is None


def test_an_expression_is_a_model_of_x_or_of_x_and_y(repo: SqliteRepo) -> None:
    sympy_store.save(repo, _expression("Line", "a*x + b", "a, b, x"))  # x listed: never a parameter
    sympy_store.save(repo, _expression("Plane", "a*x + b*y", "a, b"))
    line, plane = sorted(sympy_store.catalog(repo), key=lambda item: item["name"])
    assert line["params"] == ["a", "b"] and line["ndim"] == 1 and line["_sympy_stored"]
    assert plane["ndim"] == 2
    np.testing.assert_allclose(sympy_store.make_model(line)(np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0])),
                               [1.0, 3.0, 5.0])
    np.testing.assert_allclose(sympy_store.make_model(plane)(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 10.0])),
                               [21.0, 43.0])


def test_code_is_a_function_of_its_parameters_with_x_given(repo: SqliteRepo) -> None:
    code = "def Square(a, b):\n    f_numeric = a * x**2 + b\n    return f_numeric\n"
    sympy_store.save(repo, {"function_name": "Square", "source_type": "code", "symbols": "", "code_string": code})
    payload = sympy_store.catalog(repo)[0]
    assert payload["params"] == ["a", "b"]
    np.testing.assert_allclose(sympy_store.make_model(payload)(np.array([1.0, 2.0]), np.array([3.0, 1.0])), [4.0, 13.0])
    bad = {"function_name": "Bad", "source_type": "code", "symbols": "",
           "code_string": "def Bad(a):\n    import os\n    f_numeric = a\n    return f_numeric\n"}
    with pytest.raises(ImportError):
        sympy_store.make_model(sympy_store.payload(bad))(np.array([1.0]), np.array([1.0]))


def test_fit_curve_offers_the_stored_functions_and_fits_one(qapp, repo: SqliteRepo) -> None:
    from app.data_operations.fit_dialog import SeriesFitDialog

    x = np.linspace(0.0, 4.0, 30)
    repo.import_dataframe(pd.DataFrame({"x": x, "y": 3.0 * x + 2.0}), table_name="points", normalize_columns=False)
    sympy_store.save(repo, _expression("MyLine", "a*x + b", "a, b"))
    figure_id = repo.create_figure_descriptor(name="Fit")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="points", x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="points",
                                  sql_query='SELECT "x", "y" FROM "points"', roles={"x": "x", "y": "y"}, style={})
    dialog = SeriesFitDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.series_selector.reload(select_all_series=True)
        assert [item["name"] for item in dialog._catalog_data()["Stored functions"]] == ["MyLine"]
        assert dialog.select_model("MyLine")
        dialog.evaluate(dialog._after_fit, background=False, optimise=True)
        np.testing.assert_allclose(dialog.compute_results()[0].params, [3.0, 2.0], atol=1e-6)
    finally:
        dialog.close()
