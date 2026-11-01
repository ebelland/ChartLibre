"""Calculus on a stored function's series: exact - SymPy's derivative and integral of its formula, not of its
points - and the result a stored function of the same parameters, drawn live."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import symbolic_calculus as symbolic
from app.data.repo.function_sql import GridRange, function_sql
from app.data.sqlite_repo import SqliteRepo
from app.data_operations import sympy_store
from app.data_operations.calculus_dialog import (
    DERIV_GRADIENT,
    INTEGRAL_CUMULATIVE,
    INTEGRAL_DEFINITE,
    SeriesCalculusDialog,
)


def test_the_formulas() -> None:
    assert symbolic.derivative("a*x**3 + b", ["a", "b"]) == "3*a*x**2"
    assert symbolic.derivative("sin(k*x)", ["k"], order=2) == "-k**2*sin(k*x)"
    assert symbolic.antiderivative("2*a*x", ["a"], start=1.0) == "a*(x**2 - 1.0)"
    total, _exact = symbolic.definite("a*x**2", ["a"], {"a": 3.0}, 0.0, 2.0)
    assert total == pytest.approx(8.0)
    with pytest.raises(ValueError, match="closed-form"):
        symbolic.antiderivative("exp(-x**2)*log(x + sin(x))", [], start=0.0)
    np.testing.assert_allclose(symbolic.evaluate("2*a", ["a"], {"a": 1.5}, np.arange(3.0)), [3.0, 3.0, 3.0])


def _function_figure(repo: SqliteRepo, code: str, symbols: str, values: dict[str, float]) -> int:
    sympy_store.save(repo, {"function_name": "Curve", "source_type": "expression", "symbols": symbols,
                            "code_string": code})
    figure_id = int(repo.create_figure_descriptor(name="F"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="Curve", x_label="x", y_label="y", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Curve", style={}, roles={"x": "x", "y": "y"},
                                  sql_query=function_sql("Curve", values, GridRange(0.0, 2.0, 21)))
    return figure_id


def _dialog(repo: SqliteRepo, figure_id: int, model: str) -> SeriesCalculusDialog:
    dialog = SeriesCalculusDialog(repo=repo, figure_id=figure_id, parent=None)
    dialog.source_chart_radio.setChecked(True)
    dialog.series_selector.select_all_series()
    dialog.model_combo.setCurrentText(model)
    return dialog


def test_the_derivative_of_a_function_series_is_its_formulas(qapp, repo: SqliteRepo) -> None:
    """d/dx (a*x**3) = 3*a*x**2 exactly - a finite difference would be off at every point - and OK keeps it
    as the stored function Curve_d1, drawn live; Undo takes it away."""
    figure_id = _function_figure(repo, "a*x**3", "a", {"a": 2.0})
    dialog = _dialog(repo, figure_id, DERIV_GRADIENT)
    try:
        (result,) = dialog.compute_results()
        np.testing.assert_allclose(result.y, 6.0 * result.x ** 2, rtol=1e-12)
        assert result.metadata["symbolic"] == "3*a*x**2"
        dialog.destination_combo.setCurrentIndex(dialog.destination_combo.findData(dialog.DEST_NEW_AXIS))
        assert dialog.apply()
    finally:
        dialog.close()
    stored = sympy_store.get(repo, "Curve_d1")
    assert stored is not None and stored["code_string"] == "3*a*x**2"
    drawn = repo.load_figure_descriptor(figure_id).axes[-1].series[0]
    rows = repo.query_df(drawn.sql_query)
    np.testing.assert_allclose(rows["y"], 6.0 * rows["x"] ** 2, rtol=1e-12)
    repo.undo_last()
    assert sympy_store.get(repo, "Curve_d1") is None


def test_the_integrals_of_a_function_series_are_exact(qapp, repo: SqliteRepo) -> None:
    figure_id = _function_figure(repo, "a*x", "a", {"a": 3.0})
    dialog = _dialog(repo, figure_id, INTEGRAL_CUMULATIVE)
    try:
        (running,) = dialog.compute_results()
        np.testing.assert_allclose(running.y, 1.5 * running.x ** 2, atol=1e-12)  # from 0: 3 x²/2
        assert running.total == pytest.approx(6.0)
        dialog.model_combo.setCurrentText(INTEGRAL_DEFINITE)
        (area,) = dialog.compute_results()
        assert area.total == pytest.approx(6.0, abs=1e-12)
    finally:
        dialog.close()


def test_off_or_with_a_baseline_it_is_computed_from_the_points(qapp, repo: SqliteRepo) -> None:
    figure_id = _function_figure(repo, "a*x**3", "a", {"a": 2.0})
    dialog = _dialog(repo, figure_id, DERIV_GRADIENT)
    try:
        dialog._parameter_form_spec.set_values({"symbolic": False})
        (result,) = dialog.compute_results()
        assert "symbolic" not in result.metadata and result.function is None
        assert not np.allclose(result.y, 6.0 * result.x ** 2, rtol=1e-12)  # a finite difference
    finally:
        dialog.close()


def test_cancel_after_preview_takes_the_derived_function_away(qapp, repo: SqliteRepo) -> None:
    figure_id = _function_figure(repo, "a*x**3", "a", {"a": 2.0})
    dialog = _dialog(repo, figure_id, DERIV_GRADIENT)
    try:
        assert dialog.preview()
        for _step in range(50):
            qapp.processEvents()
        assert sympy_store.get(repo, "Curve_d1") is not None
        dialog.discard_unapplied()
        assert sympy_store.get(repo, "Curve_d1") is None
    finally:
        dialog.close()


def test_has_it_changed_and_is_it_normal_on_a_function_series(qapp, repo: SqliteRepo) -> None:
    """A function's query reads its own grid ("_i"): no table of the project, so no What next? step on one."""
    from app.data_operations.change_dialog import ChangeDialog
    from app.data_operations.normality_dialog import NormalityDialog, source_table

    figure_id = _function_figure(repo, "a*x**3", "a", {"a": 2.0})
    query = repo.load_figure_descriptor(figure_id).axes[0].series[0].sql_query
    assert source_table(query) == "" and source_table('SELECT a AS x FROM "t"') == "t"
    for cls in (ChangeDialog, NormalityDialog):
        dialog = cls(repo=repo, figure_id=figure_id, parent=None)
        try:
            dialog.source_chart_radio.setChecked(True)
            dialog.series_selector.select_all_series()
            assert dialog.preview(), cls.__name__
            html = dialog.format_results(dialog.compute_results())
            assert "no such table" not in html
        finally:
            dialog.close()
