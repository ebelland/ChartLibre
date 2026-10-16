"""Global fit: several series, some parameters shared (R-05)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from app.analysis.fit import GlobalLayout, fit_curve, fit_separately, global_model
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.data_operations.fit_dialog import SeriesFitDialog

X = np.linspace(0.0, 6.0, 40)
RATES = (0.5, 1.5)
RNG = np.random.default_rng(7)


def _decay(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * np.exp(-p[1] * x) + p[2]


def test_the_layout_places_shared_parameters_once() -> None:
    layout = GlobalLayout((True, False, True), 2)
    assert layout.index.tolist() == [[0, 1, 3], [0, 2, 3]]
    assert layout.size == 4
    assert layout.expand(np.array([[1, 2, 3], [9, 5, 9]])).tolist() == [1, 2, 5, 3]


def test_a_global_fit_recovers_shared_and_own_parameters() -> None:
    layout = GlobalLayout((True, False, True), 2)
    x = np.concatenate([np.column_stack([X, np.full(X.size, n)]) for n in range(2)])
    y = np.concatenate([_decay(X, np.array([4.0, rate, 1.0])) for rate in RATES])
    outcome = fit_curve(global_model(_decay, layout), x, y, np.array([3.0, 1.0, 1.0, 0.0]))
    np.testing.assert_allclose(outcome.params, [4.0, 0.5, 1.5, 1.0], rtol=1e-6)


def test_fitting_separately_gives_each_series_its_own_fit() -> None:
    layout = GlobalLayout((False, False, False), 2)
    x = np.concatenate([np.column_stack([X, np.full(X.size, n)]) for n in range(2)])
    noise = (0.001, 0.2)  # very different scatter: pooled errors would blur it
    y = np.concatenate([
        _decay(X, np.array([4.0, rate, 1.0])) + np.random.default_rng(3).normal(size=X.size) * sd
        for rate, sd in zip(RATES, noise)
    ])
    start = np.array([3.0, 3.0, 0.8, 1.0, 1.0, 0.0])
    combined, parts = fit_separately(
        _decay, layout, x, y, start, np.full(6, -np.inf), np.full(6, np.inf), np.zeros(6, dtype=bool),
    )
    assert len(parts) == 2 and combined.success
    assert [part.dof for part in parts] == [X.size - 3] * 2  # not the pooled 80 - 6
    assert combined.dof == 2 * (X.size - 3)
    np.testing.assert_allclose(combined.params[[0, 1]], [parts[0].params[0], parts[1].params[0]])
    np.testing.assert_allclose(combined.params[[2, 3]], [parts[0].params[1], parts[1].params[1]])
    np.testing.assert_allclose(combined.params[2], RATES[0], rtol=0.01)
    np.testing.assert_allclose(combined.params[3], RATES[1], rtol=0.25)  # the noisy one
    assert parts[1].std[1] > 20 * parts[0].std[1]  # each series' errors follow its own scatter
    with pytest.raises(ValueError):
        fit_separately(_decay, GlobalLayout((True, False, False), 2), x, y, start[:5],
                       np.full(5, -np.inf), np.full(5, np.inf), np.zeros(5, dtype=bool))


@pytest.fixture
def dialog(qapp, tmp_db_path: Path) -> Iterator[SeriesFitDialog]:
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)
    repo = SqliteRepo(db_path=tmp_db_path)
    frame = pd.DataFrame({"x": X, **{
        f"y{n}": 4.0 * np.exp(-rate * X) + 1.0 + RNG.normal(size=X.size) * 0.01 for n, rate in enumerate(RATES)
    }})
    repo.import_dataframe(frame, table_name="decays", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="Fit")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="decays", x_label="x", y_label="y", options={},
    )
    for n in range(2):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=n, name=f"s{n}",
            sql_query=f'SELECT "x", "y{n}" FROM "decays"', roles={"x": "x", "y": f"y{n}"}, style={},
        )
    built = SeriesFitDialog(repo=repo, figure_id=figure_id, parent=None)
    built.series_selector.reload(select_all_series=True)
    yield built
    built.close()
    repo.close()
    applogger.set_status_bar(None)


def _share(dialog: SeriesFitDialog, label: str) -> None:
    action = next(action for action in dialog._shared_menu.actions() if action.text() == label)
    action.setChecked(True)  # what clicking it in the Shared menu does


def test_the_dialog_fits_two_series_with_a_shared_amplitude(dialog: SeriesFitDialog) -> None:
    assert dialog.select_model("Exponential decay")
    table = dialog._params_table
    assert not dialog._shared_row.isHidden()
    assert table.rowCount() == 6  # three parameters, two series, none shared yet
    _share(dialog, "A")
    _share(dialog, "offset")
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["A", "rate k · s0", "rate k · s1", "offset"]

    dialog.on_estimate_initial_values()
    dialog.evaluate(dialog._after_fit, background=False, optimise=True)
    results = dialog.compute_results()  # what Preview and OK draw: the table, per series
    assert [result.source_table for result in results] == ["s0", "s1"]
    for result, rate in zip(results, RATES):
        np.testing.assert_allclose(result.params, [4.0, rate, 1.0], rtol=1e-2)
    assert results[0].params[0] == results[1].params[0]  # shared: one value
    assert results[0].output_table != results[1].output_table
    assert "Global fit of 2 series" in results[0].message
    assert dialog._shared_button.text() == "A, offset"


def test_one_series_checked_is_an_ordinary_fit(dialog: SeriesFitDialog) -> None:
    assert dialog.select_model("Exponential decay")
    assert dialog._params_table.rowCount() == 6
    dialog.series_selector.series_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert len(dialog.selected_series()) == 1
    assert dialog._shared_row.isHidden()
    assert dialog._params_table.rowCount() == 3


def test_each_curve_separately_fits_every_series_alone(dialog: SeriesFitDialog) -> None:
    assert dialog.select_model("Exponential decay")
    _share(dialog, "A")
    dialog._fit_mode_combo.setCurrentIndex(1)  # Each curve separately
    assert not dialog._shared_button.isEnabled()
    table = dialog._params_table
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels[:2] == ["A · s0", "A · s1"]  # nothing is shared, whatever was picked

    dialog.on_estimate_initial_values()
    dialog.evaluate(dialog._after_fit, background=False, optimise=True)
    results = dialog.compute_results()
    assert [result.source_table for result in results] == ["s0", "s1"]
    for result, rate in zip(results, RATES):
        np.testing.assert_allclose(result.params, [4.0, rate, 1.0], rtol=1e-2)
    assert "separately" in results[0].message

    dialog._fit_mode_combo.setCurrentIndex(0)  # back to global: the shared choice is still there
    assert dialog._shared_button.isEnabled()
    assert dialog._shared_button.text() == "A"


# ----------------------------------------------------------------------
# The formula with the fitted values in it (the report's and the result's expression)
# ----------------------------------------------------------------------
def _formula(expression: str, names: list[str], values: list[float]) -> str:
    import re

    from app.data_operations.fit_dialog.common import _substitute_expression_parameters

    return re.sub(r"<[^>]+>", "", _substitute_expression_parameters(expression, names, np.array(values)))


def test_fitted_values_take_the_place_of_the_formulas_symbols() -> None:
    # The catalogue names a parameter by its role ("offset", "rate k"), the formula by its symbol (C, k).
    assert _formula("y = A exp(-k x) + C", ["A", "rate k", "offset"], [4, 0.5, 1]) == "y = 4 exp(-0.5 x) + 1"
    assert _formula("<b>Linear</b><br>y = b + m x", ["intercept", "slope"], [1.5, 2]) == "Linear" "y = 1.5 + 2 x"
    # Greek words, subscripts, a symbol run into x, an exponent after the symbol.
    assert _formula("y = A sin(ωx + φ) + C", ["amplitude A", "angular frequency omega", "phase phi", "offset"],
                    [1, 2, 3, 4]) == "y = 1 sin(2x + 3) + 4"
    assert _formula("y = A₁ exp(-k₁x) + A₂ exp(-k₂x) + C", ["A1", "k1", "A2", "k2", "offset"],
                    [1, 2, 3, 4, 5]) == "y = 1 exp(-2x) + 3 exp(-4x) + 5"
    assert _formula("y = C + A exp(-(x - x₀)² / (2 σ²))", ["amplitude A", "center x0", "sigma", "offset"],
                    [2.5, -0.5, 1.25, 0.1]) == "y = 0.1 + 2.5 exp(-(x - (-0.5))² / (2 1.25²))"


def test_a_symbol_is_not_found_inside_another_and_never_inside_a_tag() -> None:
    assert _formula("y = C + A exp(-Ea / (kB T))", ["A", "Ea eV", "offset"], [1, 0.7, 0.1]) == "y = 0.1 + 1 exp(-0.7 / (kB T))"
    from app.data_operations.fit_dialog.common import _substitute_expression_parameters

    out = _substitute_expression_parameters('<span class="A">A</span>', ["A"], np.array([3.0]))
    assert out == '<span class="A">3</span>'  # the attribute is untouched


def test_the_catalogue_formulas_get_most_of_their_parameters() -> None:
    import importlib
    import inspect
    import re

    from app.plugin_api.functions.base import base_function
    from app.data_operations.fit_dialog.common import _expression_symbols

    placed = total = 0
    for name in ("app.functions.functions", "app.functions.nist_functions"):
        module = importlib.import_module(name)
        for cls in vars(module).values():
            if inspect.isclass(cls) and issubclass(cls, base_function) and cls is not base_function and cls.__module__ == name:
                expression, params = getattr(cls, "expression", None), getattr(cls, "params", None)
                if expression and params:
                    symbols = _expression_symbols(re.sub(r"<[^>]+>", " ", str(expression)), list(params))
                    placed += sum(symbol is not None for symbol in symbols)
                    total += len(params)
    assert placed / total > 0.9, (placed, total)


def test_every_series_is_reported_in_one_document_with_its_own_columns(dialog: SeriesFitDialog) -> None:
    import re

    assert dialog.select_model("Exponential decay")
    dialog._fit_mode_combo.setCurrentIndex(1)
    dialog.on_estimate_initial_values()
    dialog.evaluate(dialog._after_fit, background=False, optimise=True)
    results = dialog.compute_results()
    report = dialog.format_results(results)
    # A page has one <html> and one <body>; a report per series pasted in turn nested them.
    assert report.count("<html>") == 1 and report.count("<body") == 1
    text = re.sub(r"<[^>]+>", " ", report)
    assert "Series: s0" in text and "Series: s1" in text
    assert [result.target_col for result in results] == ["y0", "y1"]  # not both the last series'
    assert len({result.output_table for result in results}) == 2
    assert all(f"{result.params[0]:.8g}" in re.sub(r"<[^>]+>", "", result.evaluated_expression) for result in results)


def test_one_result_is_the_plain_report(dialog: SeriesFitDialog) -> None:
    assert dialog.select_model("Exponential decay")
    dialog.series_selector.series_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    dialog.on_estimate_initial_values()
    dialog.evaluate(dialog._after_fit, background=False, optimise=True)
    report = dialog.format_results(dialog.compute_results())
    assert report.count("<html>") == 1 and "Series:" not in report
