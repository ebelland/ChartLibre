"""The Function dialog can also plot a surface z = f(x, y).

Same scanner-driven idea as the 1D case, one variable more: selecting a
function whose ``ndim`` is 2 evaluates it on a meshgrid over the declared
range (reused for both axes - see ``_compute_surface_result``'s own
docstring for why) instead of a single curve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.function_dialog import SeriesFunctionDialog
from app.utils.dialog_state import clear_state


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    """See test_calculus_surface.py's own fixture: SeriesFunctionDialog's
    selected function persists to the real user.json across runs."""
    clear_state("SeriesFunctionDialog")
    yield
    clear_state("SeriesFunctionDialog")


@pytest.fixture
def figure_id(repo: SqliteRepo) -> int:
    return repo.create_figure_descriptor(name="fig")


def _select_function_by_name(dialog: SeriesFunctionDialog, name: str) -> None:
    tree = dialog._function_tree
    for i in range(tree.topLevelItemCount()):
        category = tree.topLevelItem(i)
        for j in range(category.childCount()):
            child = category.child(j)
            payload = child.data(0, 0x0100)  # Qt.ItemDataRole.UserRole
            if isinstance(payload, dict) and payload.get("name") == name:
                tree.setCurrentItem(child)
                return
    raise AssertionError(f"Function {name!r} not found in the catalog tree")


@pytest.fixture
def dialog(qapp, repo: SqliteRepo, figure_id: int) -> SeriesFunctionDialog:
    built = SeriesFunctionDialog(repo=repo, figure_id=figure_id)
    _select_function_by_name(built, "Gaussian2D")
    built.set_parameter_values({"start": -3.0, "stop": 3.0, "points": 25})
    return built


def test_a_surface_function_is_recognised(dialog: SeriesFunctionDialog) -> None:
    assert dialog._is_2d_function() is True


def test_preview_produces_an_x_y_z_table(dialog: SeriesFunctionDialog) -> None:
    results = dialog.compute_results()
    assert len(results) == 1
    result = results[0]
    assert result.z is not None

    frame = result.to_frame()
    assert list(frame.columns) == ["x", "y", "z"]
    # A 25 x 25 meshgrid, flattened.
    assert len(frame) == 25 * 25

    # Gaussian2D with default parameters (a=0, amp=1, x0=0, sx=1, y0=0, sy=1)
    # peaks at 1.0 at the origin, which the sampled grid includes.
    assert np.max(result.z) == pytest.approx(1.0, abs=1e-6)


def test_1d_functions_still_produce_a_plain_x_y_table(qapp, repo: SqliteRepo, figure_id: int) -> None:
    dialog = SeriesFunctionDialog(repo=repo, figure_id=figure_id)
    _select_function_by_name(dialog, "Linear")
    assert dialog._is_2d_function() is False

    results = dialog.compute_results()
    assert results[0].z is None
    assert list(results[0].to_frame().columns) == ["x", "y"]
