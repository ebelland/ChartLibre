"""Pure arithmetic tests for app.charts.layout_presets.

No Qt, no SqliteRepo, no Figure: a preset is just "given these axis ids, what
grid size and options does each one get", which is exactly what is asserted
here. Rendering the result (spans, sharex/sharey, twin_of) is covered
separately, in test_layout_and_overlapping_axes.py and
test_figure_layout_and_frame.py, against the machinery that already reads
those same options for a hand-built layout.
"""
from __future__ import annotations

import pytest

from app.charts import layout_presets as lp
from dev.tests._cases import check_all


def _options_by_id(plan: lp.LayoutPlan) -> dict[int, dict[str, object]]:
    return {placement.axis_id: placement.options for placement in plan.axes}


def _index_by_id(plan: lp.LayoutPlan) -> dict[int, int]:
    return {placement.axis_id: placement.axis_index for placement in plan.axes}


# ----------------------------------------------------------------------
# Degenerate cases: every preset agrees on 0 or 1 axis
# ----------------------------------------------------------------------






# ----------------------------------------------------------------------
# Every placement always carries every owned key, so applying a preset
# always replaces the last one's rather than merging with it.
# ----------------------------------------------------------------------


_CASES_EVERY_AXIS_ID_APPEARS_EXACTLY_ONCE = [(case,) for case in lp.PRESETS]


def _every_axis_id_appears_exactly_once(preset: str) -> None:
    ids = [10, 20, 30, 40, 50, 60, 70]
    plan = lp.plan_layout(preset, ids)
    placed_ids = [placement.axis_id for placement in plan.axes]
    assert sorted(placed_ids) == sorted(ids)
    assert len(placed_ids) == len(set(placed_ids))


def test_every_axis_id_appears_exactly_once() -> None:
    check_all(_every_axis_id_appears_exactly_once, _CASES_EVERY_AXIS_ID_APPEARS_EXACTLY_ONCE)




# ----------------------------------------------------------------------
# GRID
# ----------------------------------------------------------------------
def test_grid_places_every_axis_in_a_compact_square_ish_grid() -> None:
    plan = lp.plan_layout(lp.GRID, [1, 2, 3, 4])
    assert (plan.nrows, plan.ncols) == (2, 2)
    assert _index_by_id(plan) == {1: 0, 2: 1, 3: 2, 4: 3}




# ----------------------------------------------------------------------
# SHARED_GRID
# ----------------------------------------------------------------------






# ----------------------------------------------------------------------
# MAIN_AND_SECONDARY
# ----------------------------------------------------------------------
def test_main_and_secondary_gives_the_first_axis_the_full_top_row() -> None:
    plan = lp.plan_layout(lp.MAIN_AND_SECONDARY, [1, 2, 3, 4, 5])
    options = _options_by_id(plan)
    index = _index_by_id(plan)
    assert index[1] == 0
    assert options[1]["col_span"] == plan.ncols






# ----------------------------------------------------------------------
# OVERLAPPING
# ----------------------------------------------------------------------




