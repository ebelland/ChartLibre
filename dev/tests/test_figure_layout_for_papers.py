"""Panel letters, mosaic and stacked/side-by-side presets (G-18)."""
from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from app.plugin_api.charts import layout_presets, panel_labels


def test_panel_label_styles_and_beyond_z() -> None:
    assert [panel_labels.label_text("a", i) for i in range(3)] == ["a", "b", "c"]
    assert panel_labels.label_text("(A)", 2) == "(C)"
    assert panel_labels.label_text("a)", 1) == "b)"
    assert panel_labels.label_text("A.", 25) == "Z."
    assert panel_labels.label_text("a", 26) == "aa"
    assert panel_labels.label_text("", 0) == ""


def test_letters_are_written_on_each_axes_in_the_order_given() -> None:
    figure = Figure()
    axes = [figure.add_subplot(1, 3, n + 1) for n in range(3)]
    panel_labels.apply_panel_labels(axes, {"panel_labels": "(a)"})
    assert [ax.texts[0].get_text() for ax in axes] == ["(a)", "(b)", "(c)"]
    other = Figure().add_subplot()
    panel_labels.apply_panel_labels([other], {"panel_labels": "", })
    assert not other.texts


def test_stacked_and_side_by_side_share_the_first_axes_scale() -> None:
    stacked = layout_presets.plan_layout(layout_presets.STACKED_SHARED_X, [7, 8, 9])
    assert (stacked.nrows, stacked.ncols) == (3, 1)
    assert [p.options["sharex"] for p in stacked.axes] == [None, True, True]
    side = layout_presets.plan_layout(layout_presets.SIDE_BY_SIDE_SHARED_Y, [7, 8])
    assert (side.nrows, side.ncols) == (1, 2)
    assert side.axes[1].options["sharey"] is True


def test_a_mosaic_places_spans_and_leaves_extra_axes_in_rows_below() -> None:
    plan = layout_presets.plan_mosaic("AAB;CCB", [10, 11, 12])
    assert (plan.nrows, plan.ncols) == (2, 3)
    by_id = {p.axis_id: p for p in plan.axes}
    assert (by_id[10].axis_index, by_id[10].options["col_span"], by_id[10].options["row_span"]) == (0, 2, None)
    assert (by_id[11].axis_index, by_id[11].options["row_span"]) == (2, 2)
    assert by_id[12].axis_index == 3 and by_id[12].options["col_span"] == 2
    extra = layout_presets.plan_mosaic("A.;.B", [1, 2, 3])
    assert (extra.nrows, extra.ncols) == (3, 2)
    assert {p.axis_id: p.axis_index for p in extra.axes} == {1: 0, 2: 3, 3: 4}


@pytest.mark.parametrize(
    "text, problem",
    [("", "empty"), ("AB;C", "same number"), ("AB;BA", "rectangle"), ("AZ", "names no axis")],
)
def test_a_bad_mosaic_says_what_is_wrong(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        layout_presets.plan_mosaic(text, [1, 2])
