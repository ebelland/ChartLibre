"""A parameter declared advanced is in a closed "Advanced" frame under the others (its arrow opens it)."""
from __future__ import annotations

from PySide6.QtWidgets import QWidget

from app.plugin_api.data_operations.parameter_form import ParameterForm
from app.plugin_api.data_operations.parameter_spec import BoolParam, ChoiceParam, IntParam


def test_advanced_rows_are_in_a_closed_frame_shown_while_one_is_visible(qapp) -> None:
    parent = QWidget()
    form = ParameterForm((
        ChoiceParam("method", "Method:", choices=(("A", "a"), ("B", "b"))),
        IntParam("seed", "Seed:", default_value=1, advanced=True, visible_for={"method": ("b",)}),
        BoolParam("plain", "Plain:"),
    ), parent)
    card = form.advanced_card
    assert card is not None and card.collapsed
    assert form.layout.rowCount() == 2  # method and plain; the seed is in the frame's own form
    assert card.isAncestorOf(form.field("seed"))
    assert card.isHidden()  # method A: its only row is hidden, so is the frame
    form.set_values({"method": "b"})
    form.refresh_visibility()
    assert not card.isHidden()
    assert form.values()["seed"] == 1


def test_no_frame_without_advanced_parameters(qapp) -> None:
    form = ParameterForm((IntParam("n", "N:", default_value=2),), QWidget())
    assert form.advanced_card is None
