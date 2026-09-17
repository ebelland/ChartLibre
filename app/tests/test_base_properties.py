"""BaseProperties must never claim the layout slot its subclasses need.

Regression guard for a real bug: BaseProperties.__init__ used to call
self.setLayout(QVBoxLayout()) before any subclass got a chance to build its
own. QWidget accepts exactly one layout, so every subclass's own
QVBoxLayout(self) call in _build_ui() silently failed (Qt logs a warning
and keeps the first layout, not the second), leaving the widget's real
controls built but never attached to anything the widget would actually
lay out - every properties panel rendered at an empty widget's size (22x22)
instead of its content's.
"""
from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtWidgets import QWidget

from app.widgets.axis_properties import AxisPropertiesWidget
from app.widgets.base_properties import BaseProperties
from app.widgets.figure_properties import FigurePropertiesWidget
from app.widgets.series_operation import SeriesOperationWidget
from app.widgets.series_properties import SeriesPropertiesWidget

WIDGET_CLASSES = (
    FigurePropertiesWidget,
    AxisPropertiesWidget,
    SeriesPropertiesWidget,
    SeriesOperationWidget,
)


def test_base_properties_sets_no_layout_of_its_own(qapp) -> None:
    """The base class must leave the layout slot free for a subclass."""
    bare = BaseProperties()
    assert bare.layout() is None


@pytest.mark.parametrize("widget_class", WIDGET_CLASSES)
def test_each_editor_s_own_layout_actually_attaches(qapp, widget_class) -> None:
    widget = widget_class()

    layout = widget.layout()
    assert layout is not None
    assert layout.count() > 0, (
        f"{widget_class.__name__}'s own layout has no content - its "
        "_build_ui() lost the fight for the layout slot"
    )


def test_before_reload_runs_between_attaching_and_reloading(qapp) -> None:
    """The one extension point BaseProperties offers a subclass that needs
    something between "attributes set" and "fields reloaded" -
    FigurePropertiesWidget's rcParams push is the real user of this."""
    seen: list[tuple[Any, Any]] = []

    class _Probe(BaseProperties):
        def _before_reload(self) -> None:
            seen.append((self._repo, self._figure_id))

        def _reload_from_descriptor(self) -> None:
            # _before_reload must already have run by the time this does.
            assert seen == [("a-repo", 7)]

        def _set_enabled_state(self, enabled: bool) -> None:
            pass

    probe = _Probe()
    probe.set_connected_figure("a-repo", 7, figure=None)

    assert seen == [("a-repo", 7)]


@pytest.mark.parametrize("widget_class", WIDGET_CLASSES)
def test_each_editor_reports_a_real_content_sized_hint(qapp, widget_class) -> None:
    """22x22 - QWidget's own bare default - is exactly the empty-layout
    symptom; every one of these has far more controls than that."""
    widget = widget_class()

    hint = widget.sizeHint()
    assert hint.width() > 100
    assert hint.height() > 100


# ----------------------------------------------------------------------
# The ground the cards sit on
# ----------------------------------------------------------------------
def test_a_stylesheet_background_can_actually_paint(qapp) -> None:
    """WA_StyledBackground, the Qt gotcha: without it a stylesheet
    background on a plain QWidget subclass is parsed and never painted,
    which is what makes QSS look like it "does not work" on custom
    widgets. QFrame-based cards do not need it; this does."""
    from PySide6.QtCore import Qt

    assert BaseProperties().testAttribute(Qt.WidgetAttribute.WA_StyledBackground)


@pytest.mark.parametrize("qss_name", ["fluent_win11.qss", "macos_native.qss"])
def test_both_stylesheets_say_what_the_panel_ground_is(qss_name: str) -> None:
    """base_properties.py's own comment promises a #basePropertiesPanel
    rule in each sheet - it went one round without one, so it was
    describing something that did not exist."""
    from app.styles.style import _load_qss

    qss, _path = _load_qss(qss_name)
    assert qss is not None
    assert "#basePropertiesPanel" in qss


def test_the_page_is_white_on_both_platforms() -> None:
    """Both platforms read #basePropertiesPanel as the white page ground
    now - current System Settings' own white-page-grey-card convention
    (see macos_native.qss's Surfaces note) reads the same way WinUI3's
    one-white-page-with-outlined-cards already did, even though the two
    still differ on what a *card* looks like (see the test below)."""
    from app.styles.style import _load_qss

    def panel_rule(name: str) -> str:
        qss, _path = _load_qss(name)
        block = qss.split("#basePropertiesPanel", 1)[1]
        return block.split("}", 1)[0]

    assert "palette(base)" in panel_rule("fluent_win11.qss")
    assert "palette(base)" in panel_rule("macos_native.qss")


def test_the_two_platforms_still_answer_differently_on_cards() -> None:
    """White cards with a hairline border on Windows (WinUI3 Settings),
    grey inset "well" cards on macOS (current System Settings - the
    VoiceOver info box, "Controlli della barra dei menu", ...). Matching
    them up would be the bug - the page-ground convention above changed,
    the card-vs-page contrast that convention exists for did not."""
    from app.styles.style import _load_qss

    def card_rule(name: str) -> str:
        qss, _path = _load_qss(name)
        block = qss.split('[card="true"]', 1)[1]
        return block.split("}", 1)[0]

    assert "palette(base)" in card_rule("fluent_win11.qss")
    assert "palette(window)" in card_rule("macos_native.qss")


#: Not SeriesOperationWidget: its one card deliberately has a zero-margin
#: outer layout, so the hint bar fixed at its bottom (see
#: series_operation.py's module docstring) can sit flush with the card's
#: own rounded bottom corners rather than leaving a gap that would break
#: the corner match. The other three are property *editors* - a form in a
#: card - where flush-against-the-border is simply cramped, not deliberate.
_PROPERTY_EDITOR_CLASSES = (
    FigurePropertiesWidget,
    AxisPropertiesWidget,
    SeriesPropertiesWidget,
)


@pytest.mark.parametrize("widget_class", _PROPERTY_EDITOR_CLASSES)
def test_every_card_s_own_layout_has_room_to_breathe(qapp, widget_class) -> None:
    """Every card=true frame's own top-level layout must use
    apply_card_layout, not stdSizeAndlayout's zeroed margins - the latter
    is right for a layout *nested* inside a card, wrong for the outermost
    one, where content flush against the card's border reads as cramped.
    Series properties' two cards used stdSizeAndlayout here by mistake,
    the only panel of the three with no padding around its own content."""
    widget = widget_class()

    cards = [
        child for child in widget.findChildren(QWidget) if child.property("card")
    ]
    assert cards, f"{widget_class.__name__} has no card=true frame to check"

    for card in cards:
        layout = card.layout()
        if layout is None:
            continue
        left, top, right, bottom = (
            layout.contentsMargins().left(),
            layout.contentsMargins().top(),
            layout.contentsMargins().right(),
            layout.contentsMargins().bottom(),
        )
        assert (left, top, right, bottom) != (0, 0, 0, 0), (
            f"{widget_class.__name__}'s {card.objectName()!r} card has no "
            "margins on its own layout"
        )
