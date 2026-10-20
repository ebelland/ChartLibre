"""The box plot's flier symbol ("sym") is picked from the marker list, not typed."""
from __future__ import annotations

from app.widgets.dictionary_editor import _kind_for_key


def test_sym_is_a_marker() -> None:
    assert _kind_for_key("sym", {}) == "marker"
    assert _kind_for_key("sym", {"kind": "marker", "type": str}) == "marker"


def test_the_editor_offers_the_marker_combo_for_it(qapp) -> None:
    from PySide6.QtWidgets import QStyleOptionViewItem

    from app.widgets.dictionary_editor import DictEditorPanel, DictValueDelegate
    from app.widgets.drop_down_controls.marker_combo import MarkerStyleCombo

    panel = DictEditorPanel()
    panel.set_config({"sym": {"default": None, "type": str, "kind": "marker"}})
    item = panel._key_items["sym"]
    index = panel.tree.indexFromItem(item, 1)
    editor = DictValueDelegate(panel).createEditor(panel.tree.viewport(), QStyleOptionViewItem(), index)
    assert isinstance(editor, MarkerStyleCombo)


def test_the_box_plots_stats_choices_are_drop_downs() -> None:
    from app.charts.box import BoxAxisRenderer

    for name in ("stats_type", "stats_position", "stats_color_mode"):
        spec = BoxAxisRenderer.Kwargs[name]
        assert _kind_for_key(name, spec) == "enum" and spec["default"] in spec["type"]
    assert BoxAxisRenderer.Kwargs["stats_color_mode"]["editable"]  # any colour can still be typed
