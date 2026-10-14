"""An unset kwarg row stays unset when its editor is opened and left untouched.

The marker-size row (``s``, default None) opened a spin box that can only show
its minimum, 0; moving on to another row committed that 0, and every marker of
the axis vanished on the next redraw.
"""
from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtWidgets import QStyleOptionViewItem

from app.widgets.dictionary_editor import DictEditorPanel

SCHEMA = {
    "s": {"default": None, "type": float, "min": 0.0, "max": 1000.0},
    "alpha": {"default": 0.8, "type": float, "min": 0.0, "max": 1.0},
    "trend_degree": {"default": None, "type": int, "min": 0, "max": 6},
}


@pytest.fixture
def panel(qapp: Any) -> DictEditorPanel:
    return DictEditorPanel(SCHEMA)


def _open_and_close(panel: DictEditorPanel, key: str, edit: Any = None) -> Any:
    delegate = panel.tree.itemDelegateForColumn(1)
    index = panel.tree.indexFromItem(panel._key_items[key], 1)
    editor = delegate.createEditor(panel, QStyleOptionViewItem(), index)
    delegate.setEditorData(editor, index)
    if edit is not None:
        edit(editor)
    delegate.setModelData(editor, panel.tree.model(), index)
    return panel.get_values()[key]


def test_an_unset_float_row_stays_unset_when_left_untouched(panel: DictEditorPanel) -> None:
    assert _open_and_close(panel, "s") is None


def test_an_unset_int_row_stays_unset_when_left_untouched(panel: DictEditorPanel) -> None:
    assert _open_and_close(panel, "trend_degree") is None


def test_a_value_typed_into_an_unset_row_is_kept(panel: DictEditorPanel) -> None:
    assert _open_and_close(panel, "s", lambda editor: editor.setValue(25.0)) == 25.0


def test_typing_zero_counts_as_a_choice(panel: DictEditorPanel) -> None:
    def type_zero(editor: Any) -> None:
        editor.lineEdit().textEdited.emit("0")

    assert _open_and_close(panel, "s", type_zero) == 0.0


def test_a_row_with_a_value_keeps_it(panel: DictEditorPanel) -> None:
    assert _open_and_close(panel, "alpha") == 0.8
