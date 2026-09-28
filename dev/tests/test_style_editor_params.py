"""The style editor's parameter list: every style rcParam can be edited."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStyleOptionViewItem

import app.utils.config as config
from app.dialogs.edit_mpl_styles_dialog import MplStyleEditorDialog, style_rc_keys


def test_no_system_rcparams_are_offered() -> None:
    keys = style_rc_keys()
    assert not [k for k in keys if k.split(".")[0] in {"_internal", "agg", "backend", "keymap", "webagg"}]


def test_every_rcparam_toggles_or_opens_an_editor(qapp, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "USER_CONFIG_PATH", tmp_path / "user.json")
    dialog = MplStyleEditorDialog(None, initial_style_text="boxplot.notch: False\n")
    editor = dialog.editor
    keys = style_rc_keys()
    for key in keys:
        dialog._param_schema[key] = dialog._schema_for_key(key, "")
    dialog._rebuild_editor({key: "" for key in keys})
    delegate = editor.tree.itemDelegateForColumn(1)

    stuck = []
    for key in keys:
        item = editor._key_items[key]
        if editor.kind_for_key(key) == "bool":
            before = item.checkState(1)
            editor._on_item_clicked(item, 1)
            if item.checkState(1) == before:
                stuck.append(key)
        elif delegate.createEditor(editor.tree.viewport(), QStyleOptionViewItem(), editor.tree.indexFromItem(item, 1)) is None:
            stuck.append(key)
    dialog.close()
    assert stuck == []


def test_false_from_a_style_file_reads_as_unchecked(qapp, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "USER_CONFIG_PATH", tmp_path / "user.json")
    dialog = MplStyleEditorDialog(None, initial_style_text="boxplot.notch: False\n")
    assert dialog.editor._key_items["boxplot.notch"].checkState(1) == Qt.CheckState.Unchecked
    dialog.close()
