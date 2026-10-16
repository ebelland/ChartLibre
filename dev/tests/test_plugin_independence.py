"""Plugins are files that can be added or removed alone: nothing in the core names one."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.scanners.class_discovery import discover_classes_merged


def test_a_family_with_no_plugin_is_logged_as_an_error(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.ERROR):
        found = discover_classes_merged(
            roots=(tmp_path / "nowhere", tmp_path),
            base_class_name="DataOperationDialogBase",
            family="data operations",
            require_value_attr=False,
        )
    assert found == []
    assert any("No data operations found" in record.getMessage() for record in caplog.records)


def test_a_family_with_a_plugin_is_not_logged(tmp_path: Path, caplog) -> None:
    (tmp_path / "one.py").write_text(
        "class One(DataOperationDialogBase):\n"
        "    Name = 'One'\n    Category = 'Mine'\n    Order = 7\n    Goals = ['describe']\n"
    )
    with caplog.at_level(logging.ERROR):
        found = discover_classes_merged(
            roots=(tmp_path,), base_class_name="DataOperationDialogBase", family="data operations",
            string_attrs=("Category",), string_list_attrs=("Goals",), int_attrs=("Order",),
            require_value_attr=False,
        )
    assert [(e["value"], e["category"], e["order"], e["goals"]) for e in found] == [("One", "Mine", 7, ["describe"])]
    assert not caplog.records


def test_the_shipped_plugins_are_found_in_every_family() -> None:
    from app.scanners.axis_renderer_scanner import renderers
    from app.scanners.data_operation_scanner import data_operations
    from app.scanners.table_operation_scanner import table_operations

    assert renderers and data_operations and table_operations
    # Every data operation says where it lists.
    assert all(op.get("category") for op in data_operations)


def test_the_data_operations_panel_groups_by_the_plugins_own_category(qapp) -> None:
    from app.widgets.data_operation import _group_by_section

    operations = [
        {"value": "B", "category": "Zeta", "order": 20},
        {"value": "A2", "category": "Alpha", "order": 11},
        {"value": "A1", "category": "Alpha", "order": 10},
        {"value": "Loose"},
    ]
    grouped = _group_by_section(operations)
    assert [title for title, _items in grouped] == ["Alpha", "Zeta", "Other"]
    assert [op["value"] for op in grouped[0][1]] == ["A1", "A2"]
    # A plugin removed leaves the others where they were.
    assert [title for title, _i in _group_by_section(operations[:2])] == ["Alpha", "Zeta"]


def test_a_report_leaves_out_a_chart_type_that_is_not_installed(qapp, tmp_path: Path, caplog, monkeypatch) -> None:
    from app.data.sqlite_repo import SqliteRepo
    from app.scanners import axis_renderer_scanner
    from app.table_operations.differences_dialog import DifferencesDialog

    repo = SqliteRepo(db_path=tmp_path / "p.dhub")
    try:
        monkeypatch.setattr(axis_renderer_scanner, "renderers",
                            [r for r in axis_renderer_scanner.renderers if r["value"] != "Scatter Plot"])
        dialog = DifferencesDialog.__new__(DifferencesDialog)
        dialog._repo = repo
        with caplog.at_level(logging.WARNING):
            figure_id = DifferencesDialog.create_report_figure(
                dialog, "t", "<p>x</p>", [("Scatter Plot", "gone", "x", "y", [])])
        assert any("not installed" in record.getMessage() for record in caplog.records)
        assert repo.load_figure_descriptor(figure_id).axes == []
    finally:
        repo.close()
