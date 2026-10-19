"""The Analysis panel: data operations, table analyses, New chart and the DOE, a frame per JMP-like category."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from app.scanners.data_operation_scanner import data_operations
from app.scanners.table_operation_scanner import table_operations
from app.widgets.analysis_panel import AnalysisPanel, group_by_category


@pytest.fixture
def panel(qapp) -> Iterator[AnalysisPanel]:
    widget = AnalysisPanel()
    widget.show()
    yield widget
    # Destroyed here, not left to Python's collector: a panel freed later, in
    # the middle of another test's background run, took the process down.
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _names(panel: AnalysisPanel) -> list[str]:
    return [str(item.data(Qt.ItemDataRole.UserRole).get("value")) for item in panel.items()]


def test_every_operation_and_analysis_is_a_row(panel: AnalysisPanel) -> None:
    names = _names(panel)
    assert names[0] == "New Chart" and names[-1] == "Create a table from a DOE"
    assert {str(op["value"]) for op in data_operations} <= set(names)
    analyses = {str(op["value"]) for op in table_operations if op.get("category") != "Quick answers"}
    assert analyses <= set(names)
    assert "Are there differences?" not in names  # the quick answers are the Guided analysis'


def test_the_frames_follow_the_plugins_own_categories_in_order(panel: AnalysisPanel) -> None:
    titles = panel.categories()
    assert titles[:3] == ["Graph", "Basic Analysis", "Fitting Linear Models"]
    assert titles[-1] == "Design of Experiments"
    assert "Analysis" not in titles  # Peaks, Roots and Calculus are under Curves now


def test_a_category_is_placed_by_its_lowest_order() -> None:
    entries = [{"value": "b", "category": "Late", "order": 9}, {"value": "a", "category": "Early", "order": 1},
               {"value": "c", "category": "Late", "order": 2}, {"value": "d"}]
    assert [(title, [e["value"] for e in items]) for title, items in group_by_category(entries)] == [
        ("Early", ["a"]), ("Late", ["c", "b"]), ("Other", ["d"])]


def test_a_click_opens_the_right_kind_of_entry(panel: AnalysisPanel) -> None:
    opened: list[tuple[str, str]] = []
    panel.data_operation_requested.connect(lambda e: opened.append(("data", e["value"])))
    panel.table_operation_requested.connect(lambda e: opened.append(("table", e["value"])))
    panel.tool_requested.connect(lambda key: opened.append(("tool", key)))
    by_name = {item.data(Qt.ItemDataRole.UserRole)["value"]: item for item in panel.items()}
    for name in ("Fit Curve", "Fit Model", "Create a table from a DOE"):
        item = by_name[name]
        view = item.listWidget()
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.visualItemRect(item).center())
    view = by_name["New Chart"].listWidget()
    view.setCurrentItem(by_name["New Chart"])
    QTest.keyClick(view, Qt.Key.Key_Return)
    assert opened == [("data", "Fit Curve"), ("table", "Fit Model"), ("tool", "doe"), ("data", "New Chart")]


def test_quality_and_process_lists_the_pareto_plot_and_the_ishikawa_diagram(qapp) -> None:
    """As JMP's Quality and Process: Pareto Plot and Diagram (Ishikawa), opening New chart on them."""
    from app.widgets.analysis_panel import panel_entries

    quality = {e["value"]: e for e in panel_entries() if e.get("category") == "Quality and Process"}
    assert {"Control Chart", "Pareto Plot", "Ishikawa Diagram"} <= set(quality)
    assert quality["Ishikawa Diagram"]["tool"] == "chart:Fishbone Diagram"
    assert quality["Pareto Plot"]["tool"] == "chart:Pareto Chart"
