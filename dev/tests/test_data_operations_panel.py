"""The Data Operations panel: a sectioned list, searchable, opening on click or Return."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from app.scanners.data_operation_scanner import data_operations
from app.widgets.data_operation import DataOperationWidget


@pytest.fixture
def panel(qapp) -> Iterator[DataOperationWidget]:
    widget = DataOperationWidget()
    widget.resize(280, 700)
    widget.show()
    yield widget
    widget.close()


def _opened(panel: DataOperationWidget) -> list[str]:
    opened: list[str] = []
    panel.operation_requested.connect(lambda operation: opened.append(str(operation["value"])))
    return opened


def test_every_operation_and_new_plot_is_a_row_under_a_title(panel: DataOperationWidget) -> None:
    names = [str(item.data(Qt.ItemDataRole.UserRole)["value"]) for item in panel.operation_items()]
    assert names[0] == "Plot"  # the Query Builder is on the New Table page
    assert sorted(names[1:]) == sorted(str(op["value"]) for op in data_operations)
    titles = [panel._list.item(row) for row in range(panel._list.count())
              if panel._list.item(row).data(Qt.ItemDataRole.UserRole) is None]  # pyright: ignore[reportOptionalMemberAccess]
    assert len(titles) >= 6  # Plot, then the operations' own sections
    assert all(not (title.flags() & Qt.ItemFlag.ItemIsSelectable) for title in titles)  # pyright: ignore[reportOptionalMemberAccess]


def test_a_click_opens_and_return_opens_the_current_row(panel: DataOperationWidget) -> None:
    opened = _opened(panel)
    fit = next(item for item in panel.operation_items() if item.data(Qt.ItemDataRole.UserRole)["value"] == "Fit")
    rect = panel._list.visualItemRect(fit)
    panel._list.scrollToItem(fit)
    rect = panel._list.visualItemRect(fit)
    QTest.mouseClick(panel._list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert opened == ["Fit"]
    assert panel._list.selectedItems() == []
    panel._list.setFocus()
    panel._list.setCurrentItem(panel.operation_items()[0])
    QTest.keyClick(panel._list, Qt.Key.Key_Return)
    assert opened == ["Fit", "Plot"]


def test_the_hint_bar_describes_the_current_row(panel: DataOperationWidget) -> None:
    fit = next(item for item in panel.operation_items() if item.data(Qt.ItemDataRole.UserRole)["value"] == "Fit")
    panel._list.setCurrentItem(fit)
    assert fit.text() in panel._hint_label.text()



