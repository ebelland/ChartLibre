"""The Series Operations panel: a sectioned list, searchable, opening on click or Return."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from app.scanners.series_operation_scanner import series_operations
from app.widgets.series_operation import SeriesOperationWidget


@pytest.fixture
def panel(qapp) -> Iterator[SeriesOperationWidget]:
    widget = SeriesOperationWidget()
    widget.resize(280, 700)
    widget.show()
    yield widget
    widget.close()


def _opened(panel: SeriesOperationWidget) -> list[str]:
    opened: list[str] = []
    panel.operation_requested.connect(lambda operation: opened.append(str(operation["value"])))
    return opened


def test_every_operation_and_new_plot_is_a_row_under_a_title(panel: SeriesOperationWidget) -> None:
    names = [str(item.data(Qt.ItemDataRole.UserRole)["value"]) for item in panel.operation_items()]
    assert names[0] == "Plot"
    assert sorted(names[1:]) == sorted(str(op["value"]) for op in series_operations)
    titles = [panel._list.item(row) for row in range(panel._list.count())
              if panel._list.item(row).data(Qt.ItemDataRole.UserRole) is None]  # pyright: ignore[reportOptionalMemberAccess]
    assert len(titles) >= 6
    assert all(not (title.flags() & Qt.ItemFlag.ItemIsSelectable) for title in titles)  # pyright: ignore[reportOptionalMemberAccess]


def test_a_click_opens_and_return_opens_the_current_row(panel: SeriesOperationWidget) -> None:
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


def test_search_filters_rows_hides_empty_titles_and_return_opens_the_first(panel: SeriesOperationWidget) -> None:
    opened = _opened(panel)
    panel._search.setText("spectral")
    shown = [str(item.data(Qt.ItemDataRole.UserRole)["value"]) for item in panel.operation_items() if not item.isHidden()]
    assert shown == ["Spectral Analysis"]
    visible_titles = [panel._list.item(row).text() for row in range(panel._list.count())
                      if panel._list.item(row).data(Qt.ItemDataRole.UserRole) is None and not panel._list.item(row).isHidden()]  # pyright: ignore[reportOptionalMemberAccess]
    assert len(visible_titles) == 1
    QTest.keyClick(panel._search, Qt.Key.Key_Return)
    assert opened == ["Spectral Analysis"]
    panel._search.clear()
    assert not any(item.isHidden() for item in panel.operation_items())


def test_the_hint_bar_describes_the_current_row(panel: SeriesOperationWidget) -> None:
    fit = next(item for item in panel.operation_items() if item.data(Qt.ItemDataRole.UserRole)["value"] == "Fit")
    panel._list.setCurrentItem(fit)
    assert fit.text() in panel._hint_label.text()

