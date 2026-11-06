"""New plot's chart types: the categories and the recent charts first, then one category's list and Back."""
from __future__ import annotations

from PySide6.QtCore import Qt

from app.scanners.axis_renderer_scanner import renderers
from app.widgets.renderer_picker import NUM_RECENT, RendererPicker, recent_charts, remember_chart

_NAMES = [str(chart["value"]) for chart in renderers]


def _rows(picker: RendererPicker) -> list[str]:
    return [picker.home.item(row).text() for row in range(picker.home.count())]


def test_a_category_opens_its_list_and_back_returns(qapp) -> None:
    picker = RendererPicker(renderers)
    seen: list[str] = []
    picker.changed.connect(seen.append)
    assert picker.pages.currentIndex() == 0 and picker.categories()[0] == "Pairwise data"
    item = next(picker.home.item(row) for row in range(picker.home.count())
                if picker.home.item(row).text().startswith("Gridded data"))
    picker._home_chosen(item)
    assert picker.pages.currentIndex() == 1
    names = [picker.charts.item(row).text() for row in range(picker.charts.count())]
    assert "Heatmap" in names and "Scatter Plot" not in names
    picker.charts.setCurrentRow(names.index("Heatmap"))
    assert seen == ["Heatmap"] and picker.current == "Heatmap"
    picker.back.click()
    assert picker.pages.currentIndex() == 0 and picker.current == "Heatmap"
    picker.deleteLater()


def test_the_search_lists_its_matches_and_clearing_it_goes_home(qapp) -> None:
    picker = RendererPicker(renderers)
    picker.search.setText("violin")
    assert picker.pages.currentIndex() == 1 and picker.charts.item(0).text() == "Violin Plot"
    picker.search.clear()
    assert picker.pages.currentIndex() == 0
    picker.deleteLater()


def test_recent_keeps_the_last_ones_newest_first(qapp) -> None:
    for name in ("Heatmap", "Box Plot", "Pie Chart", "Bar Chart", "Box Plot"):
        remember_chart(name)
    assert NUM_RECENT == 3 and recent_charts(_NAMES) == ["Box Plot", "Bar Chart", "Pie Chart"]
    picker = RendererPicker(renderers)
    rows = _rows(picker)
    assert rows[:4] == ["Recent", "Box Plot", "Bar Chart", "Pie Chart"] and rows[4] == "Categories"
    assert not picker.home.item(0).flags() & Qt.ItemFlag.ItemIsEnabled
    picker._home_chosen(picker.home.item(1))
    assert picker.current == "Box Plot" and picker.pages.currentIndex() == 1
    assert picker.charts.currentItem().text() == "Box Plot"
    picker.deleteLater()


def test_matplotlibs_families_hold_matplotlibs_charts() -> None:
    category = {str(chart["value"]): chart["category"] for chart in renderers}
    assert category["Pie Chart"] == "Statistical distributions" and category["Stairs"] == "Pairwise data"
    assert category["Image Plot"] == "Gridded data" and category["Quiver 3D"] == "3D and volumetric data"
    assert category["Sankey Diagram"] == "Specialized charts"
