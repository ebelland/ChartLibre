"""The catalogue of what the application does, and the search over it."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton

from app.analysis import catalog
from app.scanners.axis_renderer_scanner import renderers
from app.scanners.series_operation_scanner import series_operations
from app.scanners.table_operation_scanner import table_operations
from app.widgets.catalog_search import CatalogSearch
from app.widgets.guided_panel import GuidedAnalysisPanel


def _names(entries) -> list[str]:
    return [entry.name for entry in entries]


def test_the_catalogue_lists_every_analysis_operation_chart_and_tool() -> None:
    entries = catalog.build_catalog()
    keys = {entry.key for entry in entries}
    assert {f"analysis:{op['value']}" for op in table_operations} <= keys
    assert {f"series:{op['value']}" for op in series_operations} | {"series:Plot"} <= keys
    assert {f"chart:{renderer['value']}" for renderer in renderers} <= keys
    assert {entry.payload for entry in entries if entry.kind == catalog.TOOL} == set(catalog.TOOLS)
    goals = {key for key, _title, _sentence in catalog.GOALS}
    assert all(set(entry.goals) <= goals for entry in entries)
    # Every goal leads somewhere.
    assert all(catalog.for_goal(entries, goal) for goal in goals)


def test_search_finds_by_synonym_in_either_language_and_ranks_names_first() -> None:
    entries = catalog.build_catalog()
    assert "Fit Model" in _names(catalog.search(entries, "anova"))[:2]
    assert _names(catalog.search(entries, "albero"))[0] == "Partition"
    assert _names(catalog.search(entries, "decision tree"))[0] == "Partition"
    assert _names(catalog.search(entries, "anomalie"))[0] == "Outliers"
    assert _names(catalog.search(entries, "stabilità"))[0] == "Control Chart"  # accents do not matter
    assert _names(catalog.search(entries, "box"))[0] == "Box Plot"  # the name before a keyword
    # A translated name is found too.
    italian = {"Smoothing": "Smussamento"}
    assert _names(catalog.search(entries, "smuss", italian.get))[0] == "Smoothing"
    assert catalog.search(entries, "") == [] and catalog.search(entries, "zzzz") == []


def test_return_in_the_field_chooses_the_best_match_and_the_arrows_another(qapp) -> None:
    search = CatalogSearch(collapsed=True)
    chosen = []
    search.entry_chosen.connect(chosen.append)
    assert search.results.isHidden()
    search.field.setText("regression")
    assert not search.results.isHidden()
    first, second = search.matches()[:2]
    QTest.keyClick(search.field, Qt.Key.Key_Return)
    QTest.keyClick(search.field, Qt.Key.Key_Down)
    QTest.keyClick(search.field, Qt.Key.Key_Return)
    assert chosen == [first, second]


def test_the_guided_panel_searches_and_opens_what_is_found(qapp) -> None:
    panel = GuidedAnalysisPanel()
    found = []
    panel.entry_requested.connect(found.append)
    panel.search.field.setText("fit model")
    QTest.keyClick(panel.search.field, Qt.Key.Key_Return)
    assert [entry.name for entry in found] == ["Fit Model"]


def test_every_equivalent_opens_something_and_its_words_find_it() -> None:
    entries = catalog.build_catalog()
    keys = {entry.key for entry in entries}
    assert all(key in keys for _task, _where, key, _programs in catalog.EQUIVALENTS)
    assert all(set(programs) <= set(catalog.PROGRAMS) for *_rest, programs in catalog.EQUIVALENTS)
    assert _names(catalog.search(entries, "crosstabs")) == ["Statistics"]
    assert _names(catalog.search(entries, "control chart builder"))[0] == "Control Chart"
    assert _names(catalog.search(entries, "pivottable")) == ["Edit table"]


def test_the_manual_appendix_has_a_row_per_equivalent() -> None:
    from dev.manual.make_equivalents import render

    text = render()
    assert text.count("\n  [*") == len(catalog.EQUIVALENTS)
    assert "table.header([*Task*], [*ChartLibre*], [*JMP*], [*SPSS*], [*Minitab*], [*Origin / Excel*])" in text


def test_every_goal_has_suggestions_that_exist() -> None:
    entries = catalog.build_catalog()
    keys = {entry.key for entry in entries}
    assert set(catalog.GUIDE) == {key for key, _title, _sentence in catalog.GOALS}
    assert all(key in keys for rows in catalog.GUIDE.values() for key, _why, _suits in rows)
    assert all(catalog.suggestions(entries, goal) for goal in catalog.GUIDE)


def test_the_guided_panel_suggests_by_goal_and_response_and_opens_one(qapp) -> None:
    panel = GuidedAnalysisPanel()
    opened = []
    panel.entry_requested.connect(opened.append)
    keys = [key for key, _title, _sentence in catalog.GOALS]
    panel.goals.setCurrentRow(keys.index("compare"))
    assert "chart:Mosaic Plot" in [e.key for e in panel.shown()]
    panel._response.setCurrentIndex(panel._response.findData(catalog.NUMERIC))
    shown = [e.key for e in panel.shown()]
    assert shown[:2] == ["analysis:Are there differences?", "series:Statistics"] and "chart:Mosaic Plot" not in shown
    panel._cards()[0].findChild(QPushButton).click()
    assert [e.key for e in opened] == ["analysis:Are there differences?"]
