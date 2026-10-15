"""The rail's search field keeps its icon inside the field, in every theme.

The Fluent sheet gave every QToolButton a 36 px minimum and 8 px of side
padding. A QLineEdit makes QToolButtons for its own leading icon and clear
button, so each became 54 x 38 inside a 32 px field: on Windows the search icon
sat below the field's bottom-left corner and the text was pushed right.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QToolButton

import app.styles.style as style
from app.widgets.catalog_search import CatalogSearch


@pytest.fixture(scope="module", params=["fluent_win11", "macos_native", "ubuntu_native"])
def theme(qapp: QApplication, request: pytest.FixtureRequest):
    # Once per theme for the file: applying a sheet repolishes every widget made so far.
    style.apply_platform_style(qapp, request.param)
    yield request.param
    style.apply_platform_style(qapp)


def test_the_search_icon_and_clear_button_fit_inside_the_field(qapp: QApplication, theme: str) -> None:
    search = CatalogSearch(None, compact=True, collapsed=True)
    search.resize(220, 200)
    search.show()
    qapp.processEvents()
    field = search.field
    buttons = field.findChildren(QToolButton)
    assert buttons, "a QLineEdit with a leading action and a clear button has tool buttons"
    for button in buttons:
        geometry = button.geometry()
        assert geometry.top() >= 0 and geometry.bottom() < field.height(), (theme, geometry, field.height())
        assert geometry.left() >= 0 and geometry.right() < field.width(), (theme, geometry, field.width())
        assert button.height() <= field.height() - 4, (theme, button.height(), field.height())
    search.close()
