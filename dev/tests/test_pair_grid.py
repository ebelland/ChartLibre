"""Label and field pairs that reflow instead of clipping a number (todo A-01)."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QLabel, QWidget

from app.styles.style import KEEP_MINIMUM_WIDTH, fit_spin_width, relax_minimum_width
from app.widgets.pair_grid import PairGrid
from dev.tests._cases import check_all


def _spin(qapp, suffix: str = "", maximum: float = 500.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.1, maximum)
    spin.setDecimals(2)
    spin.setSuffix(suffix)
    fit_spin_width(spin)
    return spin


def test_a_spin_box_keeps_the_width_its_widest_value_needs(qapp) -> None:
    plain = _spin(qapp)
    with_suffix = _spin(qapp, " cm")
    assert with_suffix.minimumWidth() > plain.minimumWidth()

    # relax_minimum_width lets a panel shrink; it must not squeeze a number.
    holder = QWidget()
    plain.setParent(holder)
    before = plain.minimumWidth()
    relax_minimum_width(holder)
    assert plain.property(KEEP_MINIMUM_WIDTH) is True
    assert plain.minimumWidth() == before


def test_a_huge_range_does_not_reserve_a_huge_width(qapp) -> None:
    spin = _spin(qapp, maximum=1e12)
    assert spin.minimumWidth() < 200


_CASES_PAIRS_TAKE_THE_WIDEST_ARRANGEMENT_THAT_FITS = [(900, 2, False), (260, 1, False), (120, 1, True)]


def _pairs_take_the_widest_arrangement_that_fits(qapp, width: int, per_row: int, stacked: bool) -> None:
    left, right = _spin(qapp, " cm"), _spin(qapp, " cm")
    grid = PairGrid([(QLabel("Width"), left), (QLabel("Height"), right)])
    field = left.minimumWidth()
    label = grid._pairs[0][0].sizeHint().width()
    # Sanity of the fixture: the three widths really are in different bands.
    assert 2 * (label + 8 + field) + 8 < 900
    assert label + 8 + field > 120
    grid.resize(width, 200)
    grid.show()
    qapp.processEvents()
    assert grid._arrangement == (per_row, stacked)
    if not stacked:
        # Side by side or one per row, the label sits left of its field.
        assert grid._pairs[0][0].geometry().right() <= grid._pairs[0][1].geometry().left()
    else:
        assert grid._pairs[0][0].geometry().bottom() <= grid._pairs[0][1].geometry().top()
    grid.close()


def test_pairs_take_the_widest_arrangement_that_fits(qapp) -> None:
    check_all(lambda *case: _pairs_take_the_widest_arrangement_that_fits(**dict(zip(['width', 'per_row', 'stacked'], case)), qapp=qapp), _CASES_PAIRS_TAKE_THE_WIDEST_ARRANGEMENT_THAT_FITS)


def test_the_field_is_never_narrower_than_its_minimum(qapp) -> None:
    left, right = _spin(qapp, " cm"), _spin(qapp, " cm")
    grid = PairGrid([(QLabel("Width"), left), (QLabel("Height"), right)])
    for width in (900, 400, 260, 200, 160):
        grid.resize(width, 200)
        grid.show()
        qapp.processEvents()
        assert left.width() >= left.minimumWidth()
        assert right.width() >= right.minimumWidth()
    grid.close()
