"""relax_minimum_width(widget, minimum=0) must actually lower the floor.

setMinimumWidth(0) alone is a silent no-op on any widget whose horizontal
size policy is Expanding rather than Shrink - Qt's own qSmartMinSize only
lets an explicit minimum override max(sizeHint, minimumSizeHint) when that
minimum is greater than zero. Property panels (Figure/Axis/Series) are
exactly such widgets, which is why they stayed wider than intended - "too
wide, check field min width" - even after relax_minimum_width had already
walked them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.data.sqlite_repo import SqliteRepo
from app.styles.style import relax_minimum_width
from app.widgets.axis_properties import AxisPropertiesWidget


@pytest.fixture
def widget(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]}),
        table_name="t",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="t",
            x_label="x",
            y_label="y",
            options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
    )

    built = AxisPropertiesWidget()
    built.set_connected_figure(repo, figure_id, figure=None)
    built.rebuild_kwargs_editor(axis_id)
    return built


def test_an_expanding_container_is_not_affected_by_minimum_width_alone(
    qapp,
) -> None:
    """The Qt quirk itself, isolated from this app's own widgets: proof the
    fix addresses something real rather than an artefact of one panel's
    particular layout."""
    container = QWidget()
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    label = QWidget(container)
    label.setMinimumWidth(400)
    from PySide6.QtWidgets import QHBoxLayout
    QHBoxLayout(container).addWidget(label)

    container.setMinimumWidth(0)  # the naive, ineffective call
    assert container.minimumSizeHint().width() >= 400

    relax_minimum_width(container)
    assert container.minimumSizeHint().width() < 400


def test_the_axis_properties_panel_actually_shrinks(widget: AxisPropertiesWidget) -> None:
    before = widget.minimumSizeHint().width()

    relax_minimum_width(widget)

    after = widget.minimumSizeHint().width()
    assert after < before
    # Not just smaller - small enough that a 260px left panel (this app's
    # own deliberate floor, PANEL_MIN_WIDTH + NAV_BAR_WIDTH) is the binding
    # constraint again, not this widget's own unrelaxed content.
    assert after < 100
