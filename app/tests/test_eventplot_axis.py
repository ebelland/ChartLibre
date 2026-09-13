"""Tests for the Event Plot renderer.

The one thing worth pinning beyond "it draws tick marks": each series is
its own row at its own offset - several channels stacked and told apart -
and the row labels, when asked for, name the series rather than a bare
number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base import SeriesData
from app.charts.eventplot import EventPlotAxisRenderer


def _render(series: list[SeriesData], style: dict | None = None):
    fig = Figure(figsize=(4.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    EventPlotAxisRenderer().render_axis(ax, series, style or {})
    return fig, ax


def test_each_series_draws_its_own_row() -> None:
    series = [
        SeriesData(name="a", df=pd.DataFrame({"x": [1.0, 2.0]}), style={}),
        SeriesData(name="b", df=pd.DataFrame({"x": [3.0]}), style={}),
    ]
    _fig, ax = _render(series)
    assert len(ax.collections) == 2


def test_row_labels_name_the_series_by_default() -> None:
    series = [
        SeriesData(name="channel one", df=pd.DataFrame({"x": [1.0]}), style={}),
        SeriesData(name="channel two", df=pd.DataFrame({"x": [2.0]}), style={}),
    ]
    _fig, ax = _render(series)
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "channel one",
        "channel two",
    ]


def test_row_labels_can_be_turned_off() -> None:
    series = [SeriesData(name="a", df=pd.DataFrame({"x": [1.0]}), style={})]
    _fig, ax = _render(series, {"row_labels": False})
    # Matplotlib still has default numeric ticks; the point is that our
    # series name was never written in as a label.
    assert "a" not in [t.get_text() for t in ax.get_yticklabels()]


def test_non_finite_events_are_dropped_not_drawn() -> None:
    series = [SeriesData(name="a", df=pd.DataFrame({"x": [np.nan, np.inf]}), style={})]
    _fig, ax = _render(series)
    assert len(ax.collections) == 0


def test_a_hidden_series_is_not_drawn() -> None:
    series = [
        SeriesData(
            name="a", df=pd.DataFrame({"x": [1.0]}), style={"visible": False}
        )
    ]
    _fig, ax = _render(series)
    assert len(ax.collections) == 0
