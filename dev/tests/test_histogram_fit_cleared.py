"""A fit cleared in the axis' properties stays cleared, though a report figure stored it in the axis' own options."""
from __future__ import annotations

from app.charts.histogram import HistogramAxisRenderer
from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_properties.axis_properties import AxisPropertiesWidget


def _report_histogram(repo: SqliteRepo) -> tuple[int, int]:
    """A histogram as Distribution draws one: its fit in the axis' own options, not in axis_kwargs."""
    figure_id = int(repo.create_figure_descriptor(name="Distribution"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Histogram",
                                              title="", x_label="", y_label="",
                                              options={"distribution_fit": "norm"}))
    return figure_id, axis_id


def test_clearing_the_fit_does_not_bring_the_normal_back(qapp, repo: SqliteRepo) -> None:
    figure_id, axis_id = _report_histogram(repo)
    widget = AxisPropertiesWidget()
    try:
        widget.set_connected_figure(repo, figure_id, None)
        widget.rebuild_kwargs_editor(axis_id)
        editor = widget._kwargs_editor
        assert editor is not None and editor.value_for_key("distribution_fit") == "norm"  # shown as drawn
        editor.set_value_for_key("distribution_fit", "")  # the empty choice: no fit
        kwargs = widget.clean_kwargs()
        assert kwargs["distribution_fit"] == ""  # said explicitly, so the stored "norm" is not read
        options = {**(repo.get_axis_options(axis_id) or {}), "axis_kwargs": kwargs}
        assert HistogramAxisRenderer().opt("distribution_fit", options) == ""  # nothing to draw
    finally:
        widget.deleteLater()


def test_an_empty_value_with_nothing_stored_is_still_left_out(qapp, repo: SqliteRepo) -> None:
    figure_id = int(repo.create_figure_descriptor(name="Plain"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Histogram",
                                              title="", x_label="", y_label="", options={}))
    widget = AxisPropertiesWidget()
    try:
        widget.set_connected_figure(repo, figure_id, None)
        widget.rebuild_kwargs_editor(axis_id)
        assert widget._kwargs_editor is not None
        widget._kwargs_editor.set_value_for_key("distribution_fit", "")
        assert "distribution_fit" not in widget.clean_kwargs()
    finally:
        widget.deleteLater()
