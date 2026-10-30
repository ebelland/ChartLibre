"""The Graphics page: the curve's table name, in the base for every data operation; a warning without the prefix."""
from __future__ import annotations

from app.data_operations.fit_dialog import SeriesFitDialog
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase


def _fit(repo) -> SeriesFitDialog:
    import numpy as np
    import pandas as pd

    x = np.linspace(0, 5, 40)
    repo.import_dataframe(pd.DataFrame({"x": x, "y": 2 * x + 1}), table_name="line", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="Fit")
    axis_id = repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                          title="line", x_label="x", y_label="y", options={})
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="s0",
                                  sql_query='SELECT "x", "y" FROM "line"', roles={"x": "x", "y": "y"}, style={})
    dialog = SeriesFitDialog(repo=repo, figure_id=figure_id, parent=None)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


def test_the_base_builds_the_output_name_for_every_operation(qapp, repo) -> None:
    assert DataOperationDialogBase.SHOW_OUTPUT is False  # no page unless an operation asks for one
    dialog = _fit(repo)
    try:
        assert SeriesFitDialog.SHOW_OUTPUT is True
        assert dialog._output_table_edit.text() == "_curve"
        assert dialog._output_table_name() == "_curve"
    finally:
        dialog.close()


def test_a_name_without_the_prefix_is_warned_and_still_written_as_generated(qapp, repo) -> None:
    dialog = _fit(repo)
    try:
        dialog._output_table_edit.setText("mycurve")
        assert not dialog._output_warning.isHidden()
        assert dialog._output_table_name() == "_mycurve"
        dialog._output_table_edit.setText("_mycurve")
        assert dialog._output_warning.isHidden()
        dialog._output_table_edit.setText("")
        assert dialog._output_warning.isHidden()
    finally:
        dialog.close()


def test_the_fit_window_has_a_graphics_tab(qapp, repo) -> None:
    from PySide6.QtWidgets import QTabWidget

    dialog = _fit(repo)
    try:
        titles = {tabs.tabText(i) for tabs in dialog.findChildren(QTabWidget) for i in range(tabs.count())}
        assert "Graphics" in titles
    finally:
        dialog.close()
