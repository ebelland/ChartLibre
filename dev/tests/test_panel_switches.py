"""The SHOW_* and FROM_* switches: an operation turns a panel or a data source off, and the window follows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.data_operations.outlier_dialog import SeriesOutlierDialog
from app.data_operations.time_series_dialog import TimeSeriesDialog
from app.utils.i18n import _


def _tab_titles(tabs) -> set[str]:
    return {tabs.tabText(i) for i in range(tabs.count()) if tabs.isTabVisible(i)}


def _chart_and_table(repo: SqliteRepo) -> int:
    figure_id = int(repo.create_figure_descriptor(name="chart"))
    repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                title="t", x_label="x", y_label="y", options={})
    t = np.linspace(0, 20, 200)
    repo.import_dataframe(pd.DataFrame({"time": t, "signal": np.sin(t)}), table_name="measures",
                          normalize_columns=False)
    return figure_id


def test_a_data_operation_shows_only_the_panels_it_switches_on(qapp, repo: SqliteRepo) -> None:
    class WithoutChartsOrModel(SeriesOutlierDialog):
        SHOW_CHART = False
        SHOW_MODEL = False

    dialog = WithoutChartsOrModel(repo=repo, figure_id=_chart_and_table(repo), parent=None)
    try:
        titles = _tab_titles(dialog.left_tabs)
        assert _("Data") in titles and _("Parameters") in titles
        assert _("Charts") not in titles and _("Model") not in titles
    finally:
        dialog.close()


def test_a_switched_off_source_is_not_offered(qapp, repo: SqliteRepo) -> None:
    class ChartOnly(SeriesOutlierDialog):
        FROM_DATA = False

    dialog = ChartOnly(repo=repo, figure_id=_chart_and_table(repo), parent=None)
    try:
        assert dialog.table_source is None  # no table source, so no choice between the two
    finally:
        dialog.close()


def test_a_switched_off_chart_source_leaves_the_table_source(qapp, repo: SqliteRepo) -> None:
    class TableOnly(SeriesOutlierDialog):
        FROM_CHART = False

    dialog = TableOnly(repo=repo, figure_id=_chart_and_table(repo), parent=None)
    try:
        assert dialog.source_chart_radio.isHidden()
        assert dialog.source_table_radio.isChecked()
    finally:
        dialog.close()


def test_a_table_operation_hides_its_switched_off_tabs(qapp, repo: SqliteRepo) -> None:
    repo.import_dataframe(pd.DataFrame({"giorno": range(10), "y": np.arange(10.0)}), table_name="line",
                          normalize_columns=False)

    class NoParameters(TimeSeriesDialog):
        SHOW_PARAMETERS = False

    dialog = NoParameters(repo=repo, table="line")
    try:
        titles = _tab_titles(dialog.left_tabs)
        assert _("Data") in titles
        assert _("Parameters") not in titles
        assert dialog.left_tabs.currentIndex() == 0  # opens on Data
    finally:
        dialog.close()
