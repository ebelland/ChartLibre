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


def test_an_operation_without_models_has_no_empty_model_tab(qapp, repo: SqliteRepo) -> None:
    """Distribution, Normality, Change, In control: nothing to pick, so no Model tab - Outlier keeps its; and
    no Parameters tab where there is nothing to set."""
    from app.data_operations.change_dialog import ChangeDialog
    from app.data_operations.distribution_dialog import DistributionDialog
    from app.data_operations.in_control_dialog import InControlDialog
    from app.data_operations.normality_dialog import NormalityDialog

    figure_id = _chart_and_table(repo)
    for cls in (DistributionDialog, NormalityDialog, ChangeDialog, InControlDialog, SeriesOutlierDialog):
        dialog = cls(repo=repo, figure_id=figure_id, parent=None)
        try:
            titles = _tab_titles(dialog.left_tabs)
            assert (_("Model") in titles) == (cls is SeriesOutlierDialog), cls.__name__
            if cls in (NormalityDialog, ChangeDialog):  # no parameters either: no empty Parameters tab
                assert _("Parameters") not in titles, cls.__name__
        finally:
            dialog.close()


def test_a_roles_operation_does_not_draw_the_series_selector_over_its_corner(qapp, repo: SqliteRepo) -> None:
    """Fit Y by X on a chart: the engine's figure / axis combos are not on its Data page, and not shown."""
    from app.data_operations.fit_y_by_x_dialog import FitYByXDialog

    figure_id = _chart_and_table(repo)
    axis_id = repo.load_figure_descriptor(figure_id).axes[0].id
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="signal", style={},
                                  sql_query='SELECT time AS x, signal AS y FROM "measures"', roles={"x": "x", "y": "y"})
    dialog = FitYByXDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.show()
        qapp.processEvents()
        assert not dialog.series_selector.isVisible()
        dialog.current_figure_id = figure_id
        dialog.source_chart_radio.setChecked(True)
        # Read from the chart: its series is Y, its x is X - as drawn.
        assert dialog.roles_widget.casting().roles == {"response": ["signal"], "factor": ["x"]}
    finally:
        dialog.close()


def test_fit_y_by_x_on_a_scree_plot_draws_without_an_empty_table(qapp, repo: SqliteRepo, monkeypatch) -> None:
    """One value per level of an ordinal X: Tukey has no pair to compare and returns a table with no columns -
    written, "CREATE TABLE t ()" failed and nothing was drawn."""
    from app.data_operations.fit_y_by_x_dialog import FitYByXDialog
    from app.logs.logger import applogger

    failures: list[str] = []
    monkeypatch.setattr(applogger, "exception", lambda message, *a, **k: failures.append(str(message)))
    repo.import_dataframe(pd.DataFrame({"Number": [1, 2, 3, 4], "Eigenvalue": [2.5, 1.0, 0.4, 0.1]}),
                          table_name="scree", normalize_columns=False)
    figure_id = int(repo.create_figure_descriptor(name="Factor Analysis"))
    axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
                                              title="Scree plot", x_label="Number", y_label="Eigenvalue", options={}))
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="Eigenvalue", style={},
                                  sql_query='SELECT "Number" AS x, "Eigenvalue" AS y FROM "scree"',
                                  roles={"x": "x", "y": "y"})
    dialog = FitYByXDialog(repo=repo, figure_id=figure_id, parent=None)
    try:
        dialog.current_figure_id = figure_id
        dialog.source_chart_radio.setChecked(True)
        assert dialog.roles_widget.casting().roles == {"response": ["Eigenvalue"], "factor": ["Number"]}
        assert dialog.preview()
        for _step in range(20):
            qapp.processEvents()
        assert not failures
        assert dialog.created_figure_ids
    finally:
        dialog.close()


def test_a_report_figure_is_shown_on_preview_and_dropped_on_cancel(qapp, repo: SqliteRepo) -> None:
    """Draw on: New figure - Distribution's histogram and Q-Q figure is announced to the window on Preview (its tab
    shown), and announced gone when Cancel rolls it back."""
    from app.data_operations.distribution_dialog import DistributionDialog

    figure_id = _chart_and_table(repo)
    axis_id = repo.load_figure_descriptor(figure_id).axes[0].id
    repo.create_series_descriptor(axis_id=axis_id, series_index=0, name="signal", style={},
                                  sql_query='SELECT time AS x, signal AS y FROM "measures"', roles={"x": "x", "y": "y"})
    dialog = DistributionDialog(repo=repo, figure_id=figure_id, parent=None)
    announced: list[int] = []
    dialog.preview_drawn.connect(announced.append)
    try:
        assert dialog.preview()
        assert announced and announced[-1] != figure_id and repo.get_figure_descriptor(announced[-1]) is not None
        dialog.discard_unapplied()
        assert announced[-1] == -1
    finally:
        dialog.close()


def test_the_roles_data_page_is_on_the_tab_ground_not_a_darker_box(qapp, repo: SqliteRepo) -> None:
    """A scroll area fills the page it holds with the window's grey: the roles' Data page was a box in the page."""
    from app.data_operations.vector_field_dialog import VectorFieldDialog

    _chart_and_table(repo)
    dialog = VectorFieldDialog(repo=repo, table="measures", parent=None)
    try:
        assert not dialog._inputs_panel.autoFillBackground()
    finally:
        dialog.close()
