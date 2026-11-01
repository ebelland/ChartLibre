"""The result's name on the Charts tab, in the base for every data operation; a warning without the prefix."""
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


def test_the_fit_window_names_its_result_on_its_one_charts_tab(qapp, repo) -> None:
    """No Graphics tab of its own: the curve's name is the Charts tab's Result name, as every operation's."""
    dialog = _fit(repo)
    try:
        titles = [dialog.left_tabs.tabText(i) for i in range(dialog.left_tabs.count())]
        assert "Graphics" not in titles and titles.count("Charts") == 1
        assert dialog.left_tabs.widget(titles.index("Charts")).isAncestorOf(dialog._output_table_edit)
    finally:
        dialog.close()


def test_the_report_is_copied_and_exported_from_its_menu_not_a_button(qapp, repo, tmp_path, monkeypatch) -> None:
    """Right-click on the report: Copy, Export as HTML, Export as PDF - and no Copy button under it."""
    from app.widgets import html_results

    dialog = _fit(repo)
    try:
        assert dialog.copy_html_button is None
        view = dialog._results_label
        view.setHtml("<h1>Report</h1><p>Slope 2.0</p>")
        shown: list = []
        monkeypatch.setattr(html_results, "create_menu", lambda _parent, items, **_k: shown.extend(items) or
                            type("M", (), {"exec": lambda *_a: None})())
        view._show_context_menu(view.rect().center())
        assert [item.action_id for item in shown] == ["copy", "results_export_html", "results_export_pdf"]
        written = html_results.write_html_pdf("<h1>Report</h1><p>Slope 2.0</p>", tmp_path / "report")
        assert written.name == "report.pdf" and written.read_bytes().startswith(b"%PDF")
    finally:
        dialog.close()
