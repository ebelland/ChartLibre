"""Publication export (todo R-09): graph templates and the project report."""
from __future__ import annotations

import sys

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog

from app.data.sqlite_repo import SqliteRepo
from app.utils import graph_templates as gt
from app.utils.project_report import build_project_report, write_report_html
from dev.tests._figure_factory import create_renderer_showcase_db


@pytest.fixture(scope="module")
def showcase(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[SqliteRepo, dict[str, int]]]:
    path = tmp_path_factory.mktemp("publication") / "showcase.dhub"
    figure_ids = create_renderer_showcase_db(path, n_points=300)
    repo = SqliteRepo(db_path=path)
    yield repo, figure_ids
    repo.close()


# ----------------------------------------------------------------------
# Graph templates
# ----------------------------------------------------------------------
def _scatter_figure(repo: SqliteRepo, name: str, *, title: str, style: dict, axis_options: dict) -> int:
    figure_id = repo.create_figure_descriptor(name=name, nrows=1, ncols=1, options={})
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot", title=title, x_label="x", y_label="y",
        options={"title": title, **axis_options},
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="cloud", sql_query="SELECT x, y FROM src_xy", roles={}, style=style,
    )
    return figure_id


def test_a_template_carries_the_look_and_not_the_content(showcase, tmp_path: Path) -> None:
    repo, _ids = showcase
    source = _scatter_figure(
        repo, "source", title="Source title",
        style={"color": "#d62728", "marker": "^", "markersize": 9.0, "generated_fit": True, "source_name": "x"},
        axis_options={"grid": True, "x_scale": "log", "annotations": [{"text": "note"}], "xlim": [1, 10]},
    )
    options = repo.get_figure_options(source)
    options["mpl_style"] = "axes.linewidth: 0.5\n"
    repo.set_figure_options(source, options)
    template = gt.template_from_figure(repo.load_figure_descriptor(figure_id=source))  # pyright: ignore[reportArgumentType]
    axis = template["axes"][0]
    assert axis["chart_type"] == "Scatter Plot"
    assert axis["options"] == {"grid": True, "x_scale": "log"}
    assert axis["series"] == [{"color": "#d62728", "marker": "^", "markersize": 9.0}]
    assert template["figure"] == {"mpl_style": "axes.linewidth: 0.5\n"}

    gt.save_template("Paper look", template, tmp_path)
    assert gt.list_templates(tmp_path) == ["Paper look"]
    loaded = gt.load_template("Paper look", tmp_path)

    target = _scatter_figure(repo, "target", title="Target title", style={"marker": "o", "label": "keep me"},
                             axis_options={"xlim": [5, 6]})
    assert gt.apply_template(repo, target, loaded) == 3
    applied = repo.load_figure_descriptor(figure_id=target)
    assert applied is not None
    assert applied.options["mpl_style"] == "axes.linewidth: 0.5\n"  # pyright: ignore[reportOptionalSubscript]
    axis_options = applied.axes[0].options or {}
    assert axis_options["title"] == "Target title" and axis_options["xlim"] == [5, 6]
    assert axis_options["grid"] is True and "annotations" not in axis_options
    series = applied.axes[0].series[0]
    assert series.options == {"label": "keep me", "color": "#d62728", "marker": "^", "markersize": 9.0}
    assert series.sql_query.startswith("SELECT x, y FROM src_xy")
    gt.delete_template("Paper look", tmp_path)
    assert gt.list_templates(tmp_path) == []


def test_applying_from_the_chart_menu_can_be_undone(qapp, showcase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QInputDialog

    from app.widgets.chart_panel import ChartPanel

    repo, _ids = showcase
    source = _scatter_figure(repo, "menu source", title="S", style={"color": "#2ca02c"}, axis_options={"grid": True})
    target = _scatter_figure(repo, "menu target", title="T", style={}, axis_options={})
    monkeypatch.setattr(gt, "TEMPLATES_DIR", tmp_path)
    gt.save_template("Green", gt.template_from_figure(repo.load_figure_descriptor(figure_id=source)), tmp_path)  # pyright: ignore[reportArgumentType]
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *_a, **_k: ("Green", True)))
    panel = ChartPanel(repo, target)
    panel.apply_saved_template()
    assert repo.load_figure_descriptor(figure_id=target).axes[0].series[0].options["color"] == "#2ca02c"  # pyright: ignore[reportOptionalMemberAccess]
    assert repo.undo_last() is not None
    assert "color" not in (repo.load_figure_descriptor(figure_id=target).axes[0].series[0].options or {})  # pyright: ignore[reportOptionalMemberAccess]
    panel.close()


# ----------------------------------------------------------------------
# The project report
# ----------------------------------------------------------------------
def test_the_report_has_every_figure_and_no_sql(showcase, tmp_path: Path) -> None:
    repo, ids = showcase
    report = build_project_report(repo, dpi=60, include_history=False)
    assert not report.failures
    assert len(report.images) == len(repo.load_figures_from_db()) >= len(ids)
    assert "SELECT" not in report.html
    written = write_report_html(report, tmp_path / "report")
    page = written.read_text(encoding="utf-8")
    assert written.suffix == ".html" and page.count('src="data:image/png;base64,') == len(report.images)
    assert "src=\"figure-" not in page


def test_the_report_dialog_writes_a_pdf(qapp, showcase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtPdf import QPdfDocument

    from app.dialogs.project_report_dialog import ProjectReportDialog

    repo, _ids = showcase
    dialog = ProjectReportDialog(repo)
    dialog._format_combo.setCurrentIndex(1)
    dialog._dpi_spin.setValue(72)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: (str(tmp_path / "project.pdf"), "")))
    dialog._export()
    assert dialog.written == tmp_path / "project.pdf"
    document = QPdfDocument()
    document.load(str(dialog.written))
    # A contents page, then a page per figure (at least), then the tables.
    assert document.pageCount() >= len(repo.load_figures_from_db()) + 2


def test_no_wait_cursor_anywhere() -> None:
    """QGuiApplication.setOverrideCursor(WaitCursor) crashed the app on macOS 27
    (CGImageCreate refused the cursor's image): the export dialogs say what
    they are doing in their own text instead."""
    app_dir = Path(__file__).resolve().parents[2] / "app"
    offenders = [
        str(path.relative_to(app_dir)) for path in app_dir.rglob("*.py")
        if "setOverrideCursor(" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_matplotlib_asks_macos_for_native_cursors_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its wait cursor (around any draw after a pause) and its pan cursor are
    drawn by Qt from an image on macOS - the same crash, from Matplotlib."""
    from matplotlib.backends import backend_qt

    from app.widgets import mpl_cursors

    monkeypatch.setattr(backend_qt, "cursord", dict(backend_qt.cursord))
    mpl_cursors.use_native_cursors("linux")
    assert backend_qt.cursord[mpl_cursors.Cursors.WAIT] != mpl_cursors.Qt.CursorShape.ArrowCursor or sys.platform == "darwin"
    mpl_cursors.use_native_cursors("darwin")
    assert not set(backend_qt.cursord.values()) & mpl_cursors.IMAGE_CURSORS
    app_dir = Path(__file__).resolve().parents[2] / "app"
    offenders = [
        str(path.relative_to(app_dir)) for path in app_dir.rglob("*.py")
        if path.name != "mpl_cursors.py"
        and any(shape in path.read_text(encoding="utf-8") for shape in ("WaitCursor", "BusyCursor", "SizeAllCursor"))
    ]
    assert offenders == []


def test_a_chart_that_draws_while_it_is_built_is_still_drawn(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Text chart's adjustText draws the figure half-built to measure its
    labels; that draw used to pass for the finished one, and the canvas
    stayed blank."""
    import pandas as pd

    from app.data.sqlite_repo import SqliteRepo
    from app.main_window.main_window import MainWindow
    from app.widgets.chart_panel import ChartPanel

    repo = SqliteRepo(db_path=tmp_path / "t.dhub")
    repo.import_dataframe(pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [5.0, 7.0, 6.0], "label": ["a", "b", "c"]}),
                          table_name="points", normalize_columns=False)
    for name, chart, roles, sql in (
        ("scatter", "Scatter Plot", {"x": "x", "y": "y"}, "SELECT x, y FROM points"),
        ("labels", "Text", {"x": "x", "y": "y", "text": "label"}, "SELECT x, y, label AS text FROM points"),
    ):
        figure_id = int(repo.create_figure_descriptor(name=name))
        axis_id = int(repo.create_axis_descriptor(figure_id=figure_id, axis_index=0, chart_type=chart, title=name,
                                                  x_label="x", y_label="y", options={}))
        repo.create_series_descriptor(axis_id=axis_id, series_index=0, name=name, sql_query=sql, roles=roles, style={})
    # Every chart draws its figure before it is built, the way adjustText
    # does mid-build - deterministic, where the real case depends on timing.
    from app.widgets import chart_panel as chart_panel_module

    real_render = chart_panel_module.render_figure_from_descriptor

    def render_drawing_first(*, figure, descriptor, repo):
        figure.canvas.draw()
        return real_render(figure=figure, descriptor=descriptor, repo=repo)

    monkeypatch.setattr(chart_panel_module, "render_figure_from_descriptor", render_drawing_first)
    window = MainWindow(repo, tmp_path / "t.dhub")
    window.show()
    for index in range(window._tabs.count()):
        window._tabs.setCurrentIndex(index)
        for _ in range(60):
            qapp.processEvents()
        panel = window._tabs.widget(index)
        assert isinstance(panel, ChartPanel)
        image = panel.canvas.grab().toImage()
        colours = {image.pixel(x, y) for x in range(0, image.width(), 5) for y in range(0, image.height(), 5)}
        assert len(colours) > 2, window._tabs.tabText(index)
    window.close()
    repo.close()


def test_a_charts_own_report_has_that_chart_alone(qapp, showcase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.dialogs.project_report_dialog import ProjectReportDialog
    from app.widgets.chart_panel import ChartPanel

    repo, ids = showcase
    report = build_project_report(repo, dpi=60, figure_ids=[ids["Kaplan-Meier"]])
    assert list(report.images) == [f"figure-{ids['Kaplan-Meier']}.png"]
    assert report.title == "Kaplan-Meier showcase" and "Operation history" not in report.html

    panel = ChartPanel(repo, ids["Kaplan-Meier"])
    menu_texts = [action.text() for action in panel._build_actions_menu().actions()]
    assert "Save figure" in menu_texts and "Export report…" in menu_texts
    dialogs: list[ProjectReportDialog] = []
    monkeypatch.setattr(ProjectReportDialog, "exec", lambda self: dialogs.append(self) or 0)
    panel.export_report()
    assert dialogs and dialogs[0]._figure_id == ids["Kaplan-Meier"] and dialogs[0].windowTitle() == "Export report"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: (str(tmp_path / "km.html"), "")))
    dialogs[0]._format_combo.setCurrentIndex(0)  # HTML; the dialog remembers the last format
    dialogs[0]._export()
    page = (tmp_path / "km.html").read_text(encoding="utf-8")
    assert page.count("data:image/png;base64,") == 1
    panel.close()
