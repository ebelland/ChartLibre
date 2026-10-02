"""Publication export (todo R-09): journal figures, graph templates, the project report."""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QFileDialog

from app.data.sqlite_repo import SqliteRepo
from app.utils import graph_templates as gt
from app.utils.project_report import build_project_report, write_report_html
from app.utils.publication import PublicationSettings, export_publication_figure, with_overrides
from dev.tests._figure_factory import create_renderer_showcase_db


@pytest.fixture(scope="module")
def showcase(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[SqliteRepo, dict[str, int]]]:
    path = tmp_path_factory.mktemp("publication") / "showcase.dhub"
    figure_ids = create_renderer_showcase_db(path, n_points=300)
    repo = SqliteRepo(db_path=path)
    yield repo, figure_ids
    repo.close()


# ----------------------------------------------------------------------
# A figure for a journal
# ----------------------------------------------------------------------
def test_a_pdf_is_exactly_the_column_wide_with_its_fonts_embedded(showcase, tmp_path: Path) -> None:
    repo, ids = showcase
    written = export_publication_figure(
        repo, ids["Scatter Plot"], tmp_path / "scatter", PublicationSettings(width_mm=85, height_mm=60, format="PDF")
    )
    assert written.name == "scatter.pdf"
    pdf = written.read_bytes()
    box = [float(value) for value in re.search(rb"/MediaBox \[([^\]]*)\]", pdf).group(1).split()]  # pyright: ignore[reportOptionalMemberAccess]
    assert box[2] == pytest.approx(85 / 25.4 * 72, abs=0.01) and box[3] == pytest.approx(60 / 25.4 * 72, abs=0.01)
    assert b"/FontFile2" in pdf and b"/Type3" not in pdf


def test_a_tiff_has_the_pixels_the_width_and_resolution_ask_for(showcase, tmp_path: Path) -> None:
    repo, ids = showcase
    written = export_publication_figure(
        repo, ids["Bar Chart"], tmp_path / "bars.tif",
        PublicationSettings(width_mm=180, height_mm=90, format="TIFF", dpi=300),
    )
    assert written.suffix == ".tif"
    with Image.open(written) as image:
        # Matplotlib truncates to whole pixels.
        assert image.size == (int(180 / 25.4 * 300), int(90 / 25.4 * 300))


@pytest.mark.parametrize("fmt", ["EPS", "SVG", "PNG"])
def test_every_format_writes(showcase, tmp_path: Path, fmt: str) -> None:
    repo, ids = showcase
    written = export_publication_figure(repo, ids["Kaplan-Meier"], tmp_path / "km", PublicationSettings(format=fmt, dpi=150))
    assert written.exists() and written.stat().st_size > 1000


def test_the_publication_style_replaces_the_keys_it_sets() -> None:
    merged = with_overrides("font.size: 12\nlines.linewidth: 2\nsavefig.bbox: tight  # crop", "font.size: 8\nsavefig.bbox: standard")
    assert merged.splitlines() == ["lines.linewidth: 2", "font.size: 8", "savefig.bbox: standard"]


def test_the_dialog_exports_with_the_figures_proportions(qapp, showcase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.dialogs.publication_export_dialog import PublicationExportDialog

    repo, ids = showcase
    dialog = PublicationExportDialog(repo, ids["Histogram"], aspect=0.5)
    dialog._width_preset_combo.setCurrentIndex(2)  # double column
    assert dialog._width_spin.value() == 180.0 and dialog._height_spin.value() == 90.0
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: (str(tmp_path / "hist.pdf"), "")))
    dialog._export()
    assert dialog.written is not None and dialog.written == tmp_path / "hist.pdf" and dialog.written.exists()


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
def test_the_report_has_every_figure_with_its_sql(showcase, tmp_path: Path) -> None:
    repo, ids = showcase
    report = build_project_report(repo, dpi=60, include_history=False)
    assert not report.failures
    assert len(report.images) == len(repo.load_figures_from_db()) >= len(ids)
    assert "SELECT weeks AS time" in report.html
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
