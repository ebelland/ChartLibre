"""The whole project as one report: every figure, its notes, tables and history.

What goes to a supervisor, a reviewer or a lab notebook: each figure drawn
again at a fixed size, with the notes written beside it; then the tables,
with their sizes and notes; then what was done to them - the operation
history (todo R-03). No SQL: the report is for readers of the results, not
of the queries. One HTML file with the pictures inside it, or a PDF made
from the same page (app/dialogs/project_report_dialog.py). The same report
of one figure alone is the chart's own "Export report".

The pictures are named ``figure-<id>.png`` (or ``.svg``) in the page and
returned beside it, so the HTML writer can put them inline and the PDF
writer can hand them to Qt. No Qt here (todo R-09).
"""
from __future__ import annotations

import base64
from collections.abc import Sequence
import html
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from app.charts.render_figure import render_figure_from_descriptor
from app.utils import report_html
from app.utils.i18n import _

if TYPE_CHECKING:
    from app.data.sqlite_repo import SqliteRepo

#: The size each figure is drawn at, in inches: a page's text width.
REPORT_FIGURE_SIZE_IN: tuple[float, float] = (6.5, 4.0)


@dataclass(slots=True)
class ProjectReport:
    """The report page and the pictures it names."""

    title: str
    html: str
    #: Picture name (as written in the page) -> its bytes.
    images: dict[str, bytes] = field(default_factory=dict)
    #: Figures that could not be drawn, by name, with the reason.
    failures: list[tuple[str, str]] = field(default_factory=list)

    def inline_html(self) -> str:
        """The page with every picture inside it, as a data URI: one file to send."""
        page = self.html
        for name, data in self.images.items():
            kind = "image/svg+xml" if name.endswith(".svg") else "image/png"
            page = page.replace(f'src="{name}"', f'src="data:{kind};base64,{base64.b64encode(data).decode("ascii")}"')
        return page


def render_figure_bytes(
    repo: SqliteRepo, figure_id: int, *, image_format: str = "png", dpi: int = 150,
    size_in: tuple[float, float] = REPORT_FIGURE_SIZE_IN,
) -> bytes:
    """Draw figure *figure_id* off screen and return it as PNG or SVG bytes."""
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    if descriptor is None:
        raise ValueError(f"No figure with id {figure_id}.")
    figure = Figure(figsize=size_in, dpi=dpi)
    render_figure_from_descriptor(figure=figure, descriptor=descriptor, repo=repo)
    buffer = io.BytesIO()
    figure.savefig(buffer, format=image_format, dpi=dpi, facecolor="white")
    return buffer.getvalue()


def _figure_notes(repo: SqliteRepo, figure_id: int) -> str:
    view = repo.get_figure_options(figure_id).get("view")
    return str(view.get("notes_html") or "") if isinstance(view, dict) else ""


def _new_page(block: str) -> str:
    """*block* starting a page when printed - the PDF, or a browser's Print -
    so a figure is never split from its title. Nothing changes on screen."""
    return f"<div style='page-break-before:always;'>{block}</div>"


def _local_time(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return html.escape(stamp)


def build_project_report(
    repo: SqliteRepo,
    *,
    title: str = "",
    image_format: str = "png",
    dpi: int = 150,
    include_tables: bool = True,
    include_history: bool = True,
    figure_ids: Sequence[int] | None = None,
) -> ProjectReport:
    """Build the report of the project open in *repo*.

    *image_format* "png" or "svg" (SVG keeps the figures vector in the HTML;
    a PDF needs PNG). *figure_ids* limits the report to those figures - one,
    for a chart's own "Export report", which is then titled after it and
    has no list of figures. A figure that fails to draw is named in the
    report and in ``failures`` rather than stopping it.
    """
    project = Path(str(repo.db_path)).stem if getattr(repo, "db_path", None) else "Project"
    figures = repo.load_figures_from_db()
    if figure_ids is not None:
        wanted = {int(figure_id) for figure_id in figure_ids}
        figures = [(figure_id, name) for figure_id, name in figures if int(figure_id) in wanted]
    single = figure_ids is not None and len(figures) == 1
    report = ProjectReport(title=title or (figures[0][1] if single else project), html="")
    sections: list[str] = []

    # Who made it, when, and what it is - the project's own record of itself.
    info = repo.project_info() if hasattr(repo, "project_info") else {}
    about = [(html.escape(label), html.escape(value)) for label, value in (
        (_("Author"), info.get("author", "")),
        (_("Created"), _local_time(info.get("created", "")) if info.get("created") else ""),
    ) if value]
    about_blocks: list[str] = []
    if about:
        about_blocks.append(report_html.table([_("Project"), ""], about, align=["left", "left"], wrap=True))
    if info.get("notes") and not single:
        notes_html = html.escape(info["notes"]).replace("\n", "<br>")
        about_blocks.append(f"<p style='margin:8px 0 0 0;'>{notes_html}</p>")
    if not single:
        from app.data.repo.project_info import parse_references

        items = []
        for entry in parse_references(info.get("references")):
            line = html.escape(entry["citation"])
            if entry.get("doi"):
                doi = html.escape(entry["doi"])
                line += f" <a href='https://doi.org/{doi}'>doi:{doi}</a>"
            if entry.get("url"):
                url = html.escape(entry["url"], quote=True)
                line += f" <a href='{url}'>{html.escape(entry['url'])}</a>"
            items.append(f"<li>{line}</li>")
        if items:
            about_blocks.append(f"<p style='margin:10px 0 2px 0;'><b>{html.escape(_('References'))}</b></p>"
                                f"<ol style='margin:0 0 6px 18px;'>{''.join(items)}</ol>")
    if about_blocks:
        sections.append(report_html.section(_("About this project"), *about_blocks))

    contents = "".join(
        f"<li>{html.escape(name)}</li>" for _figure_id, name in figures
    )
    if contents and not single:
        sections.append(report_html.section(_("Figures"), f"<ol style='margin:0 0 6px 18px;'>{contents}</ol>"))

    for number, (figure_id, name) in enumerate(figures, start=1):
        picture = f"figure-{figure_id}.{image_format}"
        try:
            report.images[picture] = render_figure_bytes(repo, figure_id, image_format=image_format, dpi=dpi)
            image = f'<img src="{picture}" width="620" style="max-width:100%;" alt="{html.escape(name)}">'
        except Exception as exc:  # noqa: BLE001 - one broken figure must not cost the report
            report.failures.append((name, str(exc) or type(exc).__name__))
            image = report_html.note(_("This figure could not be drawn: {reason}").format(reason=exc))
        blocks = [f"<p style='margin:0 0 8px 0;'>{image}</p>"]
        descriptor = repo.get_figure_descriptor(figure_id) if hasattr(repo, "get_figure_descriptor") else None
        written = str(getattr(descriptor, "note", "") or "")
        if written.strip():
            blocks.append(f"<p style='margin:0 0 8px 0;'>{html.escape(written).replace(chr(10), '<br>')}</p>")
        notes = _figure_notes(repo, figure_id)
        if notes.strip():
            blocks.append(f"<div style='margin:0 0 8px 0;'>{notes}</div>")
        if single:
            sections.append("".join(blocks))
        else:
            sections.append(_new_page(report_html.section(_("Figure {number}. {name}").format(number=number, name=name), *blocks)))

    if include_tables:
        tables = repo.list_user_tables()
        rows = []
        for _index, row in tables.iterrows():
            table = str(row["Table"])
            try:
                size = f"{repo.row_count(table):,} × {len(repo.get_columns(table))}"
            except (OSError, ValueError, LookupError) as exc:
                size = f"? ({exc})"
            notes = row.get("Notes")
            rows.append((html.escape(table), size, html.escape(notes) if isinstance(notes, str) else ""))
        if rows:
            sections.append(_new_page(report_html.section(_("Tables"), report_html.table([_("Table"), _("Rows × columns"), _("Notes")], rows, align=["left", "right", "left"], wrap=True))))

    if include_history:
        records = repo.operations()
        if records:
            rows = [
                (_local_time(record.applied_at), html.escape(record.operation),
                 html.escape(", ".join(str(source.get("name") or "") for source in record.sources)) or "-",
                 html.escape(", ".join(str(result.get("table") or "") for result in record.results)) or "-")
                for record in records
            ]
            sections.append(report_html.section(
                _("Operation history"), report_html.table([_("Applied"), _("Operation"), _("Read"), _("Wrote")], rows, align=["left"] * 4, wrap=True)
            ))

    subtitle = _("{count} figure(s), {date}").format(count=len(figures), date=datetime.now().strftime("%Y-%m-%d %H:%M"))
    if report.failures:
        subtitle += " - " + _("{count} could not be drawn").format(count=len(report.failures))
    report.html = report_html.document(report.title, subtitle, *sections)
    return report


def write_report_html(report: ProjectReport, path: Path | str) -> Path:
    """Write *report* as one self-contained HTML file."""
    target = Path(path)
    if target.suffix.lower() not in (".html", ".htm"):
        target = target.with_name(target.name + ".html")
    target.write_text(report.inline_html(), encoding="utf-8")
    return target
