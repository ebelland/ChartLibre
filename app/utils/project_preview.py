"""A project's preview: its first figure as a picture, beside the project.

Saving a project redraws it (MainWindow._on_save), and the demo set is built
with one (dev/demo/build_demos.py), which is what the Load demo dialog shows.
The picture is ``<project>.preview.png`` in the project's folder; the
project's ``preview_path`` entry (``__project_info__``) names it relative to
the project, so the two can be moved together.

No Qt here.
"""
from __future__ import annotations

from pathlib import Path

from app.data.sqlite_repo import SqliteRepo
from app.utils.project_report import render_figure_bytes

#: Inches and dots per inch of a preview: 1280 x 800 pixels at 200 dpi.
PREVIEW_SIZE_IN: tuple[float, float] = (6.4, 4.0)
PREVIEW_DPI: int = 200

#: What the project-info entry is called.
PREVIEW_KEY = "preview_path"


def preview_file_for(project: Path) -> Path:
    """Where *project*'s preview goes: ``<name>.preview.png`` beside it."""
    project = Path(project)
    return project.with_name(f"{project.stem}.preview.png")


def resolve_preview(project: Path, entry: str | None) -> Path | None:
    """The preview file a ``preview_path`` *entry* names, if it exists."""
    if not entry:
        return None
    path = Path(entry)
    if not path.is_absolute():
        path = Path(project).parent / path
    return path if path.is_file() else None


def render_preview(repo: SqliteRepo, figure_id: int) -> bytes:
    """Figure *figure_id* drawn as a preview PNG.

    A figure with several axes is drawn twice as large at half the
    resolution - the same pixels - so its titles do not run into each other.
    """
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    scale = 2 if descriptor is not None and len(descriptor.axes) > 1 else 1
    return render_figure_bytes(
        repo, figure_id, image_format="png", dpi=PREVIEW_DPI // scale,
        size_in=(PREVIEW_SIZE_IN[0] * scale, PREVIEW_SIZE_IN[1] * scale),
    )


def write_project_preview(repo: SqliteRepo, *, figure_id: int | None = None) -> Path | None:
    """Draw the project's first figure (or *figure_id*) into its preview.

    Records it as the project's ``preview_path`` and returns the file
    written; None, with the entry removed, for a project with no figures.
    """
    project = Path(str(repo.db_path))
    figures = repo.load_figures_from_db()
    if figure_id is None:
        if not figures:
            repo.set_project_info({PREVIEW_KEY: None})
            return None
        figure_id = int(figures[0][0])
    target = preview_file_for(project)
    target.write_bytes(render_preview(repo, int(figure_id)))
    repo.set_project_info({PREVIEW_KEY: target.name})
    return target
