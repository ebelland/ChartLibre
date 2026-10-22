"""A project's preview: its first figure as a picture, beside the project.

Saving a project redraws it (MainWindow._on_save), and the demo set is built
with one (dev/demo/build_demos.py), which is what the Load demo dialog shows.
The picture is EMBEDDED in the project (``__project_preview__``), so a copy of
the project file alone carries it. Projects saved before that keep a
``<project>.preview.png`` beside them, which is still read when there is no
embedded picture (``project_preview_bytes``).

No Qt here.
"""
from __future__ import annotations

from pathlib import Path

from app.data.sqlite_repo import SqliteRepo
from app.utils.project_report import render_figure_bytes
from app.utils.config import get_constant

#: Inches and dots per inch of a preview: 1280 x 800 pixels at 200 dpi.
PREVIEW_SIZE_IN: tuple[float, float] = (6.4, 4.0)
PREVIEW_DPI: int = get_constant("project_preview_dpi", 200)

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


def project_preview_bytes(project: Path, entry: str | None = None) -> bytes | None:
    """The preview PNG of the project at *project*: the embedded one, else the legacy file.

    *entry* is the project's old ``preview_path`` info entry, when the caller has it.
    """
    from app.data.repo.project_info import read_project_info, read_project_preview

    embedded = read_project_preview(Path(project))
    if embedded:
        return embedded
    legacy = resolve_preview(Path(project), entry if entry is not None else read_project_info(Path(project)).get(PREVIEW_KEY))
    if legacy is None:
        return None
    try:
        return legacy.read_bytes()
    except OSError:
        return None


def write_project_preview(repo: SqliteRepo, *, figure_id: int | None = None) -> bytes | None:
    """Draw the project's first figure (or *figure_id*) into its embedded preview.

    Returns the PNG; None, with the preview removed, for a project with no figures.
    """
    figures = repo.load_figures_from_db()
    if figure_id is None:
        if not figures:
            repo.set_project_preview(None)
            repo.set_project_info({PREVIEW_KEY: None})
            return None
        figure_id = int(figures[0][0])
    png = render_preview(repo, int(figure_id))
    repo.set_project_preview(png)
    # The file entry of older projects is dropped: the picture is in the project now.
    repo.set_project_info({PREVIEW_KEY: None})
    return png
