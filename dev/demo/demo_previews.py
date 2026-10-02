"""A picture of each demo project, for the Load demo dialog.

The dialog lists the demos by name, and a name says what a project is about
but not what it looks like - which is most of what someone choosing a demo
wants to know. Each preview is the project's first figure, drawn off screen
from the built .dhub exactly as the application draws it, into
demo/previews/<file name>.png, tracked beside the projects themselves.

Rebuilt by build_demos.py after the projects; on its own:

    python3 -m dev.demo.demo_previews
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from app.data.demos import DEMO_DIR, DEMO_PROJECTS, DemoProject
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.project_report import render_figure_bytes

#: Inches and dots per inch of a preview: 640 x 400 pixels, sharp at the
#: dialog's size on a 2x screen without weighing more than ~50 KB.
PREVIEW_SIZE_IN: tuple[float, float] = (6.4, 4.0)
PREVIEW_DPI: int = 100


def build_preview(demo: DemoProject, directory: Path = DEMO_DIR) -> Path | None:
    """Draw *demo*'s first figure into its preview; None if it has none.

    From a copy: opening a project can write to it (a repair on open, the
    undo store beside it), and the shipped file must stay as built.
    """
    source = Path(directory) / demo.path_name
    if not source.exists():
        applogger.warning("No demo project %s to preview.", source, show_dialog=False, raise_error=False)
        return None
    target = Path(directory) / "previews" / f"{demo.file_name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chartlibre-preview-") as scratch:
        copy = Path(scratch) / source.name
        shutil.copy(source, copy)
        repo = SqliteRepo(db_path=copy)
        try:
            figures = repo.load_figures_from_db()
            chosen = [figure for figure in figures if demo.preview and demo.preview in figure[1]] or figures
            if not chosen:
                return None
            figure_id = chosen[0][0]
            # A grid of axes gets more room and the same pixels: drawn larger,
            # at a lower resolution, so its titles do not run into each other.
            descriptor = repo.load_figure_descriptor(figure_id=figure_id)
            scale = 2 if descriptor is not None and len(descriptor.axes) > 1 else 1
            target.write_bytes(render_figure_bytes(
                repo, figure_id, image_format="png", dpi=PREVIEW_DPI // scale,
                size_in=(PREVIEW_SIZE_IN[0] * scale, PREVIEW_SIZE_IN[1] * scale),
            ))
        finally:
            repo.close()
    return target


def build_previews(directory: Path = DEMO_DIR) -> list[Path]:
    """A preview for every demo project in *directory*."""
    written = []
    for demo in DEMO_PROJECTS:
        path = build_preview(demo, directory)
        if path is not None:
            written.append(path)
    return written


if __name__ == "__main__":
    for preview in build_previews():
        print(preview)
