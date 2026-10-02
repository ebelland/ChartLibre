"""A figure exported for a journal: the column's width, real fonts, vector output.

"Save as picture" saves what is on screen, at whatever size the window gave
it; a journal wants something else. A figure exactly one column (85 mm) or
two (180 mm) wide, so it is printed at 100% and its text stays the size it
was set; text of 7-9 pt at that size; and a vector file - PDF or EPS - whose
fonts are embedded as fonts (TrueType, "Type 42"), not drawn as outlines or
substituted by the typesetter. A raster file, where asked for, at 300-600 dpi
and in TIFF, which most submission systems still prefer.

So the figure is drawn again, off screen, at the size and font size asked for -
not scaled from the window - and saved without "tight" cropping, which would
change the width just chosen. No Qt here (todo R-09).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from matplotlib import rc_context
from matplotlib.figure import Figure

from app.charts.render_figure import render_figure_from_descriptor

if TYPE_CHECKING:
    from app.data.sqlite_repo import SqliteRepo

MM_PER_INCH: float = 25.4

#: Journal column widths, as (label, millimetres). 85 and 180 mm are Elsevier's,
#: Nature's 89 and 183 round to them, and Science's 1.5 columns is about 120.
JOURNAL_WIDTHS_MM: tuple[tuple[str, float], ...] = (
    ("Single column", 85.0),
    ("One and a half columns", 120.0),
    ("Double column", 180.0),
)

#: Format -> (extension, file dialog filter). Vector formats first.
PUBLICATION_FORMATS: dict[str, tuple[str, str]] = {
    "PDF": ("pdf", "PDF Document (*.pdf)"),
    "EPS": ("eps", "Encapsulated PostScript (*.eps)"),
    "SVG": ("svg", "SVG Vector (*.svg)"),
    "TIFF": ("tiff", "TIFF Image (*.tif *.tiff)"),
    "PNG": ("png", "PNG Image (*.png)"),
}
VECTOR_FORMATS: frozenset[str] = frozenset({"PDF", "EPS", "SVG"})

#: Fonts as fonts: TrueType embedded in PDF and PostScript, text as text in SVG.
EMBEDDED_FONTS_RC: dict[str, Any] = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}


@dataclass(frozen=True, slots=True)
class PublicationSettings:
    """What a journal figure is exported as."""

    width_mm: float = 85.0
    height_mm: float = 64.0
    #: The text size, in points, at the printed size.
    font_size_pt: float = 8.0
    format: str = "PDF"
    #: Resolution of a raster format; vector formats ignore it.
    dpi: int = 600

    @property
    def extension(self) -> str:
        return PUBLICATION_FORMATS[self.format][0]


def publication_style(settings: PublicationSettings) -> str:
    """The rc lines appended to a figure's own style for the export.

    Appended, so they win over the same keys in the figure's style and leave
    everything else - colours, line widths, the figure's look - as it is. The
    font size is the one the journal asks for; titles a step above it.
    """
    size = settings.font_size_pt
    return "\n".join((
        f"font.size: {size:g}",
        f"axes.titlesize: {size + 1:g}",
        f"axes.labelsize: {size:g}",
        f"xtick.labelsize: {size - 1:g}",
        f"ytick.labelsize: {size - 1:g}",
        f"legend.fontsize: {size - 1:g}",
        f"figure.titlesize: {size + 2:g}",
        "savefig.bbox: standard",
    ))


def with_overrides(style: str, overrides: str) -> str:
    """*style* with *overrides* appended, and its own lines for those keys dropped.

    Dropped rather than left above: Matplotlib warns about a key set twice.
    """
    keys = {line.split(":", 1)[0].strip() for line in overrides.splitlines() if ":" in line}
    kept = [line for line in style.splitlines() if line.split("#", 1)[0].split(":", 1)[0].strip() not in keys]
    return "\n".join([*kept, overrides]) + "\n"


def render_for_publication(repo: SqliteRepo, figure_id: int, settings: PublicationSettings) -> Figure:
    """Draw figure *figure_id* off screen at the size and font size of *settings*."""
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    if descriptor is None:
        raise ValueError(f"No figure with id {figure_id}.")
    options = dict(descriptor.options or {})
    options["mpl_style"] = with_overrides(str(options.get("mpl_style") or ""), publication_style(settings))
    figure = Figure(figsize=(settings.width_mm / MM_PER_INCH, settings.height_mm / MM_PER_INCH), dpi=settings.dpi)
    render_figure_from_descriptor(figure=figure, descriptor=replace(descriptor, options=options), repo=repo)
    return figure


def export_publication_figure(repo: SqliteRepo, figure_id: int, path: Path | str, settings: PublicationSettings) -> Path:
    """Write figure *figure_id* to *path* as *settings* describe; return the path written.

    The extension is the format's when *path* has another one.
    """
    if settings.format not in PUBLICATION_FORMATS:
        raise ValueError(f"Unknown format {settings.format!r}.")
    target = Path(path)
    accepted = {settings.extension, "tif"} if settings.format == "TIFF" else {settings.extension}
    if target.suffix.lower().lstrip(".") not in accepted:
        target = target.with_name(target.name + "." + settings.extension)
    figure = render_for_publication(repo, figure_id, settings)
    extra: dict[str, Any] = {}
    if settings.format == "TIFF":
        extra["pil_kwargs"] = {"compression": "tiff_lzw"}
    with rc_context(EMBEDDED_FONTS_RC):
        figure.savefig(str(target), format=settings.extension, dpi=settings.dpi, bbox_inches=None,
                       facecolor="white" if settings.format in ("EPS", "TIFF") else figure.get_facecolor(), **extra)
    return target
