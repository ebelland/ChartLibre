"""Graph templates: the look of one figure, put on another.

A paper's figures should look like one another - the same style sheet,
fonts, grid, tick directions, marker per series - and setting all of it by
hand on each figure is how they end up not quite matching. A template
keeps the *look* of a figure and none of its *content*: the figure's style
sheet and layout, each axis' options except its title, labels, limits and
annotations, and each series' colour, marker and line except its data.
Applied to another figure, axis by axis and series by series in order, it
changes how the figure looks and nothing it says.

Templates are JSON files in ``user/templates``, one per name, kept with the
user's other plugins rather than in a project, so one template serves every
project. No Qt here (todo R-09).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.utils.config import USER_CONTENT_DIR

if TYPE_CHECKING:
    from app.data.descriptors import FigureDescriptor
    from app.data.sqlite_repo import SqliteRepo

TEMPLATES_DIR: Path = USER_CONTENT_DIR / "templates"

TEMPLATE_VERSION: int = 1

#: Figure options that are the figure's look.
FIGURE_LOOK_KEYS: frozenset[str] = frozenset({"mpl_style", "layout_mode", "margins", "preserve_aspect_box", "font_scale"})

#: Axis options that say what *this* axis shows rather than how: never copied.
AXIS_CONTENT_KEYS: frozenset[str] = frozenset({
    "title", "label", "suptitle", "x_label", "y_label", "z_label",
    "xlim", "ylim", "zlim", "annotations", "lines", "reference_lines", "measurements",
    "twin_of", "col_span", "row_span", "sharex", "sharey", "sharez", "subplot",
    "left", "right", "top", "bottom", "axis_id", "figure_id", "renderer", "renderer_name", "problem",
})

#: Series style keys every renderer reads: the rest come from the renderer's schema.
SERIES_LOOK_KEYS: frozenset[str] = frozenset({
    "color", "marker", "markersize", "markerfacecolor", "markeredgecolor", "markeredgewidth",
    "linestyle", "linewidth", "drawstyle", "alpha", "show_in_legend", "zorder", "hatch",
    "edgecolor", "facecolor", "cmap",
})


#: Series keys that are the series' own even when its renderer declares them:
#: its legend label and whether it is shown at all.
SERIES_CONTENT_KEYS: frozenset[str] = frozenset({"label", "visible"})


def _renderer_keys(chart_type: str) -> frozenset[str]:
    """The keys a chart type's renderer declares: its look, per series as per axis."""
    from app.scanners.axis_renderer_scanner import get_renderer, import_class_from_file

    entry = get_renderer(chart_type)
    if entry is None:
        return frozenset()
    try:
        renderer = import_class_from_file(entry)
    except (ImportError, OSError, SyntaxError, AttributeError):
        return frozenset()
    return frozenset({*getattr(renderer, "Kwargs", {}), *getattr(renderer, "Options", {})})


def series_look(style: dict[str, Any], chart_type: str) -> dict[str, Any]:
    """The part of a series' style that is its look."""
    keys = (SERIES_LOOK_KEYS | _renderer_keys(chart_type)) - SERIES_CONTENT_KEYS
    return {key: value for key, value in (style or {}).items() if key in keys}


def axis_look(options: dict[str, Any]) -> dict[str, Any]:
    """An axis' options without the ones that are its content."""
    return {key: value for key, value in (options or {}).items() if key not in AXIS_CONTENT_KEYS}


def template_from_figure(descriptor: FigureDescriptor) -> dict[str, Any]:
    """The template of a figure: its look, figure, axes and series."""
    figure_options = descriptor.options or {}
    return {
        "version": TEMPLATE_VERSION,
        "figure": {key: value for key, value in figure_options.items() if key in FIGURE_LOOK_KEYS},
        "axes": [
            {
                "chart_type": axis.name,
                "options": axis_look(axis.options or {}),
                "series": [series_look(series.options or {}, axis.name) for series in sorted(axis.series, key=lambda s: s.index)],
            }
            for axis in sorted(descriptor.axes, key=lambda a: a.index)
        ],
    }


def _template_axis(template: dict[str, Any], chart_type: str, same_type_position: int, index: int) -> dict[str, Any] | None:
    """The template axis for a figure's axis: the one of its chart type, in
    order among those, else the one in its position (the last when fewer)."""
    axes = [axis for axis in template.get("axes", []) if isinstance(axis, dict)]
    same = [axis for axis in axes if axis.get("chart_type") == chart_type]
    if same:
        return same[min(same_type_position, len(same) - 1)]
    return axes[min(index, len(axes) - 1)] if axes else None


def apply_template(repo: SqliteRepo, figure_id: int, template: dict[str, Any]) -> int:
    """Give figure *figure_id* the look in *template*; return how many items changed.

    The figure's style sheet and layout are replaced by the template's. Each
    axis takes the options of the template axis of its chart type (in
    order; else the one in its position, minus anything only that chart type
    reads) and keeps its own title, labels, limits and annotations. Each
    series takes the look of the template series in its position on that axis.
    The caller snapshots for undo first.
    """
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    if descriptor is None:
        raise ValueError(f"No figure with id {figure_id}.")
    changed = 0
    look = template.get("figure") if isinstance(template.get("figure"), dict) else {}
    if look:
        options = {key: value for key, value in (descriptor.options or {}).items() if key not in FIGURE_LOOK_KEYS}
        options.update(look)
        repo.set_figure_options(figure_id, options)
        changed += 1

    by_type: dict[str, int] = {}
    for index, axis in enumerate(sorted(descriptor.axes, key=lambda a: a.index)):
        position = by_type.get(axis.name, 0)
        by_type[axis.name] = position + 1
        source = _template_axis(template, axis.name, position, index)
        if source is None:
            continue
        same_type = source.get("chart_type") == axis.name
        foreign = set() if same_type else _renderer_keys(str(source.get("chart_type") or "")) - _renderer_keys(axis.name)
        new_options = {key: value for key, value in (axis.options or {}).items() if key in AXIS_CONTENT_KEYS}
        new_options.update({key: value for key, value in dict(source.get("options") or {}).items() if key not in foreign})
        if not same_type and "projection" in (axis.options or {}):
            new_options["projection"] = (axis.options or {})["projection"]
        elif not same_type:
            new_options.pop("projection", None)
        repo.set_axis_options(int(axis.id), new_options)
        changed += 1

        styles = [style for style in source.get("series", []) if isinstance(style, dict)]
        # Series past the template's own keep their look.
        for series, style in zip(sorted(axis.series, key=lambda s: s.index), styles):
            wanted = series_look(style, axis.name)
            current = dict(series.options or {})
            kept = {key: value for key, value in current.items() if key not in series_look(current, axis.name)}
            repo.update_series_style(int(series.id), {**kept, **wanted})
            changed += 1
    return changed


# ----------------------------------------------------------------------
# The files
# ----------------------------------------------------------------------
def _file_name(name: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "_", name.strip()).strip() or "template"
    return f"{stem}.json"


def list_templates(directory: Path | None = None) -> list[str]:
    """The saved templates' names, sorted."""
    folder = directory or TEMPLATES_DIR
    names = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            names.append(str(json.loads(path.read_text(encoding="utf-8")).get("name") or path.stem))
        except (OSError, ValueError, AttributeError):
            continue
    return sorted(names, key=str.casefold)


def save_template(name: str, template: dict[str, Any], directory: Path | None = None) -> Path:
    """Write *template* as *name*, replacing a template of that name."""
    folder = directory or TEMPLATES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _file_name(name)
    path.write_text(json.dumps({**template, "name": name.strip()}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_template(name: str, directory: Path | None = None) -> dict[str, Any]:
    """Read the template called *name*. Raises OSError or ValueError if it cannot."""
    folder = directory or TEMPLATES_DIR
    path = folder / _file_name(name)
    if not path.exists():
        for candidate in folder.glob("*.json"):
            try:
                if json.loads(candidate.read_text(encoding="utf-8")).get("name") == name:
                    path = candidate
                    break
            except (OSError, ValueError, AttributeError):
                continue
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("axes", []), list):
        raise ValueError(f"{path.name} is not a graph template.")
    return data


def delete_template(name: str, directory: Path | None = None) -> None:
    (directory or TEMPLATES_DIR).joinpath(_file_name(name)).unlink(missing_ok=True)
