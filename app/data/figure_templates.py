"""Figure templates (G-18, I-08, H-03): a figure's layout and settings, saved to use again.

A template is a saved layout preset that also remembers what a preset does not:
the chart type at each position, each axis' settings (scales, limits, ticks,
breaks, labels, spans, sharing) and the figure's size, DPI, panel letters,
border and layout mode. It keeps no data (no series, queries or roles) and no
look: colours, fonts and lines are the figure's style (.mplstyle).

Templates are JSON files in ``Templates/`` at the application's root. Applying
one matches axes by position: the figure's first axis takes the template's
first, and so on; axes the template has and the figure lacks are added empty,
axes the figure has beyond the template's keep their settings and go after
them. A chart type is applied only where the axis' series have the columns it
needs - otherwise the axis keeps its type and a note says so.

No Qt: Figure properties calls this, and records the undo step around it.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.utils.config import TEMPLATES_DIR
from app.utils.figure_metrics import OPT_FIGURE_DPI, OPT_FIGURE_HEIGHT_CM, OPT_FIGURE_WIDTH_CM

VERSION = 1
SUFFIX = ".json"

#: The figure options a template carries: the layout and size, not the name, title or style.
FIGURE_KEYS: tuple[str, ...] = (
    "border", "font_scale", "layout_mode", "panel_labels", "panel_label_position", "margins",
    "downsample_threshold", "shared_vertical_space", "shared_horizontal_space",
    OPT_FIGURE_WIDTH_CM, OPT_FIGURE_HEIGHT_CM, OPT_FIGURE_DPI,
)
#: Axis options that are content or state, not settings: left out of a template.
AXIS_SKIPPED: frozenset[str] = frozenset({"generated_by", "source_name"})


def to_template(descriptor: Any, name: str = "") -> dict[str, Any]:
    """The template of a FigureDescriptor: its layout and settings, no data, no style."""
    options = dict(descriptor.options or {})
    axes = sorted(descriptor.axes, key=lambda axis: int(axis.index))
    position_of = {int(axis.id): int(axis.index) for axis in axes}
    out_axes = []
    for axis in axes:
        axis_options = {key: value for key, value in dict(axis.options or {}).items()
                        if key not in AXIS_SKIPPED and not key.startswith("_")}
        if axis_options.get("twin_of") is not None:
            # An axis id means nothing in another figure: kept as the position it points at.
            axis_options["twin_of"] = position_of.get(int(axis_options["twin_of"]))
        out_axes.append({"position": int(axis.index), "chart_type": str(axis.name or ""),
                         "title": axis.title or "", "x_label": axis.x_label or "",
                         "y_label": axis.y_label or "", "z_label": axis.z_label or "",
                         "options": axis_options})
    return {
        "version": VERSION, "name": name or str(descriptor.name or ""),
        "nrows": int(descriptor.nrows or 1), "ncols": int(descriptor.ncols or 1),
        "figure": {key: options[key] for key in FIGURE_KEYS if key in options},
        "axes": out_axes,
    }


def _fits(chart_type: str, series: list[Any]) -> bool:
    """Whether every series has the columns *chart_type* requires (an empty axis takes any type)."""
    if not series:
        return True
    from app.scanners.axis_renderer_scanner import get_renderer

    entry = get_renderer(chart_type)
    if entry is None:
        return False
    required = set(entry.get("required") or ())
    from app.data.data_source import parse_roles

    return all(required <= set(parse_roles(one.roles)) for one in series)


def apply_template(repo: Any, figure_id: int, template: Mapping[str, Any]) -> list[str]:
    """Lay figure *figure_id* out as *template* says; the notes on what could not be applied."""
    notes: list[str] = []
    descriptor = repo.load_figure_descriptor(int(figure_id))
    if descriptor is None:
        raise ValueError(f"figure {figure_id} not found")
    wanted = sorted(template.get("axes") or (), key=lambda axis: int(axis.get("position", 0)))
    existing = sorted(descriptor.axes, key=lambda axis: int(axis.index))
    nrows, ncols = max(1, int(template.get("nrows", 1))), max(1, int(template.get("ncols", 1)))

    # The template's axes onto the figure's, in order; the ones missing made empty.
    ids: list[int] = []
    for order, spec in enumerate(wanted):
        chart_type = str(spec.get("chart_type") or "")
        if order < len(existing):
            axis = existing[order]
            if chart_type and chart_type != axis.name:
                if _fits(chart_type, list(axis.series)):
                    repo.update_axis_chart_type(axis_id=int(axis.id), chart_type=chart_type)
                else:
                    notes.append(f"Axis {order + 1} kept {axis.name}: its series lack the columns "
                                 f"{chart_type} needs.")
            ids.append(int(axis.id))
        else:
            ids.append(int(repo.create_axis_descriptor(
                figure_id=int(figure_id), axis_index=10_000 + order, chart_type=chart_type or "Scatter Plot",
                title="", x_label="", y_label="", options={})))
    position_to_id = {int(spec.get("position", 0)): axis_id for spec, axis_id in zip(wanted, ids)}

    placements: list[tuple[int, int, dict[str, Any]]] = []
    for spec, axis_id in zip(wanted, ids):
        options = dict(spec.get("options") or {})
        if options.get("twin_of") is not None:
            options["twin_of"] = position_to_id.get(int(options["twin_of"]))
        current = repo.get_axis_options(axis_id)
        # The template's settings replace the axis' own; anything else of the axis' stays.
        overrides = {**{key: None for key in current if key not in options and key not in AXIS_SKIPPED
                        and not key.startswith("_") and key not in _KEPT_AXIS_KEYS}, **options}
        placements.append((axis_id, int(spec.get("position", 0)), overrides))
        repo.update_axis_descriptor(axis_id=axis_id, title=str(spec.get("title") or ""),
                                    x_label=str(spec.get("x_label") or ""), y_label=str(spec.get("y_label") or ""),
                                    z_label=str(spec.get("z_label") or ""))
    # The figure's extra axes after the template's, the grid grown to show them.
    extra = existing[len(wanted):]
    if extra:
        first = max((int(spec.get("position", 0)) for spec in wanted), default=-1) + 1
        for offset, axis in enumerate(extra):
            placements.append((int(axis.id), first + offset, {}))
        nrows = max(nrows, math.ceil((first + len(extra)) / ncols))
        notes.append(f"{len(extra)} axes beyond the template's were kept, after its own.")
    repo.apply_axis_layout(figure_id=int(figure_id), nrows=nrows, ncols=ncols, placements=placements)

    figure_options = dict(repo.get_figure_options(int(figure_id)) or {})
    for key in FIGURE_KEYS:
        figure_options.pop(key, None)
    figure_options.update(dict(template.get("figure") or {}))
    repo.set_figure_options(int(figure_id), figure_options)
    return notes


#: Axis options a template leaves to the axis even when it does not set them (what the series drawn there need).
_KEPT_AXIS_KEYS: frozenset[str] = frozenset({"label"})


def _file_name(name: str) -> str:
    stem = re.sub(r"[^\w\- .]+", "_", name, flags=re.UNICODE).strip(" .") or "template"
    return stem + SUFFIX


def save(template: Mapping[str, Any], folder: Path = TEMPLATES_DIR) -> Path:
    """Write *template* to <folder>/<its name>.json; the path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _file_name(str(template.get("name") or "template"))
    path.write_text(json.dumps(dict(template), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load(path: Path) -> dict[str, Any]:
    """A template read from *path*; a ValueError when it is not one."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{Path(path).name} is not a template: {exc}") from exc
    if not isinstance(data, dict) or "axes" not in data or int(data.get("version", 0)) > VERSION:
        raise ValueError(f"{Path(path).name} is not a template this version reads.")
    return data


def available(folder: Path = TEMPLATES_DIR) -> list[tuple[str, Path]]:
    """(name, path) of every template in *folder*, by name."""
    found = []
    for path in sorted(folder.glob(f"*{SUFFIX}")) if folder.is_dir() else []:
        try:
            found.append((str(load(path).get("name") or path.stem), path))
        except ValueError:
            continue
    return sorted(found, key=lambda item: item[0].lower())
