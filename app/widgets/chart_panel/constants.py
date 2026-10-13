"""The chart panel's constants, and axis_text."""
from __future__ import annotations

from typing import Any, Final

from PySide6.QtCore import QSize

from app.widgets import mpl_cursors  # noqa: F401 - native cursors on macOS, before any canvas

from app.utils.config import get_constant

TOOLBAR_ICON_SIZE: Final[QSize] = QSize(*get_constant("toolbar_icon_size", [24, 24]))
TOOLBAR_ACTIONS_TO_REMOVE: Final[set[str]] = {"Save","Subplots", "Customize", "Configure subplots", "Edit axis", "Edit colors"}
DEFAULT_ASPECT_RATIO: Final[float] = get_constant("default_aspect_ratio", 16.0 / 9.0)
CONFIG_SECTION: Final[str] = "chart_panel"
CONFIG_RESIZE_MODE: Final[str] = "resize_mode"
CONFIG_MIN_ZOOM: Final[str] = "min_zoom_percent"
CONFIG_MAX_ZOOM: Final[str] = "max_zoom_percent"
CONFIG_INITIAL_ZOOM: Final[str] = "initial_zoom_percent"
CONFIG_BACKGROUND_COLOR: Final[str] = "background_color"
CONFIG_COPY_DPI: Final[str] = "copy_dpi"
FIGURE_VIEW_OPTIONS_KEY: Final[str] = "view"
RESIZE_MODES: Final[tuple[str, ...]] = ("FIT", "FIT_PROPORTIONAL", "FIXED")

#: (key, label, tooltip), in the order every picker offers them. Shared so the
#: Figure Properties combo - the only place this choice is made now - cannot
#: drift from the mode names ChartPanel itself understands.
RESIZE_MODE_CHOICES: Final[tuple[tuple[str, str, str], ...]] = (
    ("FIT_PROPORTIONAL", "Fit proportional", "Fit chart keeping proportions"),
    ("FIT", "Fit", "Fit chart without keeping proportions"),
    ("FIXED", "Fixed", "Fixed size with zoom and scrollbars"),
)
CHART_AREA_MIN_HEIGHT: Final[int] = get_constant("chart_area_min_height", 160)
NOTES_MIN_HEIGHT: Final[int] = get_constant("notes_min_height", 0)
NOTES_FIRST_OPEN_MIN_HEIGHT: Final[int] = get_constant("notes_first_open_min_height", 80)
NOTES_INITIAL_FRACTION: Final[float] = get_constant("notes_initial_fraction", 0.3)

#: How near the cursor has to be, in points, for a marker to count as picked.
#: Five is about a default marker's own radius: close enough that a click has
#: to be deliberate, wide enough that it does not have to be exact.
PICK_TOLERANCE_POINTS: Final[float] = get_constant("pick_tolerance_points", 5.0)

#: FIXED mode's on-screen reference: the dpi used to turn a figure's physical
#: size in inches into logical widget pixels, at 100% zoom.  Deliberately not
#: the figure's own configured dpi - dpi is a print/export resolution, and
#: using it for on-screen sizing too meant a figure at a fixed 20x15cm looked
#: three times as big on nothing more than its dpi going from 100 to 300,
#: although Width and Height never changed.  100 matches Matplotlib's own
#: default figure.dpi, so a figure saved before per-figure dpi existed keeps
#: the on-screen size it always had.
FIXED_MODE_SCREEN_DPI: Final[float] = get_constant("fixed_mode_screen_dpi", 100.0)

#: How often "Live updates" re-reads and redraws the chart, in milliseconds.
#: A flat poll rather than a SQLite change hook: a hook fires per write, and
#: an import streaming rows in one-by-one would then redraw once per row -
#: this instead coalesces however many rows landed into one redraw per tick.
LIVE_REFRESH_INTERVAL_MS: Final[int] = get_constant("live_refresh_interval_ms", 2000)


def axis_text(axis: Any, value: float) -> str:
    """Format *value* the way *axis* would label it.

    Asking the axis rather than formatting the float ourselves is what makes a
    time series read ``2024-03-07`` instead of ``19789``: the renderers convert
    timestamps to epoch seconds before plotting, and only the axis formatter
    knows how to turn one back. The same call gives sensible precision on a log
    axis and on a percentage axis for free, and it is the call Matplotlib's own
    navigation toolbar makes for its cursor readout - so a picked point and the
    corner of the toolbar agree about how a number is written, including the
    trailing zeros ScalarFormatter pads out ("2.000", not "2").

    Formatters are third-party code as far as this panel is concerned - a
    renderer may install any callable - so a formatter that raises degrades to
    the plain number rather than losing the whole readout.
    """
    try:
        text = axis.get_major_formatter().format_data_short(value).strip()
    except Exception:
        text = ""
    return text or f"{value:g}"
