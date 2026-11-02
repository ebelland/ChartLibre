"""Wafer Map (Quality and Process): a semiconductor wafer's measurements or defects, on the wafer and per die.

A series of points in millimetres from the wafer's centre - x, y, and a value as its z when it has one (a
film's thickness at each site, a sheet resistance). With a value: the map as a contour over the wafer, each
die's mean, and the dies binned against the limits - the yield; the uniformity figures fabs quote, and the
means ring by ring from the centre. Without one, the points are defects (particles): drawn as they are, and
counted per die - the defect density and the yield it predicts.

The wafer: its diameter, its notch or flat and where it is, the edge exclusion. Rotation turns the data
about the centre first - a map read with the notch elsewhere brought to where the mark is drawn. The
engine is app.analysis.wafer, the chart app.charts.wafer_map.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from app.analysis import wafer as wf
from app.data.data_source import quote_identifier
from app.plugin_api.data_operations.dialog_base import DataOperationDialogBase
from app.plugin_api.data_operations.parameter_spec import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
    TextParam,
)
from app.plugin_api.data_operations.results import OperationResult
from app.utils import report_html
from app.utils.i18n import _

#: The mark's place, as the Wafer tab names it: degrees counterclockwise from the right.
MARK_PLACES = (("Bottom", "270"), ("Left", "180"), ("Top", "90"), ("Right", "0"))


def _limit(text: Any) -> float | None:
    try:
        return float(str(text).strip()) if str(text or "").strip() else None
    except ValueError:
        return None


@dataclass(slots=True)
class WaferResult(OperationResult):
    """One series' wafer: its points (rotated, on a die or not), the grid of its contour, its dies."""

    source_name: str
    wafer: wf.Wafer
    points: pd.DataFrame
    dies: pd.DataFrame
    grid: pd.DataFrame
    zones: pd.DataFrame
    summary: dict[str, Any]
    die_size: tuple[float, float, float, float]
    has_value: bool
    has_limits: bool
    #: The tables the figure reads, named once (OK's undo step names them first).
    tables: dict[str, str] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        return self.dies

    def preview(self, dialog: WaferMapDialog, axis_id: int) -> None:
        del axis_id
        dialog.draw(self)

    def apply(self, dialog: WaferMapDialog, axis_id: int) -> None:
        del axis_id
        dialog.draw(self)


class WaferMapDialog(DataOperationDialogBase):
    """A wafer's map: contour, die means and bins of a measurement - or a defect map and its density."""

    Name = "Wafer Map"
    Description = ("A wafer's measurements or defects on the wafer and per die: contour map, die bins and yield, "
                   "defect density - notch or flat, any diameter, rotation.")
    Category = "Quality and Process"
    Order = 30
    Icon = """
    <path d="M12 3a9 9 0 1 0 1.2 .08L12 4.6z"/><path d="M7 8h10M7 12h10M7 16h10M9 6v12M12 4.6v14.4M15 6v12"/>
    """
    Keywords: str = (
        "wafer fetta semiconductor semiconduttori die binning resa yield notch flat defect difetti particle "
        "particelle defect density uniformity uniformità thickness spessore map mappa"
    )
    Goals: ClassVar[list[str]] = ["describe"]

    DESTINATIONS = (DataOperationDialogBase.DEST_NONE, DataOperationDialogBase.DEST_NEW_FIGURE)
    DEFAULT_DESTINATION = DataOperationDialogBase.DEST_NEW_FIGURE
    SELECTS_ALL_SERIES = True
    INPUT_MINIMUM_POINTS = 1

    PARAMETERS_TAB = "Wafer"
    PARAMS = (
        ChoiceParam("diameter", "Diameter:", choices=tuple((f"{d} mm", str(d)) for d in wf.DIAMETERS),
                    default_value="300", tooltip="The wafer's diameter."),
        ChoiceParam("mark", "Orientation mark:", choices=(("Notch", wf.NOTCH), ("Flat", wf.FLAT),
                                                         ("None", wf.NO_MARK)),
                    tooltip="A notch on 200 mm wafers and up, a flat on smaller ones."),
        ChoiceParam("mark_angle", "Mark at:", choices=MARK_PLACES, default_value="270",
                    tooltip="Where the notch or flat is drawn: the data is in the wafer's own frame, the mark "
                            "where the map shows it."),
        FloatParam("rotation", "Rotation:", default_value=0.0, minimum=-360.0, maximum=360.0, decimals=1,
                   step=90.0, tooltip="Turns the data counterclockwise about the centre, degrees: a map read "
                                      "with the notch elsewhere brought to the mark."),
        FloatParam("edge_exclusion", "Edge exclusion:", default_value=3.0, minimum=0.0, maximum=50.0, decimals=1,
                   tooltip="The ring at the edge left out, mm: its points count nowhere."),
        FloatParam("die_width", "Die width:", default_value=10.0, minimum=0.1, maximum=500.0, decimals=2,
                   tooltip="mm. The dies are the whole ones on the usable wafer."),
        FloatParam("die_height", "Die height:", default_value=10.0, minimum=0.1, maximum=500.0, decimals=2,
                   tooltip="mm."),
        FloatParam("die_offset_x", "Die grid shift x:", default_value=0.0, minimum=-500.0, maximum=500.0,
                   decimals=2, advanced=True, tooltip="The grid's shift from the centre, mm: half a die puts "
                                                      "a die corner on the centre."),
        FloatParam("die_offset_y", "Die grid shift y:", default_value=0.0, minimum=-500.0, maximum=500.0,
                   decimals=2, advanced=True, tooltip="mm."),
        TextParam("lower", "Lower limit:", placeholder="none",
                  tooltip="A measured die below it is binned Low; empty: no lower limit."),
        TextParam("upper", "Upper limit:", placeholder="none",
                  tooltip="A measured die above it is binned High; empty: no upper limit."),
        IntParam("grid_points", "Contour grid:", default_value=81, minimum=11, maximum=401, advanced=True,
                 tooltip="Points across the wafer the measurements are interpolated on."),
        BoolParam("chart_map", "Map (contour, or the defects)", default_value=True, tab="charts"),
        BoolParam("chart_dies", "Dies (means, or defects per die)", default_value=True, tab="charts"),
        BoolParam("chart_bins", "Die bins (with limits)", default_value=True, tab="charts"),
    )

    def read_series(self, row: Any, name: str) -> Any:
        """x, y and - when the series has a z - its value; else the points are defects."""
        try:
            return self.series_xyz(row, name)
        except ValueError:
            x, y = self.series_xy(row, name)
            return x, y, None

    def compute_series(self, name: str, data: Any, settings: Any) -> WaferResult:
        x, y, value = data
        params: Mapping[str, Any] = settings  # no MODELS: the parameters themselves
        wafer = wf.Wafer(diameter=float(params.get("diameter") or 300), mark=str(params.get("mark") or wf.NOTCH),
                         mark_angle=float(params.get("mark_angle") or 270),
                         edge_exclusion=float(params.get("edge_exclusion", 3.0)))
        x, y = wf.rotate(x, y, float(params.get("rotation", 0.0) or 0.0))
        keep = np.isfinite(x) & np.isfinite(y) & (np.isfinite(value) if value is not None else True)
        x, y = x[keep], y[keep]
        value = np.asarray(value, float)[keep] if value is not None else None
        inside = wafer.inside(x, y)
        width, height = float(params.get("die_width", 10.0)), float(params.get("die_height", 10.0))
        offset_x, offset_y = float(params.get("die_offset_x", 0.0)), float(params.get("die_offset_y", 0.0))
        dies = wf.die_grid(wafer, width, height, offset_x, offset_y)
        die = wf.assign_dies(x, y, dies, width, height, offset_x, offset_y)
        die[~inside] = -1
        lower, upper = _limit(params.get("lower")), _limit(params.get("upper"))
        per_die = wf.die_summary(dies, die, value, width, height, lower=lower, upper=upper)
        points = pd.DataFrame({"x": x, "y": y, **({"value": value} if value is not None else {}),
                               "Die": die, "On wafer": inside.astype(int)})
        summary: dict[str, Any] = {"Diameter (mm)": wafer.diameter, "Mark": f"{wafer.mark} at {wafer.mark_angle:g}°",
                                   "Rotation (°)": float(params.get("rotation", 0.0) or 0.0),
                                   "Edge exclusion (mm)": wafer.edge_exclusion,
                                   "Die (mm)": f"{width:g} × {height:g}", "Points": int(len(points)),
                                   "Points on the usable wafer": int(inside.sum())}
        grid = pd.DataFrame()
        if value is not None:
            summary.update(wf.uniformity(value[inside]))
            if inside.sum() >= 3:
                grid = wf.contour_grid(x[inside], y[inside], value[inside], wafer, int(params.get("grid_points", 81)))
        summary.update(per_die.summary)
        zones = wf.radial_zones(x[inside], y[inside], value[inside] if value is not None else None, wafer)
        return WaferResult(source_name=name, wafer=wafer, points=points, dies=per_die.dies, grid=grid, zones=zones,
                           summary=summary, die_size=(width, height, offset_x, offset_y),
                           has_value=value is not None, has_limits=lower is not None or upper is not None)

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """The maps are a figure of their own: nothing is added to the series' axis."""
        del results
        return selected_axis_id

    # ------------------------------------------------------------------
    # The tables and the figure
    # ------------------------------------------------------------------

    def _name_tables(self, result: WaferResult) -> dict[str, str]:
        if not result.tables:
            source = "".join(c if c.isalnum() else "_" for c in result.source_name)[:40]
            stem = self.output_stem() or f"Wafer_{source}"  # the Charts tab's Result name, when one is typed
            keys = ["points", "dies"] + (["grid"] if not result.grid.empty else [])
            result.tables = {key: self._repo.free_table_name(f"{stem}_{key}") for key in keys}
        return result.tables

    def undo_tables(self, results: Sequence[Any]) -> list[str]:
        return [name for result in results for name in self._name_tables(result).values()]

    def _wafer_options(self, result: WaferResult, display: str, **more: Any) -> dict[str, Any]:
        width, height, offset_x, offset_y = result.die_size
        wafer = result.wafer
        return {"display": display, "diameter": wafer.diameter, "mark": wafer.mark, "mark_angle": wafer.mark_angle,
                "edge_exclusion": wafer.edge_exclusion, "die_width": width, "die_height": height,
                "die_offset_x": offset_x, "die_offset_y": offset_y, **more}

    def draw(self, result: WaferResult) -> None:
        """The tables written, then the figure of the maps ticked in the Charts tab."""
        tables = self._name_tables(result)
        frames = {"points": result.points, "dies": result.dies, "grid": result.grid}
        for key, name in tables.items():
            self._repo.import_dataframe(frames[key], table_name=name, normalize_columns=False)
        q = quote_identifier
        panels: list[tuple] = []
        label = result.source_name
        if self.chart_enabled("chart_map"):
            if result.has_value and "grid" in tables:
                sql = f"SELECT x, y, value FROM {q(tables['grid'])}"
                panels.append(("Wafer Map", _("{name}: map").format(name=label), "x (mm)", "y (mm)",
                               [(label, sql, {"x": "x", "y": "y", "value": "value"}, {})],
                               self._wafer_options(result, "contour", colorbar_label=label)))
            elif not result.has_value:
                sql = f'SELECT x, y FROM {q(tables["points"])} WHERE "On wafer" = 1'
                panels.append(("Wafer Map", _("{name}: defects").format(name=label), "x (mm)", "y (mm)",
                               [(label, sql, {"x": "x", "y": "y"}, {"color": "#dc2626"})],
                               self._wafer_options(result, "points", markersize=3.0)))
        if self.chart_enabled("chart_dies"):
            column, title = ("Mean", _("{name}: die means")) if result.has_value else ("Count", _("{name}: defects per die"))
            sql = f'SELECT "X" AS x, "Y" AS y, "{column}" AS value FROM {q(tables["dies"])}'
            panels.append(("Wafer Map", title.format(name=label), "x (mm)", "y (mm)",
                           [(label, sql, {"x": "x", "y": "y", "value": "value"}, {})],
                           self._wafer_options(result, "dies", colorbar_label=_(column),
                                               cmap="viridis" if result.has_value else "Reds")))
        if self.chart_enabled("chart_bins") and result.has_value and result.has_limits:
            sql = f'SELECT "X" AS x, "Y" AS y, "Bin" AS value FROM {q(tables["dies"])}'
            panels.append(("Wafer Map", _("{name}: die bins").format(name=label), "x (mm)", "y (mm)",
                           [(label, sql, {"x": "x", "y": "y", "value": "value"}, {})],
                           self._wafer_options(result, "dies", bins=True)))
        if panels:
            self.create_report_figure(f"{_(self.Name)} · {label}",
                                      self.results_document(label, self.report_for(result)), panels)

    # ------------------------------------------------------------------
    # The report
    # ------------------------------------------------------------------

    def report_for(self, result: WaferResult) -> str:
        summary = report_html.summary_table(
            (_(key), str(value) if isinstance(value, str) else report_html.format_number(value))
            for key, value in result.summary.items())
        zones = report_html.frame_table(result.zones, _, digits=5)
        parts = [report_html.section(_("Summary"), summary), report_html.section(_("Centre to edge"), zones)]
        if result.has_value and result.has_limits:
            counts = result.dies[result.dies["Bin"] != wf.BIN_EMPTY]["Bin name"].value_counts()
            parts.append(report_html.section(_("Die bins"), report_html.summary_table(
                (_(name), report_html.format_number(int(count), digits=0)) for name, count in counts.items())))
        return report_html.section(result.source_name, "".join(parts))

    def format_results(self, results: Sequence[Any]) -> str:
        if not results:
            return ""
        subtitle = ", ".join(result.source_name for result in results)
        return self.results_document(subtitle, *(self.report_for(result) for result in results))
