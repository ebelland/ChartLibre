"""Find peaks in a chart series, and measure them.

``find_peaks`` on its own answers a question nobody asks.  A raw local-maximum
search on real data returns hundreds of hits, almost all of them noise, and the
list is useless until it is filtered by something meaningful.  So this dialog
is built around the filters rather than around the search: prominence, height,
distance and width are the parameters, and the peak list is what falls out.

**Prominence is the default filter, not height.**  Height compares a peak to
zero, which is only meaningful when the baseline is at zero - on a sloping or
raised background it selects whichever part of the signal happens to sit
highest, not the peaks.  Prominence measures how far a peak stands above the
higher of the two saddles flanking it, so it asks "how much of a peak is this,
locally", which is the question that survives a moving baseline.

Each peak is reported with its width at half prominence, and with the x bounds
of that width - which are what the Calculus dialog's integral wants, so a peak
found here can be integrated there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QWidget
from app.analysis import peaks as pk
from app.analysis.peaks import Peak, PeakSettings, find_peaks_1d, find_peaks_2d
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.utils import report_html
from app.utils.i18n import _

PEAKS_MAXIMA = "Maxima"
PEAKS_MINIMA = "Minima"
PEAKS_BOTH = "Maxima and minima"

#: The engine's name for each model.
_MODE: dict[str, str] = {
    PEAKS_MAXIMA: pk.MAXIMA,
    PEAKS_MINIMA: pk.MINIMA,
    PEAKS_BOTH: pk.BOTH,
}

#: The models offered, in combo order, with their documentation.
PEAK_MODELS: dict[str, OperationModel] = {
    PEAKS_MAXIMA: OperationModel(
        doc_title="Topographic prominence",
        doc_url="https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
    PEAKS_MINIMA: OperationModel(
        doc_title="Topographic prominence",
        doc_url="https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
    PEAKS_BOTH: OperationModel(
        doc_title="Topographic prominence",
        doc_url="https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
}


@dataclass(slots=True)
class PeakResult(TableResult):
    """Every peak found in one source series."""

    source_name: str
    result_name: str
    model: str
    peaks: list[Peak] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        if self.metadata.get("is_3d"):
            return pd.DataFrame(
                {
                    "x": [peak.x for peak in self.peaks],
                    "y": [peak.y for peak in self.peaks],
                    "z": [peak.z for peak in self.peaks],
                    "prominence": [peak.prominence for peak in self.peaks],
                    "is_minimum": [int(peak.is_minimum) for peak in self.peaks],
                }
            )
        return pd.DataFrame(
            {
                "x": [peak.x for peak in self.peaks],
                "y": [peak.y for peak in self.peaks],
                "prominence": [peak.prominence for peak in self.peaks],
                "width": [peak.width for peak in self.peaks],
                "left_x": [peak.left_x for peak in self.peaks],
                "right_x": [peak.right_x for peak in self.peaks],
                "is_minimum": [int(peak.is_minimum) for peak in self.peaks],
            }
        )


class SeriesPeaksDialog(SeriesOperationDialogBase):
    """Locate and measure peaks in a chart series."""

    MODELS = PEAK_MODELS
    MODEL_TOOLTIP = "Which turning points to look for."
    MODEL_LABEL = "Find:"

    Name: str = "Peaks"
    Description = "Find and measure peaks"

    #: Computed on a worker thread: see SeriesOperationDialogBase.evaluate.
    RUN_IN_BACKGROUND = True

    # A peak is defined by its neighbours, so the points must be in x order.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # Three is the smallest series that can contain an interior maximum.
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        ChoiceParam(
            "filter_by",
            "Filter by:",
            tooltip=(
                "Prominence measures a peak against its own surroundings and "
                "survives a sloping baseline. Height compares it to zero, "
                "which only means something when the baseline is at zero."
            ),
            choices=(
                ("Prominence (relative)", "prominence"),
                ("Height (absolute)", "height"),
            ),
        ),
        FloatParam(
            "threshold",
            "Minimum:",
            tooltip=(
                "As a fraction of the signal's full range. 0.1 keeps peaks "
                "standing at least a tenth of the range above their "
                "surroundings."
            ),
            default_value=0.05,
            minimum=0.0,
            maximum=1.0,
            decimals=4,
            step=0.01,
        ),
        IntParam(
            "distance",
            "Minimum separation:",
            tooltip=(
                "Points. Of any two peaks closer than this, the more "
                "prominent is kept - which is how a single noisy peak stops "
                "being reported as several."
            ),
            default_value=1,
            minimum=1,
            maximum=100_000,
        ),
        FloatParam(
            "min_width",
            "Minimum width:",
            tooltip=(
                "Points at half prominence. Raise it to reject single-sample "
                "spikes, which are usually instrument artefacts rather than "
                "features."
            ),
            default_value=0.0,
            minimum=0.0,
            maximum=100_000.0,
            decimals=2,
            step=1.0,
        ),
        IntParam(
            "limit",
            "Report at most:",
            tooltip="Keeps the most prominent peaks when many are found.",
            default_value=50,
            minimum=1,
            maximum=10_000,
        ),
    )

    # "Mark peaks only" was declared here and read nowhere - a checkbox that
    # did nothing at all, for either of its states (todo.txt P3-x). It is
    # gone rather than implemented because neither reading of it survives
    # contact with what this dialog already does: the result table carries
    # every measurement because the report is built from it, and the chart
    # series draws markers because joining peaks would draw a curve through
    # nothing. There was no third behaviour left for the box to choose.

    Icon = """
    <path d="M3 18l4-8 3 5 4-11 3 8 4-4"/>
    <circle cx="14" cy="4" r="1.6"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesPeaksDialog requires a repository instance.")

        self._last_results: list[PeakResult] = []
        self._parameter_form: QFormLayout | None = None

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Peaks",
            parent=parent,
            width=780,
            height=640,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def series_settings(self) -> tuple[str, dict[str, Any]]:
        return self.current_model(), self.parameter_values()

    def read_series(self, row: Any, name: str) -> tuple[bool, Any]:
        # A series with a z role is a surface, not a curve - see
        # series_origin. The 1D find_peaks search has nothing to say about
        # it, so it gets its own 2D local-maximum search instead.
        return self.read_curve_or_surface(row, name)

    def compute_series(
        self, name: str, data: tuple[bool, Any], settings: tuple[str, dict[str, Any]]
    ) -> PeakResult:
        model, params = settings
        surface, values = data
        if surface:
            return self._find_one_3d(values, name, model, params)
        x_values, y_values = values
        return self._find_one(name, x_values, y_values, model, params)

    def _find_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> PeakResult:
        peaks = find_peaks_1d(x_values, y_values, _MODE[model], self._settings(params))

        return PeakResult(
            source_name=name,
            result_name=f"{name} - peaks",
            model=model,
            peaks=peaks,
            metadata={
                "found": len(peaks),
                "filter": str(params.get("filter_by", "prominence")),
            },
        )

    @staticmethod
    def _settings(params: Mapping[str, Any]) -> PeakSettings:
        """The filters, read off the parameter values."""
        return PeakSettings(
            filter_by=str(params.get("filter_by", "prominence")),
            threshold=float(params.get("threshold", 0.05)),
            distance=int(params.get("distance", 1)),
            min_width=float(params.get("min_width", 0.0)),
            limit=int(params.get("limit", 50)),
        )

    # ------------------------------------------------------------------
    # Surfaces (a series with a z role): 2D local-maximum search
    # ------------------------------------------------------------------

    def _find_one_3d(
        self,
        grid: tuple[np.ndarray, np.ndarray, np.ndarray, bool],
        name: str,
        model: str,
        params: Mapping[str, Any],
    ) -> PeakResult:
        """Find local maxima/minima on a gridded (or interpolated) surface.

        Same filter vocabulary as the 1D search - "Minimum" as a fraction of
        the z range, "Minimum separation" as a neighbourhood size - reused
        rather than duplicated with new names, so switching a series from a
        curve to a surface does not also mean learning new parameters.
        """
        x_grid, y_grid, z_grid, interpolated = grid

        peaks = find_peaks_2d(x_grid, y_grid, z_grid, _MODE[model], self._settings(params))

        return PeakResult(
            source_name=name,
            result_name=f"{name} - peaks",
            model=model,
            peaks=peaks,
            metadata={
                "found": len(peaks),
                "filter": str(params.get("filter_by", "prominence")),
                "is_3d": True,
                "interpolated": bool(interpolated),
            },
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: PeakResult,
    ) -> ResultSeriesSpec:
        del axis_id
        is_3d = bool(result.metadata.get("is_3d"))
        # Same source series -> same axis either way: a series with a z role
        # is already drawn on a 3D-projection axis by the time this dialog
        # can select it (see series_origin), so - exactly like the 1D peaks
        # markers, which never create an axis of their own - the found peaks
        # are simply added to that same axis, now with roles x, y, z.
        sql_query = (
            f'SELECT x, y, z FROM "{table_name}"'
            if is_3d
            else f'SELECT x, y FROM "{table_name}" ORDER BY x'
        )
        roles = {"x": "x", "y": "y", "z": "z"} if is_3d else {"x": "x", "y": "y"}
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=sql_query,
            roles=roles,
            style={
                "generated_peaks": True,
                "peaks_dialog": "series_peaks",
                "source_name": result.source_name,
                "model": result.model,
                # Markers with no connecting line: the peaks are separate
                # findings, and joining them would draw a curve through
                # nothing.
                "linestyle": "",
                "marker": "v",
                "markersize": 8.0,
            },
        )

    RESULT_TABLE_PREFIX = 'Peaks'
    RESULT_TABLE_VARIANT = None

    @property
    def operation_label(self) -> str:
        return "Peaks"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[PeakResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            if not result.peaks:
                sections.append(
                    report_html.section(
                        result.source_name,
                        report_html.note(
                            _(
                                "No peaks passed the filter. Lower the minimum, "
                                "or switch from height to prominence if the "
                                "baseline is not at zero."
                            )
                        ),
                    )
                )
                continue

            if result.metadata.get("is_3d"):
                grid_note = report_html.note(
                    _(
                        "This surface's points were not a complete x/y grid, "
                        "so the peaks below were found on a linearly "
                        "interpolated one - treat them as approximate."
                    )
                    if result.metadata.get("interpolated")
                    else _("Found on the surface's own exact x/y grid.")
                )
                rows_3d = [
                    (
                        str(index + 1),
                        report_html.format_number(peak.x),
                        report_html.format_number(peak.y),
                        report_html.format_number(peak.z),
                        report_html.format_number(peak.prominence, digits=4),
                        _("minimum") if peak.is_minimum else _("maximum"),
                    )
                    for index, peak in enumerate(result.peaks)
                ]
                sections.append(
                    report_html.section(
                        f"{result.source_name} \u2014 {len(result.peaks)}",
                        grid_note,
                        report_html.table(
                            ("#", "x", "y", "z", _("Prominence"), _("Kind")),
                            rows_3d,
                            align=("right", "right", "right", "right", "right", "left"),
                        ),
                    )
                )
                continue

            rows = [
                (
                    str(index + 1),
                    report_html.format_number(peak.x),
                    report_html.format_number(peak.y),
                    report_html.format_number(peak.prominence, digits=4),
                    report_html.format_number(peak.width, digits=4),
                    f"{report_html.format_number(peak.left_x)} .. "
                    f"{report_html.format_number(peak.right_x)}",
                    _("minimum") if peak.is_minimum else _("maximum"),
                )
                for index, peak in enumerate(result.peaks)
            ]
            sections.append(
                report_html.section(
                    f"{result.source_name} \u2014 {len(result.peaks)}",
                    report_html.table(
                        (
                            "#",
                            "x",
                            "y",
                            _("Prominence"),
                            _("Width"),
                            _("Half-prominence bounds"),
                            _("Kind"),
                        ),
                        rows,
                        align=(
                            "right", "right", "right", "right", "right",
                            "right", "left",
                        ),
                    ),
                )
            )

        return report_html.document(_("Peaks"), self.current_model(), *sections)
