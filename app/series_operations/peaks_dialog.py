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
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.signal import find_peaks, peak_prominences, peak_widths

from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

PEAKS_MAXIMA = "Maxima"
PEAKS_MINIMA = "Minima"
PEAKS_BOTH = "Maxima and minima"

PEAK_MODELS = (PEAKS_MAXIMA, PEAKS_MINIMA, PEAKS_BOTH)

PEAK_DOCS = {
    PEAKS_MAXIMA: (
        "Topographic prominence",
        "https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
    PEAKS_MINIMA: (
        "Topographic prominence",
        "https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
    PEAKS_BOTH: (
        "Topographic prominence",
        "https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
}


@dataclass(slots=True)
class Peak:
    """One located peak, with the measurements that describe it.

    ``z`` is None for an ordinary 1D peak on a curve. A peak found on a
    surface (``z`` role present, see ``_find_one_3d``) sets it, and leaves
    ``width``/``left_x``/``right_x`` as NaN - those describe a 1D half-
    prominence interval, which a 2D local maximum simply does not have.
    """

    x: float
    y: float
    prominence: float
    width: float
    left_x: float
    right_x: float
    is_minimum: bool = False
    z: float | None = None


@dataclass(slots=True)
class PeakResult:
    """Every peak found in one source series."""

    source_name: str
    result_name: str
    model: str
    peaks: list[Peak] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
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

    Name: str = "Peaks"
    Description = "Find and measure peaks"

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

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        container = QWidget(panel)
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(PEAK_MODELS)
        self.model_combo.setToolTip(_("Which turning points to look for."))
        form.addRow(_("Find:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(container)
        return panel

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.mark_results_stale)

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        title, url = PEAK_DOCS[self._model()]
        set_doc_link(self._doc_link, title, url)

    def _model(self) -> str:
        return self.model_combo.currentText() or PEAKS_MAXIMA

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[PeakResult]:
        model = self._model()
        params = self.parameter_values()

        results: list[PeakResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                roles = parse_roles(row_value(row, "roles", default={}))
                if roles.get("z"):
                    # A series with a z role is a surface, not a curve - see
                    # series_origin. The 1D find_peaks search below has
                    # nothing to say about it, so it gets its own 2D local-
                    # maximum search instead; the 1D path is untouched.
                    results.append(self._find_one_3d(row, name, model, params))
                else:
                    x_values, y_values = self.series_xy(row, name)
                    results.append(self._find_one(name, x_values, y_values, model, params))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

    def _find_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> PeakResult:
        peaks: list[Peak] = []

        if model in (PEAKS_MAXIMA, PEAKS_BOTH):
            peaks.extend(self._search(x_values, y_values, params, minimum=False))
        if model in (PEAKS_MINIMA, PEAKS_BOTH):
            # A minimum is a maximum of the inverted signal. Inverting rather
            # than writing a second search keeps one implementation of the
            # prominence and width logic, which is where the subtlety is.
            peaks.extend(self._search(x_values, y_values, params, minimum=True))

        peaks.sort(key=lambda peak: peak.x)

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

    def _search(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        params: Mapping[str, Any],
        *,
        minimum: bool,
    ) -> list[Peak]:
        """Run find_peaks once, on the signal or on its inverse."""
        signal = -y_values if minimum else y_values

        # Thresholds are given as a fraction of the signal's range so that one
        # setting means the same thing on a millivolt trace and on a count
        # rate. An absolute default would be meaningless on arrival.
        span = float(np.ptp(y_values))
        if span <= 0.0:
            return []

        threshold = float(params.get("threshold", 0.05)) * span
        distance = max(1, int(params.get("distance", 1)))
        min_width = float(params.get("min_width", 0.0))

        kwargs: dict[str, Any] = {"distance": distance}
        if str(params.get("filter_by", "prominence")) == "height":
            kwargs["height"] = float(np.min(signal)) + threshold
        else:
            kwargs["prominence"] = max(threshold, 1e-12)
        if min_width > 0.0:
            kwargs["width"] = min_width

        indices, _properties = find_peaks(signal, **kwargs)
        if indices.size == 0:
            return []

        prominences = peak_prominences(signal, indices)[0]
        # rel_height=0.5 is the width at half prominence, not at half height:
        # on a raised baseline those differ, and the half-prominence width is
        # the one that describes the peak rather than the background.
        widths, _heights, left_ips, right_ips = peak_widths(
            signal, indices, rel_height=0.5
        )

        # peak_widths returns fractional sample positions, so the x bounds have
        # to be interpolated back onto the real axis rather than indexed.
        sample_positions = np.arange(x_values.size, dtype=float)
        left_x = np.interp(left_ips, sample_positions, x_values)
        right_x = np.interp(right_ips, sample_positions, x_values)
        width_in_x = right_x - left_x

        found = [
            Peak(
                x=float(x_values[index]),
                y=float(y_values[index]),
                prominence=float(prominence),
                width=float(width),
                left_x=float(left),
                right_x=float(right),
                is_minimum=minimum,
            )
            for index, prominence, width, left, right in zip(
                indices, prominences, width_in_x, left_x, right_x
            )
        ]

        limit = int(params.get("limit", 50))
        if len(found) > limit:
            # Keep the most prominent, not the first: truncating in x order
            # would discard the strongest peaks whenever they are late in the
            # series.
            found.sort(key=lambda peak: peak.prominence, reverse=True)
            found = found[:limit]

        return found

    # ------------------------------------------------------------------
    # Surfaces (a series with a z role): 2D local-maximum search
    # ------------------------------------------------------------------

    def _find_one_3d(
        self,
        row: Any,
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
        x_grid, y_grid, z_grid, interpolated = self.series_grid_xyz(row, name)

        peaks: list[Peak] = []
        if model in (PEAKS_MAXIMA, PEAKS_BOTH):
            peaks.extend(self._search_3d(x_grid, y_grid, z_grid, params, minimum=False))
        if model in (PEAKS_MINIMA, PEAKS_BOTH):
            peaks.extend(self._search_3d(x_grid, y_grid, z_grid, params, minimum=True))

        peaks.sort(key=lambda peak: (peak.x, peak.y))

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

    def _search_3d(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        params: Mapping[str, Any],
        *,
        minimum: bool,
    ) -> list[Peak]:
        """2D analogue of ``_search``: a local-maximum filter, not find_peaks.

        ``scipy.signal.find_peaks`` is a 1D algorithm; a surface's local
        maxima are found with ``scipy.ndimage.maximum_filter`` instead - a
        cell is a candidate when it equals the maximum of its own
        neighbourhood, and its "prominence" is how far it stands above the
        *minimum* of that same neighbourhood (the 2D reading of the 1D
        prominence idea: height above the nearest lower ground).

        NaN cells - present only in an interpolated grid, outside the convex
        hull of the original points (see ``series_grid_xyz``) - are pushed to
        -inf before filtering so they can never win a maximum comparison, and
        any candidate whose own cell was NaN, or whose neighbourhood touches
        a NaN (making its "minimum" -inf and its prominence infinite), is
        discarded rather than reported as a peak.
        """
        signal = -z_grid if minimum else z_grid
        finite_mask = np.isfinite(signal)
        if not np.any(finite_mask):
            return []

        finite_values = signal[finite_mask]
        span = float(np.ptp(finite_values))
        if span <= 0.0:
            return []

        threshold = float(params.get("threshold", 0.05)) * span
        distance = max(1, int(params.get("distance", 1)))
        neighborhood = 2 * distance + 1

        masked = np.where(finite_mask, signal, -np.inf)
        local_max = maximum_filter(masked, size=neighborhood, mode="nearest")
        local_min = minimum_filter(masked, size=neighborhood, mode="nearest")
        with np.errstate(invalid="ignore"):
            prominence = masked - local_min

        is_candidate = finite_mask & (masked == local_max) & np.isfinite(prominence)

        if str(params.get("filter_by", "prominence")) == "height":
            base = float(np.min(finite_values))
            keep = is_candidate & (masked >= base + threshold)
        else:
            keep = is_candidate & (prominence >= max(threshold, 1e-12))

        rows_idx, cols_idx = np.nonzero(keep)
        found = [
            Peak(
                x=float(x_grid[r, c]),
                y=float(y_grid[r, c]),
                z=float(z_grid[r, c]),
                prominence=float(prominence[r, c]),
                width=float("nan"),
                left_x=float("nan"),
                right_x=float("nan"),
                is_minimum=minimum,
            )
            for r, c in zip(rows_idx.tolist(), cols_idx.tolist())
        ]

        limit = int(params.get("limit", 50))
        if len(found) > limit:
            found.sort(key=lambda peak: peak.prominence, reverse=True)
            found = found[:limit]

        return found

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: PeakResult) -> pd.DataFrame:
        return result.to_frame()

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

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_peaks": True, "peaks_dialog": "series_peaks"}

    def result_table_name(self, axis_id: int, result: PeakResult) -> str:
        return generated_table_name(
            f"Peaks_axis{axis_id}_{result.source_name}",
            fallback="Peaks_Result",
        )

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

        return report_html.document(_("Peaks"), self._model(), *sections)
