"""Find where a series crosses a level: the roots of y(x) - level.

The question this answers is asked constantly and answered by eye: where does
the titration curve cross the endpoint, where does the response fall to half
its value, at what temperature does the difference change sign. Reading it off
the chart is accurate to about a pixel; this is accurate to the tolerance,
which is usually eleven digits better.

**Brackets first, then SciPy.** ``scipy.optimize`` solves ``f(x) = 0`` for a
function it can evaluate anywhere, and a series is not that - it is samples.
So the work is done in two halves, and the first is the one that decides
whether the answer is right:

1.  Scan consecutive samples for a sign change in ``y - level``. Every such
    pair *brackets* a crossing, by the intermediate value theorem, and a
    crossing that no pair brackets cannot be found from these samples at all.
    A sample sitting exactly on the level is a root already and is taken as
    one.
2.  Refine each bracket with a SciPy solver over an interpolant of the
    series. Bracketing solvers - Brent, bisection, TOMS 748 - cannot leave
    the interval they were given, so the refinement can only improve the
    answer, never wander off to a different crossing.

**Which interpolant is the real assumption.** Linear is the honest default:
it claims nothing between the samples that the samples do not already say,
and on a linear interpolant the root is the straight-line crossing - the
answer everyone computes by hand. Cubic is better for a smooth signal that
was sampled coarsely, and worse for a noisy one, where it overshoots between
points and can invent crossings that are not there. Both are offered; only
the first is the default.

**Newton is offered and is not the default.** ``scipy.optimize.newton`` -
the secant method here, since a sampled series has no analytic derivative -
converges faster and is not bracketed, so it can step outside the interval
and return a crossing somewhere else entirely, or none. It is kept because
it is the right tool on a smooth, well-separated curve, and its result is
checked against the bracket it came from before being accepted.

Sitting next to the peaks dialog on purpose: peaks are where the derivative
crosses zero, so a derivative from the calculus dialog run through this one
locates them a second way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QWidget
from app.analysis.roots import (
    INTERP_CUBIC,
    INTERP_LINEAR,
    INTERP_PCHIP,
    SOLVER_BISECT,
    SOLVER_BRENT,
    SOLVER_NEWTON,
    SOLVER_TOMS748,
    Root,
    find_level_curve,
    find_roots,
)
from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.results import TableResult
from app.series_operations.dialog_base import (
    OperationModel,
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.series_operations.parameter_spec import (
    ChoiceParam,
    FloatParam,
    IntParam,
)
from app.utils import report_html
from app.utils.i18n import _

# --- Solvers ----------------------------------------------------------

# The solver names are the engine's (app.analysis.roots); the dialog's own
# names for them stay, for the callers and tests that already use them.
ROOT_BRENT = SOLVER_BRENT
ROOT_BISECT = SOLVER_BISECT
ROOT_TOMS748 = SOLVER_TOMS748
ROOT_NEWTON = SOLVER_NEWTON

#: The models offered, in combo order.
ROOT_MODELS: dict[str, OperationModel] = {
    ROOT_BRENT: OperationModel(
        doc_title="Brent's method",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html",
    ),
    ROOT_BISECT: OperationModel(
        doc_title="Bisection method",
        doc_url="https://en.wikipedia.org/wiki/Bisection_method",
    ),
    ROOT_TOMS748: OperationModel(
        doc_title="TOMS 748",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.toms748.html",
    ),
    ROOT_NEWTON: OperationModel(
        doc_title="Secant method",
        doc_url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html",
    ),
}





@dataclass(slots=True)
class RootResult(TableResult):
    """Every crossing found in one source series."""

    source_name: str
    result_name: str
    model: str
    level: float
    roots: list[Root] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        if self.metadata.get("is_3d"):
            # A NaN row between two curve_index groups: several disconnected
            # level curves (a saddle's z=0 set is two crossing lines) are
            # stored end to end in one table, and without a break matplotlib
            # would draw a spurious line segment joining the end of one
            # curve to the start of the next.
            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            curve_indices: list[int] = []
            last_curve: int | None = None
            for root in self.roots:
                if last_curve is not None and root.curve_index != last_curve:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                    zs.append(float("nan"))
                    curve_indices.append(last_curve)
                xs.append(root.x)
                ys.append(root.y)
                zs.append(root.z if root.z is not None else float("nan"))
                curve_indices.append(root.curve_index)
                last_curve = root.curve_index
            return pd.DataFrame({"x": xs, "y": ys, "z": zs, "curve_index": curve_indices})
        return pd.DataFrame(
            {
                "x": [root.x for root in self.roots],
                "y": [root.y for root in self.roots],
                "level": [self.level for _root in self.roots],
                "rising": [int(root.rising) for root in self.roots],
                "method": [root.method for root in self.roots],
                "iterations": [root.iterations for root in self.roots],
            }
        )


class SeriesRootsDialog(SeriesOperationDialogBase):
    """Solve y(x) = level for x, on every crossing the samples bracket."""

    MODELS = ROOT_MODELS
    MODEL_LABEL = "Solver:"
    MODEL_TOOLTIP = (
        "How each bracketed crossing is refined. The three bracketing "
        "solvers cannot leave the interval they were given; Newton "
        "can, and is checked afterwards."
    )

    Name: str = "Roots"
    Description = "Find where a series crosses a level"

    # A crossing is a property of consecutive samples, so x has to be in
    # order and each x can only have one y.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # Two points are one interval, which is the smallest thing that can
    # bracket a crossing.
    INPUT_MINIMUM_POINTS = 2

    PARAMS = (
        FloatParam(
            "level",
            "Level:",
            tooltip=(
                "The y value to solve for. 0 finds the zeros; any other "
                "value finds where the series crosses that level, which is "
                "the same problem shifted."
            ),
            default_value=0.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=0.1,
        ),
        ChoiceParam(
            "interpolation",
            "Between samples:",
            tooltip=(
                "What the series is assumed to do between two points. "
                "Linear claims nothing the samples do not; the two curved "
                "options fit a smooth signal better and can overshoot a "
                "noisy one into crossings that are not there."
            ),
            choices=(
                ("Straight line", INTERP_LINEAR),
                ("Cubic spline", INTERP_CUBIC),
                ("Monotone cubic (PCHIP)", INTERP_PCHIP),
            ),
        ),
        IntParam(
            "tolerance_digits",
            "Tolerance (digits):",
            tooltip=(
                "How many decimal places of x the solver has to settle "
                "before it stops. Nine is far below the precision of any "
                "measured x; the cost of raising it is a few iterations."
            ),
            default_value=9,
            minimum=3,
            maximum=14,
        ),
        IntParam(
            "max_iter",
            "Maximum iterations:",
            tooltip=(
                "Per crossing. Brent and TOMS 748 rarely need ten; the "
                "limit is what stops a pathological bracket from hanging "
                "the dialog."
            ),
            default_value=100,
            minimum=5,
            maximum=10_000,
        ),
        IntParam(
            "limit",
            "Report at most:",
            tooltip=(
                "Keeps the first crossings in x order when a noisy series "
                "crosses the level hundreds of times."
            ),
            default_value=100,
            minimum=1,
            maximum=10_000,
        ),
    )

    #: A curve crossing a horizontal line, with the crossing marked.
    Icon = """
    <path d="M3 12h18"/>
    <path d="M3 19c4 0 5-14 9-14s5 10 9 10"/>
    <circle cx="8.4" cy="12" r="1.8"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesRootsDialog requires a repository instance.")

        self._last_results: list[RootResult] = []

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Roots",
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


    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()

    def _model(self) -> str:
        return self.current_model(ROOT_BRENT)


    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[RootResult]:
        model = self._model()
        params = self.parameter_values()

        results: list[RootResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                roles = parse_roles(row_value(row, "roles", default={}))
                if roles.get("z"):
                    # A series with a z role is a surface: "the roots" are a
                    # whole level *curve* (z = level), not a handful of x
                    # crossings, so it gets its own path (matplotlib's
                    # contour extraction) rather than the bracket-then-SciPy
                    # machinery below, which assumes a single-valued y(x).
                    results.append(self._solve_one_3d(row, name, params))
                else:
                    x_values, y_values = self.series_xy(row, name)
                    results.append(self._solve_one(name, x_values, y_values, model, params))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

    def _solve_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> RootResult:
        level = float(params.get("level", 0.0))
        interpolation = str(params.get("interpolation", INTERP_LINEAR))
        search = find_roots(
            x_values,
            y_values,
            level=level,
            solver=model,
            interpolation=interpolation,
            tolerance_digits=params.get("tolerance_digits", 9),
            max_iter=int(params.get("max_iter", 100)),
            limit=int(params.get("limit", 100)),
        )
        for note in search.notes:
            applogger.info(note)
        return RootResult(
            source_name=name,
            result_name=f"{name} - roots",
            model=model,
            level=level,
            roots=search.roots,
            metadata={
                "found": len(search.roots),
                "truncated": search.truncated,
                "interpolation": interpolation,
                "samples": int(x_values.size),
            },
        )

    # --- Surfaces (a series with a z role): the level curve z = level -----

    def _solve_one_3d(
        self,
        row: Any,
        name: str,
        params: Mapping[str, Any],
    ) -> RootResult:
        """Find the level curve z = level on a gridded (or interpolated) surface.

        The bracket-and-refine machinery assumes y is single-valued in x,
        which a surface's level set is not - a saddle's z = 0 set is two
        crossing lines - so it is matplotlib's contour extraction instead
        (app.analysis.roots.find_level_curve).
        """
        level = float(params.get("level", 0.0))
        x_grid, y_grid, z_grid, interpolated = self.series_grid_xyz(row, name)
        search = find_level_curve(
            x_grid, y_grid, z_grid, level=level, limit=int(params.get("limit", 100))
        )
        return RootResult(
            source_name=name,
            result_name=f"{name} - roots",
            model="Contour (matplotlib)",
            level=level,
            roots=search.roots,
            metadata={
                "found": len(search.roots),
                "truncated": search.truncated,
                "is_3d": True,
                "interpolated": bool(interpolated),
                "curves": search.curves,
            },
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: RootResult,
    ) -> ResultSeriesSpec:
        del axis_id
        if result.metadata.get("is_3d"):
            # A level curve is a curve, not scattered findings - drawn with a
            # connecting line (3D Line Plot), unlike the 1D case's isolated
            # markers. Points are read back in the row order matplotlib's
            # contour produced them in, which is what keeps a polyline a
            # polyline rather than a scribble - the same reason
            # Line3DAxisRenderer never reorders its own input.
            return ResultSeriesSpec(
                name=result.result_name,
                sql_query=f'SELECT x, y, z, curve_index FROM "{table_name}" ORDER BY curve_index, rowid',
                roles={"x": "x", "y": "y", "z": "z"},
                style={
                    "generated_roots": True,
                    "roots_dialog": "series_roots",
                    "source_name": result.source_name,
                    "model": result.model,
                    "level": result.level,
                    "linestyle": "-",
                    "marker": "",
                },
            )
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_roots": True,
                "roots_dialog": "series_roots",
                "source_name": result.source_name,
                "model": result.model,
                "level": result.level,
                # Markers, no line: the crossings are separate findings,
                # and joining them would draw a horizontal line at the
                # level that looks like a series.
                "linestyle": "",
                "marker": "o",
                "markersize": 7.0,
            },
        )

    RESULT_TABLE_PREFIX = 'Roots'
    RESULT_TABLE_VARIANT = None

    @property
    def operation_label(self) -> str:
        return "Roots"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[RootResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            is_3d = bool(result.metadata.get("is_3d"))

            if not result.roots:
                message = (
                    _(
                        "The surface never reaches this level - z stays on "
                        "one side of it everywhere on the grid."
                    )
                    if is_3d
                    else _(
                        "The series never crosses this level. Check "
                        "the level against the data's range - a "
                        "crossing no two samples straddle cannot be "
                        "found from these samples."
                    )
                )
                sections.append(
                    report_html.section(result.source_name, report_html.note(message))
                )
                continue

            if is_3d:
                grid_note = report_html.note(
                    _(
                        "This surface's points were not a complete x/y grid, "
                        "so this level curve was traced on a linearly "
                        "interpolated one - treat it as approximate."
                    )
                    if result.metadata.get("interpolated")
                    else _("Traced on the surface's own exact x/y grid.")
                )
                curves = int(result.metadata.get("curves", 0))
                rows_3d = [
                    (
                        str(index + 1),
                        str(root.curve_index + 1),
                        report_html.format_number(root.x),
                        report_html.format_number(root.y),
                    )
                    for index, root in enumerate(result.roots)
                ]
                heading = f"{result.source_name} — {curves} curve(s), {len(result.roots)} point(s)"
                if result.metadata.get("truncated"):
                    heading = f"{heading} ({_('truncated')})"
                sections.append(
                    report_html.section(
                        heading,
                        grid_note,
                        report_html.table(
                            ("#", _("Curve"), "x", "y"),
                            rows_3d,
                            align=("right", "right", "right", "right"),
                        ),
                    )
                )
                continue

            rows = [
                (
                    str(index + 1),
                    report_html.format_number(root.x),
                    _("rising") if root.rising else _("falling"),
                    root.method,
                    str(root.iterations) if root.iterations else "-",
                    report_html.format_number(root.y - result.level, digits=3),
                )
                for index, root in enumerate(result.roots)
            ]
            heading = f"{result.source_name} — {len(result.roots)}"
            if result.metadata.get("truncated"):
                heading = f"{heading} ({_('truncated')})"

            sections.append(
                report_html.section(
                    heading,
                    report_html.table(
                        (
                            "#",
                            "x",
                            _("Direction"),
                            _("Solver"),
                            _("Iterations"),
                            _("Residual"),
                        ),
                        rows,
                        align=("right", "right", "left", "left", "right", "right"),
                    ),
                )
            )

        subtitle = _("level {level}").format(
            level=report_html.format_number(results[0].level)
        )
        # results[0].model, not self._model(): a 3D result's actual method
        # was always "Contour (matplotlib)" regardless of which 1D solver
        # happens to be selected in the combo right now.
        return report_html.document(
            _("Roots"), f"{results[0].model} — {subtitle}", *sections
        )
