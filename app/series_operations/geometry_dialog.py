"""Move a series' coordinates: rotate, translate, scale, mirror, shear.

Unlike every other operation here, this one computes nothing into a table.
A rigid motion is exactly expressible in SQL, so the result is a *query* -
the source series' own query wrapped in the affine expressions - and the
series that gets created reads through it. Three consequences, all of them
the point:

* the transformed series follows its source, because it *is* its source
  read through an expression; edit the source query and the rotation
  rotates whatever the new rows are;
* nothing is duplicated, so a million-row series costs a query, not a
  second million rows in the project file;
* the transform is legible and editable afterwards in the Query Builder,
  which is where someone who wants 30.5 degrees instead of 30 will go.

Every transform here is affine::

    x' = a x + b y + e
    y' = c x + d y + f

so composing them is matrix multiplication and the SQL is always the same
two lines; only the six numbers, and how they are spelled, change. The
rotation spells its own out as ``cos(radians(30))`` rather than as
0.8660254 wherever SQLite can evaluate that, because the query is meant to
be read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.data.data_source import parse_roles, row_value
from app.logs.logger import applogger
from app.series_operations.dialog_base import ResultSeriesSpec, SeriesOperationDialogBase
from app.series_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam
from app.styles.style import CardFrame, create_doc_link
from app.utils.i18n import _

ROTATE = "Rotate"
TRANSLATE = "Translate"
SCALE = "Scale"
MIRROR = "Mirror"
SHEAR = "Shear"

MODELS: tuple[str, ...] = (ROTATE, TRANSLATE, SCALE, MIRROR, SHEAR)

#: Mirror axes, as the angle (degrees) of the mirror line through the centre.
MIRROR_LINES: tuple[tuple[str, float], ...] = (
    ("Horizontal line (flip up/down)", 0.0),
    ("Vertical line (flip left/right)", 90.0),
    ("Diagonal y = x", 45.0),
    ("Diagonal y = -x", 135.0),
)

#: Before and after, in the two colours the eye reads as "was" and "is".
BEFORE_COLOUR = "#2563EB"
AFTER_COLOUR = "#DC2626"


@dataclass(frozen=True, slots=True)
class Affine:
    """The six numbers of a 2D affine map, and how to spell them in SQL.

    ``a``..``f`` are always filled, so anything that needs to *compute* the
    transform (the preview plot) can. ``sql_x``/``sql_y`` carry the
    expression templates, which may spell a coefficient as a trigonometric
    call instead of as a decimal - the numbers and the spelling are the same
    map either way.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    sql_x: str
    sql_y: str

    def apply(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.a * x + self.b * y + self.e,
            self.c * x + self.d * y + self.f,
        )


@dataclass
class GeometryResult:
    """One transformed series: the query to create, and what it looks like."""

    source_name: str
    model: str
    transform: Affine
    sql: str
    x_role: str
    y_role: str
    roles: dict[str, Any] = field(default_factory=dict)
    before_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    before_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    after_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    after_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    show_grid: bool = True


#: Below this, a coefficient is floating-point noise rather than a number
#: anyone chose. The midpoint of data symmetric about zero comes out as
#: 1.1e-16 rather than 0, and writing that into the query makes a centre of
#: "the middle of the data" read as though it had been measured.
_NOISE = 1e-12


def _number(value: float) -> str:
    """Format a coefficient for SQL, for a person to read.

    Whole numbers lose their ".0", near-zero noise becomes 0, and anything
    else gets twelve significant digits - which is more than the transform
    needs and less than float repr's full seventeen.
    """
    if abs(value) < _NOISE:
        return "0"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12g}"


class _BeforeAfterView(QWidget):
    """The results pane: where the points were, and where they went.

    The HTML table the other operations show would say nothing useful here -
    a rotation is a picture, and "0.866, -0.5" is not one. Blue is the
    source, red is the result, and the optional grid is a square mesh over
    the source's own extent put through the same map, which is the thing
    that actually shows a shear or a scale for what it is.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(5.0, 4.0), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)
        self.clear(_("Choose a series, then press Preview."))

    def clear(self, message: str) -> None:
        self._figure.clear()
        axes = self._figure.add_subplot(111)
        axes.set_axis_off()
        axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axes.transAxes,
            color="#6B7280",
            wrap=True,
        )
        self._canvas.draw_idle()

    def show_result(self, result: GeometryResult) -> None:
        self._figure.clear()
        axes = self._figure.add_subplot(111)

        if result.show_grid and result.before_x.size:
            self._draw_grid(axes, result)

        axes.plot(
            result.before_x,
            result.before_y,
            marker="o",
            linestyle="none",
            markersize=4,
            color=BEFORE_COLOUR,
            label=_("Before"),
            zorder=3,
        )
        axes.plot(
            result.after_x,
            result.after_y,
            marker="o",
            linestyle="none",
            markersize=4,
            color=AFTER_COLOUR,
            label=_("After"),
            zorder=4,
        )

        axes.set_title(f"{result.model}: {result.source_name}")
        axes.set_xlabel(result.x_role)
        axes.set_ylabel(result.y_role)
        # Equal aspect or a rotation is not a rotation: a circle turned 30
        # degrees on unequal axes comes out an ellipse, and the one thing
        # this panel exists to show is the shape of the motion.
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, alpha=0.25)
        axes.legend(loc="best", fontsize="small")
        self._canvas.draw_idle()

    @staticmethod
    def _draw_grid(axes, result: GeometryResult, divisions: int = 6) -> None:
        """Put a square mesh over the source's extent through the same map."""
        x_min, x_max = float(result.before_x.min()), float(result.before_x.max())
        y_min, y_max = float(result.before_y.min()), float(result.before_y.max())
        # A series along a single row or column has no extent one way; give
        # it one so the mesh is a mesh rather than a line.
        if math.isclose(x_min, x_max):
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if math.isclose(y_min, y_max):
            y_min, y_max = y_min - 0.5, y_max + 0.5

        xs = np.linspace(x_min, x_max, divisions + 1)
        ys = np.linspace(y_min, y_max, divisions + 1)

        for value in xs:
            line_x = np.full_like(ys, value)
            axes.plot(line_x, ys, color=BEFORE_COLOUR, alpha=0.20, linewidth=0.8, zorder=1)
            moved_x, moved_y = result.transform.apply(line_x, ys)
            axes.plot(moved_x, moved_y, color=AFTER_COLOUR, alpha=0.25, linewidth=0.8, zorder=2)
        for value in ys:
            line_y = np.full_like(xs, value)
            axes.plot(xs, line_y, color=BEFORE_COLOUR, alpha=0.20, linewidth=0.8, zorder=1)
            moved_x, moved_y = result.transform.apply(xs, line_y)
            axes.plot(moved_x, moved_y, color=AFTER_COLOUR, alpha=0.25, linewidth=0.8, zorder=2)


class SeriesGeometryDialog(SeriesOperationDialogBase):
    """Rotate, move, scale, mirror or shear a series, as a query."""

    Name: str = "Geometry"
    Description = "Rotate, move, scale or mirror a series' coordinates"

    # One point is a perfectly good thing to rotate about another; nothing
    # here reads a neighbour, so none of the ordering rules apply.
    INPUT_MINIMUM_POINTS = 1

    #: A square with a rotated copy of itself behind it.
    Icon = """
    <rect x="3" y="9" width="10" height="10" rx="1"/>
    <path d="M14.5 4.5l6.2 6.2-6.2 6.2-6.2-6.2z"/>
    """

    PARAMS = (
        FloatParam(
            "angle",
            "Angle (degrees):",
            tooltip=(
                "Counter-clockwise, about the centre below. The query "
                "spells this out as cos(radians(...)) rather than as a "
                "decimal, so it can be read and edited afterwards."
            ),
            default_value=30.0,
            minimum=-360.0,
            maximum=360.0,
            decimals=4,
            step=15.0,
            visible_for={"model": (ROTATE,)},
        ),
        ChoiceParam(
            "mirror_line",
            "Mirror across:",
            tooltip="The line to reflect in, through the centre below.",
            choices=MIRROR_LINES,
            visible_for={"model": (MIRROR,)},
        ),
        FloatParam(
            "dx",
            "Move x by:",
            tooltip="Added to every x. Negative moves left.",
            default_value=0.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=1.0,
            visible_for={"model": (TRANSLATE,)},
        ),
        FloatParam(
            "dy",
            "Move y by:",
            tooltip="Added to every y. Negative moves down.",
            default_value=0.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=1.0,
            visible_for={"model": (TRANSLATE,)},
        ),
        FloatParam(
            "sx",
            "Scale x by:",
            tooltip="1 leaves x alone; 2 doubles it about the centre below.",
            default_value=1.0,
            minimum=-1.0e6,
            maximum=1.0e6,
            decimals=6,
            step=0.1,
            visible_for={"model": (SCALE,)},
        ),
        FloatParam(
            "sy",
            "Scale y by:",
            tooltip="1 leaves y alone; 2 doubles it about the centre below.",
            default_value=1.0,
            minimum=-1.0e6,
            maximum=1.0e6,
            decimals=6,
            step=0.1,
            visible_for={"model": (SCALE,)},
        ),
        FloatParam(
            "kx",
            "Shear x per y:",
            tooltip="Adds this much x for each unit of y above the centre.",
            default_value=0.0,
            minimum=-1.0e6,
            maximum=1.0e6,
            decimals=6,
            step=0.1,
            visible_for={"model": (SHEAR,)},
        ),
        FloatParam(
            "ky",
            "Shear y per x:",
            tooltip="Adds this much y for each unit of x right of the centre.",
            default_value=0.0,
            minimum=-1.0e6,
            maximum=1.0e6,
            decimals=6,
            step=0.1,
            visible_for={"model": (SHEAR,)},
        ),
        FloatParam(
            "cx",
            "Centre x:",
            tooltip=(
                "The fixed point. Rotation, scaling, mirroring and shearing "
                "all leave this point exactly where it is; a translation "
                "ignores it."
            ),
            default_value=0.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=1.0,
            visible_for={"model": (ROTATE, SCALE, MIRROR, SHEAR)},
        ),
        FloatParam(
            "cy",
            "Centre y:",
            tooltip="The other half of the fixed point.",
            default_value=0.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=1.0,
            visible_for={"model": (ROTATE, SCALE, MIRROR, SHEAR)},
        ),
        BoolParam(
            "use_data_centre",
            "Centre on the data",
            tooltip=(
                "Use the middle of the series' own extent as the fixed "
                "point instead of the two values above - which is what is "
                "wanted when the intent is 'turn this shape', not 'turn it "
                "about the origin'."
            ),
            default_value=True,
            visible_for={"model": (ROTATE, SCALE, MIRROR, SHEAR)},
        ),
        BoolParam(
            "show_grid",
            "Show the deformation grid",
            tooltip=(
                "Draws a square mesh over the source's extent and the same "
                "mesh after the transform. A rotation turns it, a scale "
                "stretches it, a shear leans it over - which is easier to "
                "read off the mesh than off the points."
            ),
            default_value=True,
        ),
    )

    def __init__(self, *, repo, figure_id: int, parent: QWidget | None = None) -> None:
        self._last_results: list[GeometryResult] = []
        self._sql_supports_trig: bool | None = None
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Geometry",
            parent=parent,
            width=900,
            height=660,
        )
        self.model_combo.addItems([_(name) for name in MODELS])
        self.series_selector.reload(select_all_series=True)
        self.mark_results_stale()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)

    def build_results_pane(self) -> QWidget:
        card = CardFrame(self, "operationResultsCard")
        self._plot_view = _BeforeAfterView(card)
        card.layout().addWidget(self._plot_view)
        return card

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(lambda *_a: self.mark_results_stale())
        self.series_selector.axis_changed.connect(lambda *_a: self.mark_results_stale())
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

    def _on_model_changed(self, *_args: Any) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        self.mark_results_stale()

    def mark_results_stale(self, *ignored: Any) -> None:
        super().mark_results_stale(*ignored)
        view = getattr(self, "_plot_view", None)
        if view is not None:
            view.clear(_("Press Preview to see where the points move."))

    # ------------------------------------------------------------------
    # The transform
    # ------------------------------------------------------------------

    def _current_model(self) -> str:
        text = self.model_combo.currentText()
        for name in MODELS:
            if text in (name, _(name)):
                return name
        return ROTATE

    def _supports_trig_sql(self) -> bool:
        """Say whether this database can evaluate cos()/radians() itself.

        SQLite only has the math functions when it was built with
        SQLITE_ENABLE_MATH_FUNCTIONS, which most builds now are and some
        still are not. Where they are missing the same rotation is written
        with its coefficients already worked out - the same map, spelled
        less helpfully, rather than a query that errors.
        """
        if self._sql_supports_trig is None:
            try:
                self._repo.query_df("SELECT cos(radians(0.0)) AS probe")
                self._sql_supports_trig = True
            except Exception:
                applogger.info(
                    "SQLite has no math functions here; writing the geometry "
                    "query with numeric coefficients instead."
                )
                self._sql_supports_trig = False
        return bool(self._sql_supports_trig)

    def _build_affine(self, values: Mapping[str, Any], cx: float, cy: float) -> Affine:
        """Return the affine map, and the SQL that spells it."""
        model = self._current_model()
        x_term = f'("{{x}}" - {_number(cx)})'
        y_term = f'("{{y}}" - {_number(cy)})'

        if model == TRANSLATE:
            dx = float(values.get("dx", 0.0))
            dy = float(values.get("dy", 0.0))
            return Affine(
                1.0, 0.0, 0.0, 1.0, dx, dy,
                sql_x=f'"{{x}}" + {_number(dx)}',
                sql_y=f'"{{y}}" + {_number(dy)}',
            )

        if model == SCALE:
            sx = float(values.get("sx", 1.0))
            sy = float(values.get("sy", 1.0))
            return Affine(
                sx, 0.0, 0.0, sy, cx * (1.0 - sx), cy * (1.0 - sy),
                sql_x=f"{_number(cx)} + {x_term} * {_number(sx)}",
                sql_y=f"{_number(cy)} + {y_term} * {_number(sy)}",
            )

        if model == SHEAR:
            kx = float(values.get("kx", 0.0))
            ky = float(values.get("ky", 0.0))
            return Affine(
                1.0, kx, ky, 1.0, -kx * cy, -ky * cx,
                sql_x=f"{_number(cx)} + {x_term} + {y_term} * {_number(kx)}",
                sql_y=f"{_number(cy)} + {y_term} + {x_term} * {_number(ky)}",
            )

        if model == MIRROR:
            phi = float(values.get("mirror_line", 0.0))
            double = math.radians(2.0 * phi)
            cos2, sin2 = math.cos(double), math.sin(double)
            if self._supports_trig_sql():
                cos_sql = f"cos(radians({_number(2.0 * phi)}))"
                sin_sql = f"sin(radians({_number(2.0 * phi)}))"
            else:
                cos_sql, sin_sql = _number(cos2), _number(sin2)
            return Affine(
                cos2, sin2, sin2, -cos2,
                cx - cx * cos2 - cy * sin2,
                cy - cx * sin2 + cy * cos2,
                sql_x=f"{_number(cx)} + {x_term} * {cos_sql} + {y_term} * {sin_sql}",
                sql_y=f"{_number(cy)} + {x_term} * {sin_sql} - {y_term} * {cos_sql}",
            )

        # Rotation, the default.
        angle = float(values.get("angle", 0.0))
        radians = math.radians(angle)
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        if self._supports_trig_sql():
            cos_sql = f"cos(radians({_number(angle)}))"
            sin_sql = f"sin(radians({_number(angle)}))"
        else:
            cos_sql, sin_sql = _number(cos_a), _number(sin_a)
        return Affine(
            cos_a, -sin_a, sin_a, cos_a,
            cx - cx * cos_a + cy * sin_a,
            cy - cx * sin_a - cy * cos_a,
            sql_x=f"{_number(cx)} + {x_term} * {cos_sql} - {y_term} * {sin_sql}",
            sql_y=f"{_number(cy)} + {x_term} * {sin_sql} + {y_term} * {cos_sql}",
        )

    @staticmethod
    def _build_sql(
        source_sql: str,
        transform: Affine,
        x_col: str,
        y_col: str,
        passthrough: Sequence[str],
    ) -> str:
        """Wrap *source_sql* in the transform, keeping every other column.

        The source goes in a CTE rather than an inline subquery so the
        generated query reads as "these rows, moved like this" - and so the
        source's own SQL, which may be long, stays in one piece and editable
        where the user left it.
        """
        x_expr = transform.sql_x.format(x=x_col, y=y_col)
        y_expr = transform.sql_y.format(x=x_col, y=y_col)
        columns = [
            f'    {x_expr} AS "{x_col}"',
            f'    {y_expr} AS "{y_col}"',
        ]
        columns.extend(f'    "{name}"' for name in passthrough)
        body = ",\n".join(columns)
        return (
            "WITH source AS (\n"
            f"{source_sql.strip().rstrip(';')}\n"
            ")\n"
            "SELECT\n"
            f"{body}\n"
            "FROM source"
        )

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute_results(self) -> list[GeometryResult]:
        rows = self.selected_series()
        if not rows:
            raise ValueError("select a series to transform")

        row = rows[0]
        name = self._series_display_name(row)
        source_sql = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not source_sql:
            raise ValueError("the series has no SQL query")

        frame = self._repo.query_df(source_sql)
        if frame.empty:
            raise ValueError("the series query returned no rows")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]

        x_col = str(roles.get("x") or "")
        y_col = str(roles.get("y") or "")
        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0]
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else x_col
        if x_col == y_col:
            raise ValueError(
                "this series has only one numeric column, so there is no "
                "second coordinate to move"
            )

        before_x = self.numeric_x(frame[x_col], name)
        before_y = self.numeric_y(frame[y_col])

        values = dict(self.parameter_values())
        if bool(values.get("use_data_centre", True)):
            cx = float((np.nanmin(before_x) + np.nanmax(before_x)) / 2.0)
            cy = float((np.nanmin(before_y) + np.nanmax(before_y)) / 2.0)
        else:
            cx = float(values.get("cx", 0.0))
            cy = float(values.get("cy", 0.0))

        transform = self._build_affine(values, cx, cy)
        after_x, after_y = transform.apply(before_x, before_y)
        passthrough = [c for c in columns if c not in (x_col, y_col)]
        sql = self._build_sql(source_sql, transform, x_col, y_col, passthrough)

        result = GeometryResult(
            source_name=name,
            model=self._current_model(),
            transform=transform,
            sql=sql,
            x_role=x_col,
            y_role=y_col,
            roles=dict(roles) or {"x": x_col, "y": y_col},
            before_x=before_x,
            before_y=before_y,
            after_x=after_x,
            after_y=after_y,
            show_grid=bool(values.get("show_grid", True)),
        )
        self._last_results = [result]
        return [result]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def format_results(self, results: Sequence[GeometryResult]) -> str:
        """Draw the before/after picture, and return the query as text.

        The pane is a plot rather than the shared HTML view, so what this
        returns is only what lands in the log and on the clipboard - which
        for this operation is the one thing worth copying, the SQL.
        """
        if results:
            self._plot_view.show_result(results[0])
        return "\n\n".join(result.sql for result in results)

    def publish_results(self, formatted: str) -> None:
        """Keep the plot; do not replace it with text.

        The base writes ``formatted`` into the shared HTML view, which this
        operation does not show. Left to run, it would be writing into a
        widget that is not on screen - harmless, but it also logs the text
        as the result, and the result here is the picture beside it.
        """
        applogger.info("Geometry query:\n%s", formatted)

    def results_report_html(self, formatted: str, results: Sequence[Any]) -> str:
        del results
        return formatted

    # ------------------------------------------------------------------
    # Writing the series
    # ------------------------------------------------------------------

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_geometry": True, "geometry_dialog": "series_geometry"}

    def result_to_frame(self, result: GeometryResult) -> pd.DataFrame:
        """The transformed rows.

        Nothing in this operation's own path calls this - it writes a query,
        not a table - but the base class offers it to anything that wants
        the numbers, and computing them here beats raising.
        """
        return pd.DataFrame({result.x_role: result.after_x, result.y_role: result.after_y})

    def result_series_spec(
        self, axis_id: int, table_name: str, result: GeometryResult
    ) -> ResultSeriesSpec:
        del axis_id, table_name
        style = dict(self.generated_style_filter)
        style["color"] = AFTER_COLOUR
        return ResultSeriesSpec(
            name=f"{result.source_name} ({_(result.model).lower()})",
            sql_query=result.sql,
            roles={"x": result.x_role, "y": result.y_role},
            style=style,
        )

    def apply_results_to_axis(self, axis_id: int, results: Sequence[GeometryResult]) -> None:
        """Create the series, and no table.

        The base writes each result to a table and points a series at it.
        There is nothing to write here: the series' own query carries the
        transform, so the only artifact is the descriptor.
        """
        self.remove_previous_generated_series(axis_id)
        for result in results:
            self.create_result_series(axis_id, "", result)

    def preview_results_to_axis(self, axis_id: int, results: Sequence[GeometryResult]) -> None:
        """Same again for the preview: a descriptor, no table."""
        self.remove_preview_artifacts(axis_id)
        for result in results:
            self.create_preview_series(axis_id, "", result)
