"""Move a series' coordinates: rotate, translate, roto-translate, scale, mirror, shear.

Unlike every other operation here, this one computes nothing into a table.
A rigid motion is exactly expressible in SQL, so the result is a *query* -
the source series' own query wrapped in the motion's expressions - and the
series that gets created reads through it. Three consequences, all of them
the point:

* the transformed series follows its source, because it *is* its source
  read through an expression; edit the source query and the rotation
  rotates whatever the new rows are;
* nothing is duplicated, so a million-row series costs a query, not a
  second million rows in the project file;
* the transform is legible and editable afterwards in the Query Builder,
  which is where someone who wants 30.5 degrees instead of 30 will go.

A series with a z role is moved in 3D: three rotation angles (about x, then
y, then z), a translation and a scale along z as well, and a 3D preview.
Mirror and shear act in the x-y plane and leave z alone. The arithmetic -
matrices, composition, the SQL - is app.analysis.geometry; this module reads
the series and the parameters and shows the result. A 2D rotation spells its
coefficients as ``cos(radians(30))`` wherever SQLite can evaluate that,
because the query is meant to be read.
"""
from __future__ import annotations

import base64
import html
import io
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from app.widgets import mpl_cursors  # noqa: F401 - native cursors on macOS
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.analysis import geometry as geo
from app.data.data_source import parse_roles, quote_identifier, row_value, resolve_role_column
from app.logs.logger import applogger
from app.series_operations.results import OperationResult
from app.series_operations.dialog_base import OperationModel, ResultSeriesSpec, SeriesOperationDialogBase
from app.series_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam
from app.styles.style import CardFrame
from app.utils import report_html
from app.utils.config import get_constant
from app.utils.i18n import _

ROTATE = "Rotate"
TRANSLATE = "Translate"
ROTO_TRANSLATE = "Roto-translation"
SCALE = "Scale"
MIRROR = "Mirror"
SHEAR = "Shear"


@dataclass(frozen=True, slots=True, kw_only=True)
class GeometryModel(OperationModel):
    #: Turns by an angle (or three, in 3D).
    rotates: bool = False
    #: Moves by an offset.
    moves: bool = False
    #: Works about a fixed point: every model but a plain translation.
    centred: bool = True


#: The motions offered, in combo order.
GEOMETRY_MODELS: dict[str, GeometryModel] = {
    ROTATE: GeometryModel(rotates=True),
    TRANSLATE: GeometryModel(moves=True, centred=False),
    ROTO_TRANSLATE: GeometryModel(rotates=True, moves=True),
    SCALE: GeometryModel(),
    MIRROR: GeometryModel(),
    SHEAR: GeometryModel(),
}
#: The same, as the name lists visible_for rules compare against.
_ROTATING = tuple(name for name, model in GEOMETRY_MODELS.items() if model.rotates)
_MOVING = tuple(name for name, model in GEOMETRY_MODELS.items() if model.moves)
_CENTRED = tuple(name for name, model in GEOMETRY_MODELS.items() if model.centred)

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


@dataclass
class GeometryResult(OperationResult):
    """One transformed series: the query to create, and what it looks like."""

    source_name: str
    model: str
    transform: geo.Motion
    sql: str
    x_role: str
    y_role: str
    z_role: str | None = None
    roles: dict[str, Any] = field(default_factory=dict)
    before_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    before_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    before_z: np.ndarray | None = None
    after_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    after_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    after_z: np.ndarray | None = None
    show_grid: bool = True
    #: What the report lists: the settings that shaped this motion, as text.
    settings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_3d(self) -> bool:
        return self.z_role is not None

    def to_df(self) -> pd.DataFrame:
        """The transformed rows.

        Nothing in this operation's own path calls this - it writes a query,
        not a table - but any result can be asked for its numbers.
        """
        data = {self.x_role: self.after_x, self.y_role: self.after_y}
        if self.z_role is not None and self.after_z is not None:
            data[self.z_role] = self.after_z
        return pd.DataFrame(data)

    # A series and no table: its own query carries the transform, so the
    # descriptor is the only thing to write.
    def preview(self, dialog: Any, axis_id: int) -> None:
        dialog.create_preview_series(axis_id, "", self)

    def apply(self, dialog: Any, axis_id: int) -> None:
        dialog.create_result_series(axis_id, "", self)


#: Dots per inch of the before/after picture: the charts' on-screen dpi.
PREVIEW_DPI: float = get_constant("fixed_mode_screen_dpi", 100.0)


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
        # Its own dpi: the charts' (figure.dpi, 200 by default) is meant for
        # export, and taken from rcParams it drew this picture twice as large.
        self._figure = Figure(figsize=(5.0, 4.0), dpi=PREVIEW_DPI, layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)
        self.clear(_("Choose a series, then press Preview."))

    @property
    def figure(self) -> Figure:
        """The picture as drawn now - what the report embeds."""
        return self._figure

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
        if result.is_3d:
            self._show_3d(result)
            return
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

        axes.set_title(f"{_(result.model)}: {result.source_name}")
        axes.set_xlabel(result.x_role)
        axes.set_ylabel(result.y_role)
        # Equal aspect or a rotation is not a rotation: a circle turned 30
        # degrees on unequal axes comes out an ellipse, and the one thing
        # this panel exists to show is the shape of the motion.
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, alpha=0.25)
        axes.legend(loc="best", fontsize="small")
        self._canvas.draw_idle()

    def _show_3d(self, result: GeometryResult) -> None:
        """Before and after in three dimensions, on equal axes."""
        axes = self._figure.add_subplot(111, projection="3d")
        assert result.before_z is not None and result.after_z is not None
        if result.show_grid and result.before_x.size:
            self._draw_grid_3d(axes, result)
        axes.scatter(result.before_x, result.before_y, result.before_z, s=10, color=BEFORE_COLOUR, label=_("Before"))
        axes.scatter(result.after_x, result.after_y, result.after_z, s=10, color=AFTER_COLOUR, label=_("After"))
        axes.set_title(f"{_(result.model)}: {result.source_name}")
        axes.set_xlabel(result.x_role)
        axes.set_ylabel(result.y_role)
        axes.set_zlabel(result.z_role or "z")
        # Equal box, for the same reason as the 2D equal aspect.
        everything = [
            np.concatenate([result.before_x, result.after_x]),
            np.concatenate([result.before_y, result.after_y]),
            np.concatenate([result.before_z, result.after_z]),
        ]
        axes.set_box_aspect([max(float(np.ptp(v)), 1e-9) for v in everything])
        axes.legend(loc="best", fontsize="small")
        self._canvas.draw_idle()

    @staticmethod
    def _extent(values: np.ndarray) -> tuple[float, float]:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        # A series along a single row or column has no extent one way; give
        # it one so the mesh is a mesh rather than a line.
        if math.isclose(low, high):
            return low - 0.5, high + 0.5
        return low, high

    @classmethod
    def _draw_grid(cls, axes, result: GeometryResult, divisions: int = 6) -> None:
        """Put a square mesh over the source's extent through the same map."""
        xs = np.linspace(*cls._extent(result.before_x), divisions + 1)
        ys = np.linspace(*cls._extent(result.before_y), divisions + 1)
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

    @classmethod
    def _draw_grid_3d(cls, axes, result: GeometryResult, divisions: int = 4) -> None:
        """The source's bounding box as a wire cage, and the same cage moved."""
        assert result.before_z is not None
        xs = cls._extent(result.before_x)
        ys = cls._extent(result.before_y)
        zs = cls._extent(result.before_z)
        del divisions
        edges = []
        for a in xs:
            for b in ys:
                edges.append(((a, a), (b, b), zs))
        for a in xs:
            for c in zs:
                edges.append(((a, a), ys, (c, c)))
        for b in ys:
            for c in zs:
                edges.append((xs, (b, b), (c, c)))
        for ex, ey, ez in edges:
            ex, ey, ez = (np.asarray(v, dtype=float) for v in (ex, ey, ez))
            axes.plot(ex, ey, ez, color=BEFORE_COLOUR, alpha=0.25, linewidth=0.8)
            mx, my, mz = result.transform.apply(ex, ey, ez)
            axes.plot(mx, my, mz, color=AFTER_COLOUR, alpha=0.3, linewidth=0.8)


class SeriesGeometryDialog(SeriesOperationDialogBase):
    """Rotate, move, scale, mirror or shear a series, in 2D or 3D, as a query."""

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

    #: ``dims`` in a visible_for rule is not a parameter: it is "2D" or "3D"
    #: from the selected series' roles (see parameter_context), so a 3D
    #: series shows three angles and a z offset, and a 2D one does not.
    PARAMS = (
        FloatParam(
            "angle",
            "Angle (degrees):",
            tooltip=(
                "Counter-clockwise, about the centre below. The query "
                "spells this out as cos(radians(...)) rather than as a "
                "decimal, so it can be read and edited afterwards."
            ),
            default_value=30.0, minimum=-360.0, maximum=360.0, decimals=4, step=15.0,
            visible_for={"model": _ROTATING, "dims": ("2D",)},
        ),
        FloatParam(
            "angle_x",
            "Rotation about x (degrees):",
            tooltip="Applied first. Positive turns y towards z (right-hand rule).",
            default_value=0.0, minimum=-360.0, maximum=360.0, decimals=4, step=15.0,
            visible_for={"model": _ROTATING, "dims": ("3D",)},
        ),
        FloatParam(
            "angle_y",
            "Rotation about y (degrees):",
            tooltip="Applied second. Positive turns z towards x.",
            default_value=0.0, minimum=-360.0, maximum=360.0, decimals=4, step=15.0,
            visible_for={"model": _ROTATING, "dims": ("3D",)},
        ),
        FloatParam(
            "angle_z",
            "Rotation about z (degrees):",
            tooltip="Applied last. Positive turns x towards y - the 2D rotation.",
            default_value=30.0, minimum=-360.0, maximum=360.0, decimals=4, step=15.0,
            visible_for={"model": _ROTATING, "dims": ("3D",)},
        ),
        ChoiceParam(
            "mirror_line",
            "Mirror across:",
            tooltip="The line to reflect in, through the centre below. In 3D, z is left alone.",
            choices=MIRROR_LINES,
            visible_for={"model": (MIRROR,)},
        ),
        FloatParam(
            "dx",
            "Move x by:",
            tooltip="Added to every x, after any rotation. Negative moves left.",
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
            visible_for={"model": _MOVING},
        ),
        FloatParam(
            "dy",
            "Move y by:",
            tooltip="Added to every y, after any rotation. Negative moves down.",
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
            visible_for={"model": _MOVING},
        ),
        FloatParam(
            "dz",
            "Move z by:",
            tooltip="Added to every z, after any rotation.",
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
            visible_for={"model": _MOVING, "dims": ("3D",)},
        ),
        FloatParam(
            "sx",
            "Scale x by:",
            tooltip="1 leaves x alone; 2 doubles it about the centre below.",
            default_value=1.0, minimum=-1.0e6, maximum=1.0e6, decimals=6, step=0.1,
            visible_for={"model": (SCALE,)},
        ),
        FloatParam(
            "sy",
            "Scale y by:",
            tooltip="1 leaves y alone; 2 doubles it about the centre below.",
            default_value=1.0, minimum=-1.0e6, maximum=1.0e6, decimals=6, step=0.1,
            visible_for={"model": (SCALE,)},
        ),
        FloatParam(
            "sz",
            "Scale z by:",
            tooltip="1 leaves z alone; 2 doubles it about the centre below.",
            default_value=1.0, minimum=-1.0e6, maximum=1.0e6, decimals=6, step=0.1,
            visible_for={"model": (SCALE,), "dims": ("3D",)},
        ),
        FloatParam(
            "kx",
            "Shear x per y:",
            tooltip="Adds this much x for each unit of y above the centre.",
            default_value=0.0, minimum=-1.0e6, maximum=1.0e6, decimals=6, step=0.1,
            visible_for={"model": (SHEAR,)},
        ),
        FloatParam(
            "ky",
            "Shear y per x:",
            tooltip="Adds this much y for each unit of x right of the centre.",
            default_value=0.0, minimum=-1.0e6, maximum=1.0e6, decimals=6, step=0.1,
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
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
            visible_for={"model": _CENTRED},
        ),
        FloatParam(
            "cy",
            "Centre y:",
            tooltip="The other half of the fixed point.",
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
            visible_for={"model": _CENTRED},
        ),
        FloatParam(
            "cz",
            "Centre z:",
            tooltip="The fixed point's z.",
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12, decimals=6, step=1.0,
            visible_for={"model": _CENTRED, "dims": ("3D",)},
        ),
        BoolParam(
            "use_data_centre",
            "Centre on the data",
            tooltip=(
                "Use the middle of the series' own extent as the fixed "
                "point instead of the values above - which is what is "
                "wanted when the intent is 'turn this shape', not 'turn it "
                "about the origin'."
            ),
            default_value=True,
            visible_for={"model": _CENTRED},
        ),
        BoolParam(
            "show_grid",
            "Show the deformation grid",
            tooltip=(
                "Draws a square mesh (a box in 3D) over the source's extent "
                "and the same mesh after the transform. A rotation turns it, "
                "a scale stretches it, a shear leans it over - which is "
                "easier to read off the mesh than off the points."
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
        # Shown translated, keyed by the untranslated name: the parameters'
        # visible_for rules read the key (see parameter_context).
        for name in GEOMETRY_MODELS:
            self.model_combo.addItem(_(name), name)
        self.series_selector.reload(select_all_series=True)
        self._on_model_changed()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_results_pane(self) -> QWidget:
        card = CardFrame(self, "operationResultsCard")
        self._plot_view = _BeforeAfterView(card)
        card.layout().addWidget(self._plot_view)
        return card

    def connect_operation_signals(self) -> None:
        # A different series may be 2D or 3D, which changes the parameters.
        self.series_selector.selection_changed.connect(lambda *_a: self._on_model_changed())
        self.series_selector.axis_changed.connect(lambda *_a: self._on_model_changed())
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

    def _on_model_changed(self, *_args: Any) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        self.mark_results_stale()

    def parameter_context(self) -> Mapping[str, Any]:
        return {**super().parameter_context(), "dims": "3D" if self._z_role_of_selection() else "2D"}

    def _z_role_of_selection(self) -> str | None:
        """The selected series' z column, or None for a 2D series."""
        selector = getattr(self, "series_selector", None)
        rows = selector.selected_series() if selector is not None else []
        if not rows:
            return None
        roles = parse_roles(row_value(rows[0], "roles", default={}))
        return str(roles.get("z") or "") or None

    def mark_results_stale(self, *ignored: Any) -> None:
        super().mark_results_stale(*ignored)
        view = getattr(self, "_plot_view", None)
        if view is not None:
            view.clear(_("Press Preview to see where the points move."))

    # ------------------------------------------------------------------
    # The transform
    # ------------------------------------------------------------------

    def _current_model(self) -> str:
        return self.current_model(ROTATE)

    def _supports_trig_sql(self) -> bool:
        """Say whether this database can evaluate cos()/radians() itself.

        SQLite only has the math functions when it was built with
        SQLITE_ENABLE_MATH_FUNCTIONS, which most builds now are and some
        still are not. Where they are missing the same rotation is written
        with its coefficients already worked out - the same map, spelled
        less helpfully, rather than a query that errors.
        """
        if self._sql_supports_trig is None:
            self._sql_supports_trig = self._repo.supports_sql_math()
            if not self._sql_supports_trig:
                applogger.info(
                    "SQLite has no math functions here; writing the geometry "
                    "query with numeric coefficients instead."
                )
        return bool(self._sql_supports_trig)

    def _build_motion(
        self, values: Mapping[str, Any], centre: tuple[float, float, float], is_3d: bool
    ) -> tuple[geo.Motion, Any]:
        """The motion the parameters describe, and how to spell its matrix in SQL."""
        model = self._current_model()
        spell = None
        if GEOMETRY_MODELS[model].rotates:
            if is_3d:
                matrix = geo.rotation_3d(
                    float(values.get("angle_x", 0.0)),
                    float(values.get("angle_y", 0.0)),
                    float(values.get("angle_z", 0.0)),
                )
            else:
                angle = float(values.get("angle", 0.0))
                matrix = geo.rotation_2d(angle)
                if self._supports_trig_sql():
                    cos_sql = f"cos(radians({geo.number(angle)}))"
                    sin_sql = f"sin(radians({geo.number(angle)}))"
                    spelled = {(0, 0): cos_sql, (0, 1): f"-{sin_sql}", (1, 0): sin_sql, (1, 1): cos_sql}
                    spell = lambda row, col: spelled.get((row, col))  # noqa: E731
            translation = (
                (float(values.get("dx", 0.0)), float(values.get("dy", 0.0)), float(values.get("dz", 0.0)))
                if model == ROTO_TRANSLATE
                else (0.0, 0.0, 0.0)
            )
            return geo.Motion.of(matrix, centre=centre, translation=translation), spell
        if model == TRANSLATE:
            translation = (float(values.get("dx", 0.0)), float(values.get("dy", 0.0)), float(values.get("dz", 0.0)))
            return geo.Motion.of(translation=translation), None
        if model == SCALE:
            matrix = geo.scale(float(values.get("sx", 1.0)), float(values.get("sy", 1.0)), float(values.get("sz", 1.0)))
            return geo.Motion.of(matrix, centre=centre), None
        if model == SHEAR:
            matrix = geo.shear_xy(float(values.get("kx", 0.0)), float(values.get("ky", 0.0)))
            return geo.Motion.of(matrix, centre=centre), None
        # Mirror.
        return geo.Motion.of(geo.mirror_xy(float(values.get("mirror_line", 0.0))), centre=centre), None

    @staticmethod
    def _build_sql(
        source_sql: str,
        transform: geo.Motion,
        columns: Sequence[str],
        passthrough: Sequence[str],
        spell: Any = None,
    ) -> str:
        """Wrap *source_sql* in the transform, keeping every other column.

        *columns* are the coordinates - x and y, and z for a 3D series. The
        source goes in a CTE rather than an inline subquery so the generated
        query reads as "these rows, moved like this" - and so the source's
        own SQL, which may be long, stays in one piece and editable where
        the user left it.
        """
        quoted = tuple(quote_identifier(name) for name in columns)
        expressions = transform.sql(quoted, spell)
        lines = [f"    {expression} AS {name}" for expression, name in zip(expressions, quoted)]
        lines.extend(f"    {quote_identifier(name)}" for name in passthrough)
        body = ",\n".join(lines)
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

        x_col = resolve_role_column(columns, roles, "x") or ""
        y_col = resolve_role_column(columns, roles, "y") or ""
        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0]
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else x_col
        if x_col == y_col:
            raise ValueError(
                "this series has only one numeric column, so there is no "
                "second coordinate to move"
            )
        z_col = str(roles.get("z") or "") or None
        if z_col is not None and z_col not in columns:
            z_col = None

        before_x = self.numeric_x(frame[x_col], name)
        before_y = self.numeric_y(frame[y_col])
        before_z = self.numeric_y(frame[z_col]) if z_col is not None else None

        values = dict(self.parameter_values())
        if bool(values.get("use_data_centre", True)):
            def middle(v: np.ndarray | None) -> float:
                return float((np.nanmin(v) + np.nanmax(v)) / 2.0) if v is not None else 0.0

            centre = (middle(before_x), middle(before_y), middle(before_z))
        else:
            centre = (
                float(values.get("cx", 0.0)),
                float(values.get("cy", 0.0)),
                float(values.get("cz", 0.0)) if z_col is not None else 0.0,
            )

        transform, spell = self._build_motion(values, centre, z_col is not None)
        if z_col is not None:
            after_x, after_y, after_z = transform.apply(before_x, before_y, before_z)
            coordinates = [x_col, y_col, z_col]
        else:
            after_x, after_y = transform.apply(before_x, before_y)
            after_z = None
            coordinates = [x_col, y_col]
        passthrough = [c for c in columns if c not in coordinates]
        sql = self._build_sql(source_sql, transform, coordinates, passthrough, spell)

        default_roles = {"x": x_col, "y": y_col, **({"z": z_col} if z_col else {})}
        result = GeometryResult(
            source_name=name,
            model=self._current_model(),
            transform=transform,
            sql=sql,
            x_role=x_col,
            y_role=y_col,
            z_role=z_col,
            roles=dict(roles) or default_roles,
            before_x=before_x,
            before_y=before_y,
            before_z=before_z,
            after_x=after_x,
            after_y=after_y,
            after_z=after_z,
            show_grid=bool(values.get("show_grid", True)),
            settings=self._report_settings(self._current_model(), values, centre, z_col is not None, before_x.size),
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
        """The report under the chart: what was done, the picture, then the query.

        It used to be the query alone, which says what to run but not what
        happened; the before/after picture is the result of this operation,
        so it goes in too, as a small embedded image.
        """
        del formatted
        blocks: list[str] = []
        for result in results:
            summary = report_html.summary_table(
                [(_("Motion"), _(result.model)), *result.settings]
            )
            image = self._picture_markup()
            query = (
                "<pre style='white-space:pre-wrap;font-family:Menlo,Consolas,monospace;"
                f"font-size:9pt;margin:0;'>{html.escape(result.sql)}</pre>"
            )
            blocks.append(
                report_html.section(result.source_name, summary, image, report_html.section(_("Query"), query))
            )
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"<h3 style='margin:0 0 4px 0;'>{html.escape(self.operation_label)}</h3>"
            f"<p style='margin:0 0 8px 0;color:#666;font-size:9pt;'>{stamp}</p>"
            + "".join(blocks)
        )

    def _picture_markup(self) -> str:
        """The before/after plot now in the results pane, as an embedded PNG."""
        buffer = io.BytesIO()
        try:
            self._plot_view.figure.savefig(buffer, format="png", dpi=80)
        except Exception:  # noqa: BLE001 - a report without its picture is still a report
            applogger.exception("Could not render the geometry picture for the report")
            return ""
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"<p><img src='data:image/png;base64,{encoded}' width='400'></p>"

    @staticmethod
    def _report_settings(
        model_name: str,
        values: Mapping[str, Any],
        centre: tuple[float, float, float],
        is_3d: bool,
        points: int,
    ) -> list[tuple[str, str]]:
        """The settings of the current model, as label/value rows."""
        def num(key: str, default: float = 0.0) -> str:
            return geo.number(float(values.get(key, default)))

        model = GEOMETRY_MODELS.get(model_name)
        rows: list[tuple[str, str]] = []
        if model is not None and model.rotates:
            if is_3d:
                rows.append((_("Angles (x, y, z)"), f"{num('angle_x')}°, {num('angle_y')}°, {num('angle_z')}°"))
            else:
                rows.append((_("Angle"), f"{num('angle')}°"))
        if model is not None and model.moves:
            offset = f"{num('dx')}, {num('dy')}" + (f", {num('dz')}" if is_3d else "")
            rows.append((_("Offset"), offset))
        if model_name == SCALE:
            rows.append((_("Scale"), f"{num('sx', 1)}, {num('sy', 1)}" + (f", {num('sz', 1)}" if is_3d else "")))
        if model_name == SHEAR:
            rows.append((_("Shear"), f"{num('kx')}, {num('ky')}"))
        if model_name == MIRROR:
            line = float(values.get("mirror_line", 0.0))
            rows.append((_("Mirror line"), next((_(label) for label, angle in MIRROR_LINES if angle == line), f"{line:g}°")))
        if model is None or model.centred:
            centre_text = ", ".join(geo.number(v) for v in (centre if is_3d else centre[:2]))
            rows.append((_("Centre"), centre_text))
        rows.append((_("Points"), str(points)))
        return rows

    # ------------------------------------------------------------------
    # Writing the series
    # ------------------------------------------------------------------

    def result_series_spec(
        self, axis_id: int, table_name: str, result: GeometryResult
    ) -> ResultSeriesSpec:
        del axis_id, table_name
        style = dict(self.generated_style_filter)
        style["color"] = AFTER_COLOUR
        # Every role the source had, not just x and y. The query already
        # carries the other columns through untouched (see _build_sql), so
        # naming only the two coordinates here would leave a scatter's
        # colour and size columns present in the rows and unused by the
        # chart - the moved series would draw in flat colour beside a
        # source that is coloured by value, which reads as the transform
        # having thrown the data away.
        roles = dict(result.roles)
        roles["x"] = result.x_role
        roles["y"] = result.y_role
        if result.z_role is not None:
            roles["z"] = result.z_role
        return ResultSeriesSpec(
            name=f"{result.source_name} ({_(result.model).lower()})",
            sql_query=result.sql,
            roles=roles,
            style=style,
        )
