"""Ishikawa (fishbone) diagram: the causes of one problem, grouped by category.

https://matplotlib.org/stable/gallery/specialty_plots/ishikawa_diagram.html

Role columns, one row per cause:
    category    required, the bone the cause hangs from (Method, Machine...)
    cause       required, the cause itself
    subcause    optional, a finer cause written under its cause

The problem - the fish's head - is the "problem" option, or the series name
when that is empty. Nothing here is numeric: the layout is computed from how
many categories and causes there are, and the axes' own frame is switched off.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.charts.base import BaseAxisRenderer, SeriesData


class FishboneAxisRenderer(BaseAxisRenderer):
    """Causes grouped on bones along a spine that ends in the problem."""

    Name: str = "Fishbone Diagram"
    Category: str = "Statistical distributions"
    Description: str = (
        "Ishikawa cause-and-effect diagram: one row per cause, grouped by "
        "category on bones that lead to the problem."
    )
    Link: str = "https://matplotlib.org/stable/gallery/specialty_plots/ishikawa_diagram.html"

    RequiredRoles: list[str] = ["category", "cause"]
    OptionalRoles: list[str] = ["subcause"]

    Kwargs: dict[str, object] = {
        "problem": {
            "default": "",
            "type": str,
            "group": "Fishbone",
            "description": "The problem written in the head. Empty uses the series name.",
        },
        "fontsize": {
            "default": 9.0,
            "type": float,
            "min": 5.0,
            "max": 30.0,
            "group": "Fishbone",
            "description": "Size of the cause text; categories and the problem are drawn larger.",
        },
        "bone_color": {
            "default": "#4A4A4A",
            "type": str,
            "kind": "color",
            "group": "Fishbone",
            "description": "Colour of the spine and the bones.",
        },
        "head_color": {
            "default": "#F4C542",
            "type": str,
            "kind": "color",
            "group": "Fishbone",
            "description": "Fill of the problem box at the head.",
        },
        "category_color": {
            "default": "#DCE8F5",
            "type": str,
            "kind": "color",
            "group": "Fishbone",
            "description": "Fill of the category boxes at the ends of the bones.",
        },
    }

    #: Layout units: each bone column is this wide, bones rise this high.
    _COLUMN_WIDTH: float = 6.0
    _BONE_HEIGHT: float = 4.0
    _HEAD_WIDTH: float = 4.0

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        valid = self.valid_series(series)
        if not valid:
            return
        sd = valid[0]
        merged = self.merge_style(axis_options, sd.style or {})

        groups = self._groups(sd.df)
        if not groups:
            return

        fontsize = float(self.opt("fontsize", merged) or 9.0)
        bone_color = str(self.opt("bone_color", merged) or "#4A4A4A")
        head_color = str(self.opt("head_color", merged) or "#F4C542")
        category_color = str(self.opt("category_color", merged) or "#DCE8F5")
        problem = str(self.opt("problem", merged) or "").strip() or str(sd.name or "")

        columns = max(1, math.ceil(len(groups) / 2))
        width = self._COLUMN_WIDTH
        spine_end = columns * width
        ax.plot([0.0, spine_end], [0.0, 0.0], color=bone_color, linewidth=2.5, zorder=1)
        ax.annotate(
            "",
            xy=(spine_end + 0.25, 0.0),
            xytext=(spine_end - 0.6, 0.0),
            arrowprops={"arrowstyle": "-|>", "color": bone_color, "linewidth": 2.5},
        )
        ax.text(
            spine_end + 0.4, 0.0, _wrap(problem, 16),
            ha="left", va="center", fontsize=fontsize * 1.3, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": head_color, "edgecolor": bone_color},
        )

        for index, (category, causes) in enumerate(groups):
            column = index // 2
            side = 1.0 if index % 2 == 0 else -1.0
            # The first categories sit nearest the head, as in the classic drawing.
            base_x = spine_end - column * width - 0.15 * width
            tip_x = base_x - 0.45 * width
            tip_y = side * self._BONE_HEIGHT
            ax.plot([base_x, tip_x], [0.0, tip_y], color=bone_color, linewidth=1.8, zorder=1)
            ax.text(
                tip_x, tip_y + side * 0.35, _wrap(category, 18),
                ha="center", va="bottom" if side > 0 else "top",
                fontsize=fontsize * 1.15, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": category_color, "edgecolor": bone_color},
            )
            for position, (cause, subcauses) in enumerate(causes):
                # Spread along the bone, the first cause nearest its tip.
                t = 1.0 - (position + 0.75) / (len(causes) + 0.5)
                x = base_x + (tip_x - base_x) * t
                y = tip_y * t
                twig_start = x - 0.42 * width
                ax.plot([twig_start, x], [y, y], color=bone_color, linewidth=0.9, zorder=1)
                # Cause above its twig, subcauses under it, all ending at the bone.
                ax.text(x - 0.1, y + 0.08, _wrap(cause, 18), ha="right", va="bottom", fontsize=fontsize)
                if subcauses:
                    ax.text(
                        x - 0.1, y - 0.08,
                        "\n".join(f"\u2013 {_wrap(sub, 20)}" for sub in subcauses),
                        ha="right", va="top", fontsize=fontsize * 0.8, color=bone_color,
                    )

        ax.set_xlim(-0.2 * width, spine_end + 0.4 + self._HEAD_WIDTH)
        ax.set_ylim(-self._BONE_HEIGHT - 1.4, self._BONE_HEIGHT + 1.4)
        ax.set_axis_off()
        ax._dhub_axis_off = True  # noqa: SLF001 - read by render_figure

    @staticmethod
    def _groups(df: Any) -> list[tuple[str, list[tuple[str, list[str]]]]]:
        """[(category, [(cause, [subcauses])])], in order of first appearance."""
        frame = pd.DataFrame(
            {
                "category": pd.Series(df["category"]).astype(object),
                "cause": pd.Series(df["cause"]).astype(object),
                "subcause": (
                    pd.Series(df["subcause"]).astype(object)
                    if "subcause" in df.columns
                    else pd.Series([None] * len(df), dtype=object)
                ),
            }
        )
        groups: dict[str, dict[str, list[str]]] = {}
        for category, cause, subcause in frame.itertuples(index=False):
            if category is None or cause is None or str(cause).strip() == "":
                continue
            causes = groups.setdefault(str(category).strip(), {})
            subs = causes.setdefault(str(cause).strip(), [])
            if subcause is not None and str(subcause).strip() not in ("", "nan", "None"):
                subs.append(str(subcause).strip())
        return [(category, list(causes.items())) for category, causes in groups.items()]


def _wrap(text: str, width: int) -> str:
    """Break long text onto lines of about *width* characters."""
    import textwrap

    return "\n".join(textwrap.wrap(str(text), width=width)) or str(text)
