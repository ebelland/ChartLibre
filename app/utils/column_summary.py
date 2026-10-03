"""The statistics shown beside a histogram made straight from one column.

The preview's "Histogram and statistics" makes a figure without the New
plot window and puts this summary in the figure's notes, where operation
reports go. The arithmetic is app.analysis.statistics.describe; this only
lays it out. No Qt here.
"""
from __future__ import annotations

import html
import math
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.statistics import describe
from app.utils.i18n import _


def numeric_values(values: Any) -> np.ndarray:
    """The values that are numbers, as floats; text and empty cells dropped."""
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return numbers[np.isfinite(numbers)]


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return "-" if math.isnan(number) else f"{number:.6g}"


def summary_html(column: str, values: Any) -> str:
    """An HTML table of *column*'s count, location, spread and shape."""
    total = len(values)
    numbers = numeric_values(values)
    stats = describe(numbers)
    rows: list[tuple[str, str]] = [
        (_("Values"), str(stats["n"])),
        (_("Empty or not a number"), str(total - int(stats["n"]))),
    ]
    if stats["n"]:
        rows += [
            (_("Mean"), _number(stats["mean"])),
            (_("95% CI of the mean"), f"{_number(stats['ci95_low'])} .. {_number(stats['ci95_high'])}"),
            (_("Standard deviation"), _number(stats["std"])),
            (_("Minimum"), _number(stats["min"])),
            (_("First quartile"), _number(stats["q1"])),
            (_("Median"), _number(stats["median"])),
            (_("Third quartile"), _number(stats["q3"])),
            (_("Maximum"), _number(stats["max"])),
            (_("Interquartile range"), _number(stats["iqr"])),
            (_("Skewness"), _number(stats["skewness"])),
            (_("Kurtosis"), _number(stats["kurtosis"])),
        ]
    body = "".join(
        f"<tr><td>{html.escape(label)}</td><td style='text-align:right'>{value}</td></tr>" for label, value in rows
    )
    title = html.escape(_("Statistics of '{column}'").format(column=column))
    return f"<h3>{title}</h3><table cellspacing='0' cellpadding='3'>{body}</table>"
