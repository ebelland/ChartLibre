"""Read the NIST Statistical Reference Datasets in dev/tests/data/nist (todo R-13).

The files are the originals from https://www.itl.nist.gov/div898/strd/,
unchanged: certified values and data together, so a test can say where
every number it checks comes from. NIST publishes them as a public-domain
yardstick - what the commercial packages are measured against.

Agreement is measured as NIST does, in correct significant digits: the log
relative error ``LRE = -log10(|estimate - certified| / |certified|)``
(absolute when the certified value is 0), capped at the digits certified.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "nist"

#: The certified values carry 15 significant digits for the linear,
#: univariate and ANOVA sets and 11 for the nonlinear ones: past that a
#: difference is the certificate's rounding, not an error.
CERTIFIED_DIGITS = 15.0
NONLINEAR_DIGITS = 11.0

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def _float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def lre(estimate: float, certified: float, digits: float = CERTIFIED_DIGITS) -> float:
    """Correct significant digits of *estimate*, NIST's log relative error."""
    if not math.isfinite(estimate):
        return 0.0
    error = abs(estimate - certified)
    if error == 0.0:
        return digits
    scale = abs(certified) if certified != 0.0 else 1.0
    return max(0.0, min(digits, -math.log10(error / scale)))


def _lines(name: str) -> list[str]:
    return (DATA / f"{name}.dat").read_text(encoding="latin-1").splitlines()


def _table_after_last_data_header(lines: list[str]) -> tuple[list[str], np.ndarray]:
    """The columns named on the last ``Data:`` line, and the rows under it."""
    start = max(i for i, line in enumerate(lines) if line.startswith("Data:"))
    columns = lines[start].split()[1:]
    rows = [[_float(value) for value in line.split()] for line in lines[start + 1:] if line.strip()]
    return columns, np.asarray(rows, dtype=float)


# ----------------------------------------------------------------------
# Nonlinear regression
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Nonlinear:
    name: str
    #: Lower, Average or Higher, as NIST grades them.
    difficulty: str
    x: np.ndarray
    y: np.ndarray
    start1: np.ndarray
    start2: np.ndarray
    params: np.ndarray
    std: np.ndarray
    rss: float
    residual_sd: float
    dof: int


def nonlinear(name: str) -> Nonlinear:
    lines = _lines(name)
    text = "\n".join(lines)
    rows = [
        re.match(rf"\s*b(\d+)\s*=\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*$", line)
        for line in lines
    ]
    found = [match for match in rows if match]
    start1 = np.array([_float(m.group(2)) for m in found])
    start2 = np.array([_float(m.group(3)) for m in found])
    params = np.array([_float(m.group(4)) for m in found])
    std = np.array([_float(m.group(5)) for m in found])

    def value(label: str) -> float:
        match = re.search(rf"{label}:\s*({_NUMBER})", text)
        assert match, f"{name}: no {label}"
        return _float(match.group(1))

    difficulty = re.search(r"(Lower|Average|Higher) Level of Difficulty", text)
    columns, table = _table_after_last_data_header(lines)
    y_column = columns.index("y")
    x_columns = [i for i, column in enumerate(columns) if column != "y"]
    x = table[:, x_columns[0]] if len(x_columns) == 1 else table[:, x_columns].T
    return Nonlinear(
        name=name,
        difficulty=difficulty.group(1) if difficulty else "",
        x=x,
        y=table[:, y_column],
        start1=start1,
        start2=start2,
        params=params,
        std=std,
        rss=value("Residual Sum of Squares"),
        residual_sd=value("Residual Standard Deviation"),
        dof=int(value("Degrees of Freedom")),
    )


# ----------------------------------------------------------------------
# Linear regression
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Linear:
    name: str
    x: np.ndarray
    y: np.ndarray
    #: B0, B1, ... - the intercept first when the model has one.
    params: np.ndarray
    std: np.ndarray
    residual_sd: float
    r_squared: float


def linear(name: str) -> Linear:
    lines = _lines(name)
    text = "\n".join(lines)
    found = [
        re.match(rf"\s*B(\d+)\s+({_NUMBER})\s+({_NUMBER})\s*$", line) for line in lines
    ]
    found = [match for match in found if match]
    residual_sd = re.search(rf"Standard Deviation\s+({_NUMBER})", text)
    r_squared = re.search(rf"R-Squared\s+({_NUMBER})", text)
    assert residual_sd and r_squared, name
    columns, table = _table_after_last_data_header(lines)
    y_column = columns.index("y")
    x_columns = [i for i, column in enumerate(columns) if column != "y"]
    return Linear(
        name=name,
        x=table[:, x_columns[0]] if len(x_columns) == 1 else table[:, x_columns],
        y=table[:, y_column],
        params=np.array([_float(m.group(2)) for m in found]),
        std=np.array([_float(m.group(3)) for m in found]),
        residual_sd=_float(residual_sd.group(1)),
        r_squared=_float(r_squared.group(1)),
    )


# ----------------------------------------------------------------------
# Univariate summary statistics
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Univariate:
    name: str
    values: np.ndarray
    mean: float
    std: float
    autocorrelation: float


def univariate(name: str) -> Univariate:
    lines = _lines(name)
    text = "\n".join(lines)
    span = re.search(r"Data\s*:\s*lines\s+(\d+)\s+to\s+(\d+)", text)
    assert span, name
    first, last = int(span.group(1)), int(span.group(2))
    values = [_float(token) for line in lines[first - 1:last] for token in line.split()]

    def value(label: str) -> float:
        match = re.search(rf"{re.escape(label)}:\s*({_NUMBER})", text)
        assert match, f"{name}: no {label}"
        return _float(match.group(1))

    return Univariate(name, np.asarray(values, dtype=float), value("ybar"), value("s"), value("r(1)"))


# ----------------------------------------------------------------------
# One-way analysis of variance
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Anova:
    name: str
    groups: dict[str, np.ndarray]
    between_df: int
    between_ss: float
    within_df: int
    within_ss: float
    f_statistic: float
    r_squared: float
    residual_sd: float


def anova(name: str) -> Anova:
    lines = _lines(name)
    text = "\n".join(lines)
    between = re.search(
        rf"Between \w+\s+(\d+)\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})", text
    )
    within = re.search(rf"Within \w+\s+(\d+)\s+({_NUMBER})\s+({_NUMBER})", text)
    r_squared = re.search(rf"Certified R-Squared\s+({_NUMBER})", text)
    residual_sd = re.search(rf"Standard Deviation\s+({_NUMBER})", text)
    assert between and within and r_squared and residual_sd, name
    _columns, table = _table_after_last_data_header(lines)
    groups: dict[str, list[float]] = {}
    for group, response in table:
        groups.setdefault(str(int(group)), []).append(float(response))
    return Anova(
        name=name,
        groups={key: np.asarray(values) for key, values in groups.items()},
        between_df=int(between.group(1)),
        between_ss=_float(between.group(2)),
        within_df=int(within.group(1)),
        within_ss=_float(within.group(2)),
        f_statistic=_float(between.group(4)),
        r_squared=_float(r_squared.group(1)),
        residual_sd=_float(residual_sd.group(1)),
    )
