"""The 27 NIST StRD nonlinear models, as ChartLibre's Fit engine takes them: ``f(x, p)``.

Written from the formula in each file's header (dev/tests/data/nist).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

Model = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _exp3(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b[0] * np.exp(-b[1] * x) + b[2] * np.exp(-b[3] * x) + b[4] * np.exp(-b[5] * x)


def _gauss(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        b[0] * np.exp(-b[1] * x)
        + b[2] * np.exp(-((x - b[3]) ** 2) / b[4] ** 2)
        + b[5] * np.exp(-((x - b[6]) ** 2) / b[7] ** 2)
    )


def _rational33(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (b[0] + b[1] * x + b[2] * x**2 + b[3] * x**3) / (1 + b[4] * x + b[5] * x**2 + b[6] * x**3)


def _nelson(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    # NIST states this one as log[y] = ...; the response fitted is log(y).
    x1, x2 = x
    return b[0] - b[1] * x1 * np.exp(-b[2] * x2)


MODELS: dict[str, Model] = {
    "Misra1a": lambda x, b: b[0] * (1 - np.exp(-b[1] * x)),
    "Chwirut2": lambda x, b: np.exp(-b[0] * x) / (b[1] + b[2] * x),
    "Chwirut1": lambda x, b: np.exp(-b[0] * x) / (b[1] + b[2] * x),
    "Lanczos3": _exp3,
    "Gauss1": _gauss,
    "Gauss2": _gauss,
    "DanWood": lambda x, b: b[0] * x ** b[1],
    "Misra1b": lambda x, b: b[0] * (1 - (1 + b[1] * x / 2) ** (-2)),
    "Kirby2": lambda x, b: (b[0] + b[1] * x + b[2] * x**2) / (1 + b[3] * x + b[4] * x**2),
    "Hahn1": _rational33,
    "Nelson": _nelson,
    "MGH17": lambda x, b: b[0] + b[1] * np.exp(-x * b[3]) + b[2] * np.exp(-x * b[4]),
    "Lanczos1": _exp3,
    "Lanczos2": _exp3,
    "Gauss3": _gauss,
    "Misra1c": lambda x, b: b[0] * (1 - (1 + 2 * b[1] * x) ** (-0.5)),
    "Misra1d": lambda x, b: b[0] * b[1] * x * ((1 + b[1] * x) ** (-1)),
    "Roszman1": lambda x, b: b[0] - b[1] * x - np.arctan(b[2] / (x - b[3])) / np.pi,
    "ENSO": lambda x, b: (
        b[0]
        + b[1] * np.cos(2 * np.pi * x / 12) + b[2] * np.sin(2 * np.pi * x / 12)
        + b[4] * np.cos(2 * np.pi * x / b[3]) + b[5] * np.sin(2 * np.pi * x / b[3])
        + b[7] * np.cos(2 * np.pi * x / b[6]) + b[8] * np.sin(2 * np.pi * x / b[6])
    ),
    "MGH09": lambda x, b: b[0] * (x**2 + x * b[1]) / (x**2 + x * b[2] + b[3]),
    "Thurber": _rational33,
    "BoxBOD": lambda x, b: b[0] * (1 - np.exp(-b[1] * x)),
    "Rat42": lambda x, b: b[0] / (1 + np.exp(b[1] - b[2] * x)),
    "MGH10": lambda x, b: b[0] * np.exp(b[1] / (x + b[2])),
    "Eckerle4": lambda x, b: (b[0] / b[1]) * np.exp(-0.5 * ((x - b[2]) / b[1]) ** 2),
    "Rat43": lambda x, b: b[0] / ((1 + np.exp(b[1] - b[2] * x)) ** (1 / b[3])),
    "Bennett5": lambda x, b: b[0] * (b[1] + x) ** (-1 / b[2]),
}

#: Fitted to log(y), as the certificate is.
LOG_RESPONSE = frozenset({"Nelson"})
