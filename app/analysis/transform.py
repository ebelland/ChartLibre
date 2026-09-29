"""Transform: the arithmetic behind the Transform series operation.

Reshapes the y values of a series point for point - a power transform, a
quantile map, standardising or robust scaling (scikit-learn) - over plain
arrays. No Qt, nothing logged (todo R-01); a request the data cannot meet
(Box-Cox on a value that is not positive) is a ``ValueError``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

KIND_POWER = "power"
KIND_QUANTILE = "quantile"
KIND_STANDARD = "standard"
KIND_ROBUST = "robust"

POWER_YEO_JOHNSON = "yeo-johnson"
POWER_BOX_COX = "box-cox"


@dataclass(frozen=True, slots=True)
class TransformSettings:
    """What a transform reads besides the values; each kind reads its own."""

    #: Power transform: POWER_YEO_JOHNSON or POWER_BOX_COX.
    method: str = POWER_YEO_JOHNSON
    #: Quantile transform: "uniform" or "normal", and how finely to estimate.
    output_distribution: str = "uniform"
    n_quantiles: int = 1000
    #: Robust scaler.
    with_centering: bool = True
    with_scaling: bool = True


@dataclass(frozen=True, slots=True)
class Transformed:
    """The reshaped values, and what the transform settled on."""

    y: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)


def transform_values(kind: str, y: np.ndarray, settings: TransformSettings | None = None) -> Transformed:
    """Reshape *y* with *kind* (KIND_POWER, KIND_QUANTILE, KIND_STANDARD, KIND_ROBUST)."""
    settings = settings or TransformSettings()
    values = np.asarray(y, dtype=float)
    column = values.reshape(-1, 1)
    details: dict[str, Any] = {}

    if kind == KIND_POWER:
        if settings.method == POWER_BOX_COX and not np.all(values > 0):
            raise ValueError(
                "Box-Cox requires every value to be strictly positive; "
                "use Yeo-Johnson instead, or subtract a baseline first."
            )
        transformer = PowerTransformer(
            method="box-cox" if settings.method == POWER_BOX_COX else "yeo-johnson"
        )
        details["method"] = settings.method

    elif kind == KIND_QUANTILE:
        n_quantiles = max(2, min(int(settings.n_quantiles), values.size))
        transformer = QuantileTransformer(
            output_distribution="normal" if settings.output_distribution == "normal" else "uniform",
            n_quantiles=n_quantiles,
            random_state=0,
        )
        details["n_quantiles"] = n_quantiles

    elif kind == KIND_ROBUST:
        transformer = RobustScaler(
            with_centering=bool(settings.with_centering),
            with_scaling=bool(settings.with_scaling),
        )

    elif kind == KIND_STANDARD:
        transformer = StandardScaler()

    else:
        raise ValueError(f"Unsupported transform: {kind}")

    return Transformed(np.asarray(transformer.fit_transform(column), dtype=float).ravel(), details)
