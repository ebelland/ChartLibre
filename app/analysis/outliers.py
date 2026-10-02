"""Outliers: the arithmetic behind the Outliers series operation.

Four detectors test y alone - z-score, interquartile range, median absolute
deviation, residual from a rolling median - and four fit the joint (x, y)
point cloud (scikit-learn: Isolation Forest, Local Outlier Factor, One-Class
SVM, Elliptic Envelope), which catches a point at an ordinary y but far from
the curve everything else traces at that x. Plain arrays in, a mask out; no
Qt, no repository, nothing logged (todo R-01). What is done with the outliers
- hiding or colouring the rows - is the dialog's business.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis import NUMERICAL_FAILURES
from scipy.ndimage import median_filter
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

OUTLIER_ZSCORE = "Z-score threshold"
OUTLIER_IQR = "Interquartile range"
OUTLIER_MAD = "Median absolute deviation"
OUTLIER_ROLLING = "Rolling median residual"
OUTLIER_ISOLATION_FOREST = "Isolation Forest"
OUTLIER_LOCAL_OUTLIER_FACTOR = "Local Outlier Factor"
OUTLIER_ONE_CLASS_SVM = "One-Class SVM"
OUTLIER_ELLIPTIC_ENVELOPE = "Elliptic Envelope"

#: The detectors that fit the (x, y) cloud rather than y alone.
SHAPE_AWARE: frozenset[str] = frozenset({
    OUTLIER_ISOLATION_FOREST,
    OUTLIER_LOCAL_OUTLIER_FACTOR,
    OUTLIER_ONE_CLASS_SVM,
    OUTLIER_ELLIPTIC_ENVELOPE,
})


@dataclass(frozen=True, slots=True)
class OutlierSettings:
    """What the detectors read; each reads its own."""

    #: Z-score, MAD and rolling median: multiples of the spread estimate.
    threshold: float = 3.0
    #: IQR: multiples of the interquartile range beyond the quartiles.
    iqr_factor: float = 1.5
    #: Rolling median: window, in points (made odd and fitted to the series).
    window: int = 11
    #: Isolation Forest, LOF, Elliptic Envelope: expected fraction of outliers.
    contamination: float = 0.05
    #: LOF: neighbours each point's density is compared against.
    n_neighbors: int = 20
    #: One-Class SVM.
    nu: float = 0.05


def odd_window(value: int, n_values: int, minimum: int = 3) -> int:
    window = max(minimum, int(value))
    if window % 2 == 0:
        window += 1
    if window > n_values:
        window = n_values if n_values % 2 == 1 else n_values - 1
    return max(minimum, window)


def outlier_mask(
    method: str, x_data: np.ndarray, y_data: np.ndarray, settings: OutlierSettings | None = None
) -> np.ndarray:
    """True for each point *method* calls an outlier. *x_data* sorted (the rolling median walks it in order)."""
    settings = settings or OutlierSettings()
    model = method
    if model in SHAPE_AWARE:
        return _shape_aware_mask(x_data, y_data, model, settings)

    threshold = float(settings.threshold)
    if model == OUTLIER_ZSCORE:
        mean = float(np.mean(y_data))
        std = float(np.std(y_data, ddof=0))
        if std <= 0.0:
            return np.zeros_like(y_data, dtype=bool)
        return np.abs((y_data - mean) / std) > threshold
    if model == OUTLIER_IQR:
        q1 = float(np.percentile(y_data, 25.0))
        q3 = float(np.percentile(y_data, 75.0))
        iqr = q3 - q1
        if iqr <= 0.0:
            return np.zeros_like(y_data, dtype=bool)
        factor = float(settings.iqr_factor)
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return (y_data < lower) | (y_data > upper)
    if model == OUTLIER_MAD:
        median = float(np.median(y_data))
        mad = float(np.median(np.abs(y_data - median)))
        if mad <= 0.0:
            std = float(np.std(y_data, ddof=0))
            if std <= 0.0:
                return np.zeros_like(y_data, dtype=bool)
            return np.abs(y_data - median) > threshold * std
        scaled = 1.4826 * mad
        return np.abs(y_data - median) > threshold * scaled
    if model == OUTLIER_ROLLING:
        window = odd_window(int(settings.window), y_data.size)
        median_values = median_filter(y_data, size=window, mode="reflect")
        residuals = np.abs(y_data - median_values)
        scale = float(np.median(residuals))
        if scale <= 0.0:
            std = float(np.std(residuals, ddof=0))
            if std <= 0.0:
                return np.zeros_like(y_data, dtype=bool)
            scale = std
        return residuals > threshold * scale
    raise ValueError(f"Unsupported outlier detection model: {model}")


def robust_scaled(features: np.ndarray) -> np.ndarray:
    """Each column centred on its median and divided by its interquartile range.

    For the detectors that measure Euclidean distance between points (Local
    Outlier Factor, One-Class SVM): without it the column with the larger
    numbers decides alone - x in seconds since 1970 and y in volts, and y
    stops counting at all. The IQR rather than the standard deviation, so the
    outliers being looked for do not set the scale. A column with no spread
    falls back to its standard deviation, then to 1.
    """
    scaled = np.array(features, dtype=float, copy=True)
    for column in range(scaled.shape[1]):
        values = scaled[:, column]
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        spread = q3 - q1
        if not np.isfinite(spread) or spread <= 0:
            spread = float(np.std(values))
        if not np.isfinite(spread) or spread <= 0:
            spread = 1.0
        scaled[:, column] = (values - median) / spread
    return scaled


def _shape_aware_mask(
    x_data: np.ndarray, y_data: np.ndarray, model: str, settings: OutlierSettings
) -> np.ndarray:
    """Fit one of the four shape-aware estimators on the (x, y) cloud.

    Every one of these reports its verdict the same way - -1 is an
    outlier, +1 is not - straight from fit_predict/predict, so the four
    branches differ only in which estimator is built and how its own
    parameters are read.

    LOF and the SVM see the cloud through :func:`robust_scaled`; Isolation
    Forest splits one feature at a time and the Elliptic Envelope measures
    Mahalanobis distance, so neither depends on the units.
    """
    features = np.column_stack([x_data, y_data])
    n_samples = features.shape[0]

    if model == OUTLIER_ISOLATION_FOREST:
        contamination = float(settings.contamination)
        labels = IsolationForest(
            contamination=contamination, random_state=0
        ).fit_predict(features)
        return labels == -1

    if model == OUTLIER_LOCAL_OUTLIER_FACTOR:
        contamination = float(settings.contamination)
        # n_neighbors must be below the sample count, or scikit-learn
        # raises instead of just capping it - a short series would
        # otherwise crash rather than fall back to the largest
        # neighbourhood that still makes sense for it.
        n_neighbors = max(1, min(int(settings.n_neighbors), n_samples - 1))
        labels = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination
        ).fit_predict(robust_scaled(features))
        return labels == -1

    if model == OUTLIER_ONE_CLASS_SVM:
        nu = float(settings.nu)
        estimator = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        scaled = robust_scaled(features)
        labels = estimator.fit(scaled).predict(scaled)
        return labels == -1

    if model == OUTLIER_ELLIPTIC_ENVELOPE:
        contamination = float(settings.contamination)
        try:
            estimator = EllipticEnvelope(contamination=contamination, random_state=0)
            labels = estimator.fit(features).predict(features)
        except NUMERICAL_FAILURES as exc:
            # Typically a singular covariance matrix - too few points, or
            # points that are collinear / have (near-)zero variance in x
            # or y. Both are properties of the data, not a bug, so this
            # is worth its own clear message rather than a raw LinAlgError.
            raise ValueError(
                "Elliptic Envelope could not fit a covariance for this "
                f"series ({n_samples} point(s)): {exc}"
            ) from exc
        return labels == -1

    raise ValueError(f"Unsupported outlier detection model: {model}")
