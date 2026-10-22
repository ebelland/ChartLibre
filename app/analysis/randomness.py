"""Randomness and serial-independence diagnostics for ordered numeric data.

The numerical tests return :class:`RandomnessResult` objects and never log or
raise for ordinary data limitations.  Callers decide how notes are presented.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class RandomnessResult:
    """One diagnostic result; ``note`` explains an unavailable/qualified test."""

    test: str
    n: int
    statistic: float = np.nan
    pvalue: float = np.nan
    note: str = ""

    @property
    def available(self) -> bool:
        return bool(np.isfinite(self.statistic))


def finite(values: Any) -> np.ndarray:
    """Return a flat float array with NaN and infinities removed."""
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _unavailable(test: str, n: int, note: str) -> RandomnessResult:
    return RandomnessResult(test=test, n=int(n), note=note)


def runs_test(values: Any, *, centre: float | None = None) -> RandomnessResult:
    """Wald-Wolfowitz runs test above/below ``centre`` (median by default).

    Values exactly on the centre are omitted.  The returned statistic is the
    normal-approximation z score with a 0.5 continuity correction.
    """
    x = finite(values)
    n = int(x.size)
    if n < 10:
        return _unavailable("Wald-Wolfowitz runs", n, "Needs at least 10 finite observations.")
    split = float(np.median(x) if centre is None else centre)
    signs = x[x != split] > split
    effective = int(signs.size)
    dropped = n - effective
    if effective < 10:
        return _unavailable(
            "Wald-Wolfowitz runs", n,
            f"Only {effective} observations remain after dropping {dropped} tie(s) at the centre.",
        )
    n_above = int(np.sum(signs))
    n_below = effective - n_above
    if n_above == 0 or n_below == 0:
        return _unavailable("Wald-Wolfowitz runs", n, "Needs observations on both sides of the centre.")
    runs = int(1 + np.count_nonzero(signs[1:] != signs[:-1]))
    expected = 1.0 + 2.0 * n_above * n_below / effective
    variance = (
        2.0 * n_above * n_below * (2.0 * n_above * n_below - effective)
        / (effective * effective * (effective - 1.0))
    )
    if not np.isfinite(variance) or variance <= 0.0:
        return _unavailable("Wald-Wolfowitz runs", n, "The run-count variance is zero.")
    delta = runs - expected
    corrected = np.sign(delta) * max(abs(delta) - 0.5, 0.0)
    z_value = float(corrected / np.sqrt(variance))
    pvalue = float(2.0 * stats.norm.sf(abs(z_value)))
    note = f"runs = {runs}; expected = {expected:.4g}; centre = {split:.6g}"
    if dropped:
        note += f"; {dropped} tie(s) at the centre omitted"
    return RandomnessResult("Wald-Wolfowitz runs", n, z_value, pvalue, note)


def turning_point_test(values: Any) -> RandomnessResult:
    """Turning-point test for serial randomness using its asymptotic normal law.

    Consecutive ties make local extrema ambiguous, therefore the test returns
    an explanatory note instead of silently deleting or jittering them.
    """
    x = finite(values)
    n = int(x.size)
    if n < 6:
        return _unavailable("Turning-point randomness", n, "Needs at least 6 finite observations.")
    if np.any(np.diff(x) == 0.0):
        return _unavailable("Turning-point randomness", n, "Consecutive ties make turning points ambiguous.")
    turns = int(np.count_nonzero(((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) |
                                 ((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))))
    expected = 2.0 * (n - 2.0) / 3.0
    variance = (16.0 * n - 29.0) / 90.0
    z_value = float((turns - expected) / np.sqrt(variance))
    pvalue = float(2.0 * stats.norm.sf(abs(z_value)))
    return RandomnessResult(
        "Turning-point randomness", n, z_value, pvalue,
        f"turning points = {turns}; expected = {expected:.4g}",
    )


def durbin_watson(values: Any) -> RandomnessResult:
    """Durbin-Watson statistic for an ordered residual sequence.

    It is reported without a generic p-value because its null distribution
    depends on the fitted regression design.  Values near 2 indicate little
    lag-1 correlation; this is a diagnostic, not a stand-alone hypothesis test.
    """
    x = finite(values)
    n = int(x.size)
    if n < 3:
        return _unavailable("Durbin-Watson", n, "Needs at least 3 finite residuals.")
    denominator = float(np.dot(x, x))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        return _unavailable("Durbin-Watson", n, "Residual sum of squares is zero.")
    statistic = float(np.dot(np.diff(x), np.diff(x)) / denominator)
    return RandomnessResult("Durbin-Watson", n, statistic, np.nan, "No generic p-value; 2 indicates no lag-1 correlation.")


def ljung_box(values: Any, *, lags: Iterable[int] = (1, 5, 10, 20), model_df: int = 0) -> list[RandomnessResult]:
    """Ljung-Box portmanteau tests at selected lags.

    ``model_df`` is the number of fitted AR/MA parameters.  A requested lag
    must be greater than ``model_df`` and no larger than ``n // 2``.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    x = finite(values)
    n = int(x.size)
    requested = sorted({int(lag) for lag in lags if int(lag) > 0})
    if not requested:
        return [_unavailable("Ljung-Box", n, "No positive lags were requested.")]
    if n < 8:
        return [_unavailable("Ljung-Box", n, "Needs at least 8 finite observations.")]
    if float(np.ptp(x)) == 0.0:
        return [_unavailable("Ljung-Box", n, "The series is constant.")]
    maximum = n // 2
    usable = [lag for lag in requested if model_df < lag <= maximum]
    skipped = [lag for lag in requested if lag not in usable]
    if not usable:
        return [_unavailable("Ljung-Box", n, f"No requested lag is in {model_df + 1}..{maximum}.")]
    try:
        table = acorr_ljungbox(x, lags=usable, model_df=max(0, int(model_df)), return_df=True)
    except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
        return [_unavailable("Ljung-Box", n, str(exc))]
    rows: list[RandomnessResult] = []
    for lag, row in table.iterrows():
        note = f"lag = {int(lag)}; null: no autocorrelation through this lag"
        if skipped and int(lag) == usable[-1]:
            note += f"; skipped unavailable lag(s): {', '.join(map(str, skipped))}"
        rows.append(RandomnessResult(
            f"Ljung-Box (lag {int(lag)})", n,
            float(row["lb_stat"]), float(row["lb_pvalue"]), note,
        ))
    return rows


def all_diagnostics(values: Any, *, lags: Iterable[int] = (1, 5, 10, 20),
                    model_df: int = 0, centre: float | None = None) -> list[RandomnessResult]:
    """Runs, turning points, Durbin-Watson and Ljung-Box in report order."""
    return [
        runs_test(values, centre=centre),
        turning_point_test(values),
        durbin_watson(values),
        *ljung_box(values, lags=lags, model_df=model_df),
    ]
