"""The compiled kernels give what the plain-Python code gave (Numba or not)."""
from __future__ import annotations

import numpy as np

from app.analysis import control_charts, smoothing


def test_the_kalman_kernel_is_a_local_level_smoother() -> None:
    y = np.r_[np.zeros(50), np.ones(50)] + np.random.default_rng(0).normal(0, 0.05, 100)
    smoothed = smoothing._kalman_smoother_1d(y, 1e-3, 0.05**2, 1.0)
    assert smoothed.shape == y.shape
    assert abs(smoothed[:40].mean()) < 0.05 and abs(smoothed[60:].mean() - 1) < 0.05
    assert np.std(np.diff(smoothed)) < np.std(np.diff(y))


def test_the_run_rules_flag_where_each_pattern_ends() -> None:
    values = np.r_[np.full(9, 0.5), [0.0], np.arange(6.0) * 0.1 - 0.9, np.zeros(4)]
    n = values.size
    sigma = np.ones(n)
    found = control_charts.find_violations(values, np.arange(n, dtype=float), 0.0, sigma, 3 * sigma, -3 * sigma, True)
    by_rule = {rule: [v.index for v in found if rule in v.rules] for rule in range(2, 9)}
    assert by_rule[2] == [8]  # the ninth point above the centre
    assert by_rule[3] == [15, 16]  # the sixth rising, and the seventh (still rising, to 0)
