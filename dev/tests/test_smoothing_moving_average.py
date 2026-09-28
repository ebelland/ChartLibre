"""The moving average keeps its level at the ends of the series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.series_operations.smoothing_dialog import _moving_average


@pytest.mark.parametrize("centered", [True, False])
def test_a_constant_series_stays_constant_to_the_last_point(centered: bool) -> None:
    np.testing.assert_allclose(_moving_average(np.full(30, 120.0), 7, centered), 120.0)


@pytest.mark.parametrize("centered", [True, False])
def test_the_window_shrinks_at_the_ends_like_pandas(centered: bool) -> None:
    y = np.random.default_rng(3).normal(100.0, 5.0, 60)
    expected = pd.Series(y).rolling(9, center=centered, min_periods=1).mean().to_numpy()
    np.testing.assert_allclose(_moving_average(y, 9, centered), expected)
