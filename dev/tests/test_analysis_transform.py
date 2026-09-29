"""Transform (app.analysis.transform): answers known in advance.

Standardising must give mean 0 and standard deviation 1, the robust scaler
median 0 and interquartile range 1, a quantile map must keep the order and
land in its target range, a power transform must pull a skewed sample toward
symmetry. No dialog, no Qt.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from app.analysis import transform as tf

RNG = np.random.default_rng(4)
SKEWED = RNG.lognormal(mean=0.0, sigma=0.9, size=400)


def test_the_standard_scaler_gives_mean_zero_and_unit_deviation() -> None:
    out = tf.transform_values(tf.KIND_STANDARD, SKEWED).y
    assert np.mean(out) == pytest.approx(0.0, abs=1e-12)
    assert np.std(out) == pytest.approx(1.0)


def test_the_robust_scaler_gives_median_zero_and_unit_iqr() -> None:
    out = tf.transform_values(tf.KIND_ROBUST, SKEWED).y
    assert np.median(out) == pytest.approx(0.0, abs=1e-12)
    assert stats.iqr(out) == pytest.approx(1.0)


def test_the_robust_scaler_can_skip_centering_or_scaling() -> None:
    no_centre = tf.transform_values(tf.KIND_ROBUST, SKEWED, tf.TransformSettings(with_centering=False)).y
    np.testing.assert_allclose(no_centre, SKEWED / stats.iqr(SKEWED))
    no_scale = tf.transform_values(tf.KIND_ROBUST, SKEWED, tf.TransformSettings(with_scaling=False)).y
    np.testing.assert_allclose(no_scale, SKEWED - np.median(SKEWED))


def test_a_uniform_quantile_map_keeps_the_order_and_fills_zero_to_one() -> None:
    out = tf.transform_values(tf.KIND_QUANTILE, SKEWED).y
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)
    np.testing.assert_array_equal(np.argsort(out, kind="stable"), np.argsort(SKEWED, kind="stable"))


def test_a_normal_quantile_map_is_close_to_a_standard_normal() -> None:
    result = tf.transform_values(
        tf.KIND_QUANTILE, SKEWED, tf.TransformSettings(output_distribution="normal", n_quantiles=400)
    )
    inside = result.y[np.abs(result.y) < 5]  # the two ends are clipped to +-5.2 by design
    assert np.mean(inside) == pytest.approx(0.0, abs=0.1)
    assert np.std(inside) == pytest.approx(1.0, abs=0.15)


def test_the_number_of_quantiles_is_clipped_to_the_series_and_reported() -> None:
    result = tf.transform_values(tf.KIND_QUANTILE, np.arange(20.0), tf.TransformSettings(n_quantiles=1000))
    assert result.details == {"n_quantiles": 20}


@pytest.mark.parametrize("method", [tf.POWER_YEO_JOHNSON, tf.POWER_BOX_COX])
def test_a_power_transform_makes_a_skewed_sample_more_symmetric(method: str) -> None:
    out = tf.transform_values(tf.KIND_POWER, SKEWED, tf.TransformSettings(method=method))
    assert abs(stats.skew(out.y)) < 0.25 * abs(stats.skew(SKEWED))
    assert out.details == {"method": method}


def test_box_cox_refuses_a_value_that_is_not_positive_but_yeo_johnson_does_not() -> None:
    data = np.array([-1.0, 0.5, 2.0, 3.5, 8.0, 15.0])
    with pytest.raises(ValueError, match="strictly positive"):
        tf.transform_values(tf.KIND_POWER, data, tf.TransformSettings(method=tf.POWER_BOX_COX))
    assert tf.transform_values(tf.KIND_POWER, data).y.size == data.size


def test_the_result_has_the_length_of_the_input_and_an_unknown_kind_is_refused() -> None:
    assert tf.transform_values(tf.KIND_STANDARD, np.arange(7.0)).y.shape == (7,)
    with pytest.raises(ValueError, match="Unsupported transform"):
        tf.transform_values("nope", np.arange(7.0))
