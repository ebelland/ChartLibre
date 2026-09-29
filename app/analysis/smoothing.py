"""Smoothing: the arithmetic behind the Smoothing series operation.

Fourteen methods for a 1D series (a moving average up to total-variation
denoising), six for a 2D surface and five for a 3D volume. Every function
takes plain arrays and a mapping of parameters and returns arrays - no Qt,
no repository, no dialog - so the same code runs behind the window, in a
test or on a worker thread (todo R-01).

A problem with the data or the method is a ``ValueError``; a method whose
optional package is not installed is an ``ImportError``. Neither is logged
here: the caller decides how to tell the user, and a dialog that shows the
message once is better than an engine that shows it and carries on.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple, cast

import numpy as np
from scipy import fft
from scipy.interpolate import (
    RBFInterpolator,
    RectBivariateSpline,
    SmoothBivariateSpline,
    UnivariateSpline,
)
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.signal import butter, medfilt, savgol_filter, sosfiltfilt, wiener

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    from statsmodels.tsa.filters.hp_filter import hpfilter
except ImportError:  # pragma: no cover
    sm_lowess = None
    hpfilter = None

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError:  # pragma: no cover
    denoise_tv_chambolle = None

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


def _require_dependency(dependency: Any, name: str) -> Any:
    if dependency is None:
        raise ImportError(
            f"Smoothing method requires optional dependency '{name}', "
            "which is not installed in the current Python environment."
        )
    return dependency


# ---------------------------------------------------------------------------
# Methods, by the name the dialog lists them under
# ---------------------------------------------------------------------------

SMOOTH_MOVING_AVERAGE = "Moving Average"
SMOOTH_SAVGOL = "Savitzky-Golay Filter (SciPy)"
SMOOTH_GAUSSIAN = "Gaussian Filter (SciPy)"
SMOOTH_MEDIAN = "Median Filter (SciPy)"
SMOOTH_WIENER = "Wiener Filter (SciPy)"
SMOOTH_SPLINE = "Smoothing Spline (SciPy)"
SMOOTH_LOWESS = "LOWESS / LOESS"
SMOOTH_KALMAN = "Kalman Smoother"
SMOOTH_FFT = "FFT Low-Pass Filter"
SMOOTH_BUTTERWORTH = "Butterworth Filter (SciPy)"
SMOOTH_WAVELET = "Wavelet Denoising"
SMOOTH_WHITTAKER = "Whittaker-Eilers Smoother"
SMOOTH_HP = "Hodrick-Prescott Filter"
SMOOTH_TV = "Total Variation Denoising"

SMOOTH2D_GAUSSIAN = "2D Gaussian Surface Filter"
SMOOTH2D_MEDIAN = "2D Median Surface Filter"
SMOOTH2D_SPLINE = "2D SmoothBivariateSpline"
SMOOTH2D_RECT_SPLINE = "2D RectBivariateSpline"
SMOOTH2D_RBF = "2D RBF Surface Smoothing"
SMOOTH2D_TV = "2D Total Variation Surface Denoising"

SMOOTH3D_GAUSSIAN = "3D Gaussian Volume Filter"
SMOOTH3D_MEDIAN = "3D Median Volume Filter"
SMOOTH3D_FFT = "3D FFT Low-Pass Volume Filter"
SMOOTH3D_RBF = "3D RBF Volume Smoothing"
SMOOTH3D_TV = "3D Total Variation Volume Denoising"


#: The methods that need an optional package, with the package (None when
#: it is not installed).
_NEEDS: dict[str, Any] = {
    SMOOTH_LOWESS: sm_lowess,
    SMOOTH_HP: hpfilter,
    SMOOTH_TV: denoise_tv_chambolle,
    SMOOTH2D_TV: denoise_tv_chambolle,
    SMOOTH3D_TV: denoise_tv_chambolle,
    SMOOTH_WAVELET: pywt,
}


def is_available(method: str) -> bool:
    """True unless *method* needs an optional package that is not installed."""
    return _NEEDS.get(method, True) is not None


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _odd_window(value: int, n_values: int, minimum: int = 3) -> int:
    """Return an odd integer window length within the available data size."""
    window = max(minimum, int(value))
    if window % 2 == 0:
        window += 1
    if window > n_values:
        window = n_values if n_values % 2 == 1 else n_values - 1
    return max(minimum, window)


def _moving_average(y_values: np.ndarray, window: int, centered: bool) -> np.ndarray:
    """Moving average whose window shrinks at the ends instead of reading zeros.

    np.convolve pads with zeros, so a plain convolution averaged the first and
    last half-window of points with nothing and pulled them towards zero - a
    price series smoothed that way plunged at both ends of the chart. Each
    output is instead the mean of the points its window actually covers,
    which is pandas' rolling(..., min_periods=1).
    """
    safe_window = max(1, int(window))
    kernel = np.ones(safe_window, dtype=float)
    values = np.asarray(y_values, dtype=float)
    mode = "same" if centered else "full"
    sums = np.convolve(values, kernel, mode=mode)[: values.size]
    counts = np.convolve(np.ones_like(values), kernel, mode=mode)[: values.size]
    return np.asarray(sums / counts, dtype=float)


def _fft_low_pass_1d(y_values: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Low-pass smooth a 1D signal in the frequency domain."""
    ratio = float(np.clip(cutoff_ratio, 0.001, 0.999))
    mean = float(np.nanmean(y_values))
    spectrum = np.asarray(cast(Any, fft.rfft)(y_values - mean), dtype=complex)
    frequencies = np.asarray(cast(Any, fft.rfftfreq)(y_values.size), dtype=float)
    spectrum[frequencies > ratio * float(np.max(frequencies))] = 0.0
    filtered = cast(Any, fft.irfft)(spectrum, n=y_values.size)
    return np.asarray(filtered, dtype=float) + mean


def _fft_low_pass_nd(values: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Low-pass smooth an N-dimensional grid in the frequency domain."""
    data = np.asarray(values, dtype=float)
    ratio = float(np.clip(cutoff_ratio, 0.001, 0.999))
    mean = float(np.nanmean(data))
    spectrum = np.asarray(cast(Any, fft.fftn)(data - mean), dtype=complex)
    frequency_grids = cast(
        list[np.ndarray],
        np.meshgrid(
            *[np.asarray(cast(Any, fft.fftfreq)(length), dtype=float) for length in data.shape],
            indexing="ij",
        ),
    )
    radius = np.sqrt(sum(grid * grid for grid in frequency_grids))
    spectrum[radius > ratio * float(np.max(radius))] = 0.0
    inverse = cast(Any, fft.ifftn)(spectrum)
    return np.asarray(np.real(np.asarray(inverse, dtype=complex)), dtype=float) + mean


def _butterworth(
    y_values: np.ndarray,
    cutoff: float,
    fs: float,
    order: int,
    btype: str,
    high_cutoff: float,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter."""
    wn: float | list[float]
    if btype in {"bandpass", "bandstop"}:
        wn = [float(cutoff), float(high_cutoff)]
    else:
        wn = float(cutoff)

    sos = butter(
        max(1, int(order)),
        wn,
        btype=btype,
        fs=max(float(fs), 1e-12),
        output="sos",
    )
    return cast(Any, sosfiltfilt)(sos, y_values)


def _wavelet_denoise(
    y_values: np.ndarray,
    wavelet: str,
    level: int,
    threshold_factor: float,
    mode: str,
) -> np.ndarray:
    """Denoise using discrete wavelet thresholding."""
    pywt_module = _require_dependency(pywt, "PyWavelets")
    coeffs = pywt_module.wavedec(
        y_values,
        wavelet=wavelet,
        mode="symmetric",
        level=level or None,
    )
    detail_coeffs = coeffs[1:]
    if not detail_coeffs:
        return y_values.copy()

    sigma = 0.0
    if detail_coeffs[-1].size:
        sigma = float(np.median(np.abs(detail_coeffs[-1])) / 0.6745)
    threshold = threshold_factor * sigma * math.sqrt(2.0 * math.log(max(y_values.size, 2)))
    new_coeffs = [coeffs[0]] + [
        pywt_module.threshold(coeff, threshold, mode=mode) for coeff in detail_coeffs
    ]
    output = pywt_module.waverec(new_coeffs, wavelet=wavelet, mode="symmetric")
    return np.asarray(output[: y_values.size], dtype=float)


def _whittaker_eilers(
    y_values: np.ndarray,
    lam: float,
    diff_order: int,
) -> np.ndarray:
    """Dense Whittaker-Eilers smoother for chart-sized series."""
    n_values = y_values.size
    identity = np.eye(n_values)
    difference = np.diff(identity, n=max(1, int(diff_order)), axis=0)
    system = identity + max(float(lam), 0.0) * difference.T @ difference
    return np.linalg.solve(system, y_values)


def _kalman_smoother_1d(
    y_values: np.ndarray,
    process_variance: float,
    measurement_variance: float,
    initial_covariance: float,
) -> np.ndarray:
    """Smooth a 1D signal with a local-level Kalman filter plus RTS pass.

    The state is a single latent level.  The forward pass removes measurement
    noise; the backward Rauch-Tung-Striebel pass reduces lag and produces a
    true smoother rather than only a filter.
    """
    values = np.asarray(y_values, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()

    q_value = max(float(process_variance), 1e-12)
    r_value = max(float(measurement_variance), 1e-12)
    p0_value = max(float(initial_covariance), 1e-12)

    filtered = np.empty_like(values, dtype=float)
    predicted = np.empty_like(values, dtype=float)
    filter_cov = np.empty_like(values, dtype=float)
    predict_cov = np.empty_like(values, dtype=float)

    state = float(values[0])
    covariance = p0_value

    for index, observation in enumerate(values):
        predicted[index] = state
        predict_cov[index] = covariance + q_value

        gain = predict_cov[index] / (predict_cov[index] + r_value)
        state = predicted[index] + gain * (float(observation) - predicted[index])
        covariance = (1.0 - gain) * predict_cov[index]

        filtered[index] = state
        filter_cov[index] = covariance

    smoothed = filtered.copy()
    for index in range(values.size - 2, -1, -1):
        gain = filter_cov[index] / max(predict_cov[index + 1], 1e-12)
        smoothed[index] = filtered[index] + gain * (
            smoothed[index + 1] - predicted[index + 1]
        )

    return smoothed


# ---------------------------------------------------------------------------
# Public smoothing functions
# ---------------------------------------------------------------------------

def smooth_1d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    method: str,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Smooth one XY series and return smoothed Y values."""
    x_clean, y_clean = clean_xy(x_values, y_values)
    n_values = y_clean.size

    if method == SMOOTH_MOVING_AVERAGE:
        return _moving_average(
            y_clean,
            int(params.get("window", 7)),
            bool(params.get("centered", True)),
        )

    if method == SMOOTH_SAVGOL:
        window = _odd_window(int(params.get("window", 7)), n_values)
        polyorder = min(int(params.get("polyorder", 2)), window - 1)
        return np.asarray(
            savgol_filter(
                y_clean,
                window_length=window,
                polyorder=polyorder,
                deriv=int(params.get("deriv", 0)),
                delta=float(params.get("delta", 1.0)),
                mode=str(params.get("savgol_mode", params.get("mode", "interp"))),
            ),
            dtype=float,
        )

    if method == SMOOTH_GAUSSIAN:
        return gaussian_filter1d(
            y_clean,
            sigma=float(params.get("sigma", 2.0)),
            mode=str(params.get("mode", "nearest")),
            truncate=float(params.get("truncate", 4.0)),
        )

    if method == SMOOTH_MEDIAN:
        kernel = _odd_window(int(params.get("kernel", 5)), n_values)
        return medfilt(y_clean, kernel_size=kernel)

    if method == SMOOTH_WIENER:
        noise = params.get("noise")
        return wiener(
            y_clean,
            mysize=_odd_window(int(params.get("window", 7)), n_values),
            noise=None if noise in (None, 0.0, "") else float(noise),
        )

    if method == SMOOTH_SPLINE:
        spline_order = int(np.clip(int(params.get("spline_k", 3)), 1, min(5, n_values - 1)))
        spline = UnivariateSpline(
            x_clean,
            y_clean,
            s=float(params.get("spline_s", max(n_values, 1))),
            k=spline_order,
        )
        return np.asarray(spline(x_clean), dtype=float)

    if method == SMOOTH_LOWESS:
        sm_lowess_fn = _require_dependency(sm_lowess, "statsmodels")
        return sm_lowess_fn(
            y_clean,
            x_clean,
            frac=float(params.get("lowess_frac", 0.25)),
            it=int(params.get("lowess_it", 3)),
            return_sorted=False,
        )

    if method == SMOOTH_KALMAN:
        return _kalman_smoother_1d(
            y_clean,
            process_variance=float(params.get("kalman_process_variance", 1e-4)),
            measurement_variance=float(params.get("kalman_measurement_variance", 1e-2)),
            initial_covariance=float(params.get("kalman_initial_covariance", 1.0)),
        )

    if method == SMOOTH_FFT:
        return _fft_low_pass_1d(y_clean, float(params.get("fft_cutoff_ratio", 0.20)))

    if method == SMOOTH_BUTTERWORTH:
        return _butterworth(
            y_clean,
            cutoff=float(params.get("butter_cutoff", 0.20)),
            fs=float(params.get("butter_fs", 1.0)),
            order=int(params.get("butter_order", 4)),
            btype=str(params.get("butter_type", "lowpass")),
            high_cutoff=float(params.get("butter_high_cutoff", 0.40)),
        )

    if method == SMOOTH_WAVELET:
        _require_dependency(pywt, "PyWavelets")
        return _wavelet_denoise(
            y_clean,
            wavelet=str(params.get("wavelet", "db4")),
            level=int(params.get("wavelet_level", 0)),
            threshold_factor=float(params.get("wavelet_threshold_factor", 1.0)),
            mode=str(params.get("wavelet_threshold_mode", "soft")),
        )

    if method == SMOOTH_WHITTAKER:
        return _whittaker_eilers(
            y_clean,
            lam=float(params.get("whittaker_lambda", 1000.0)),
            diff_order=int(params.get("whittaker_order", 2)),
        )

    if method == SMOOTH_HP:
        hpfilter_fn = _require_dependency(hpfilter, "statsmodels")
        _cycle, trend = cast(Any, hpfilter_fn)(y_clean, lamb=float(params.get("hp_lambda", 1600.0)))
        return np.asarray(trend, dtype=float)

    if method == SMOOTH_TV:
        denoise_fn = _require_dependency(denoise_tv_chambolle, "scikit-image")
        return np.asarray(
            denoise_fn(
                y_clean,
                weight=float(params.get("tv_weight", 0.15)),
            ),
            dtype=float,
        )

    raise ValueError(f"Unsupported 1D smoothing method: {method}")


def clean_xy(x_values: Any, y_values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite 1D XY arrays sorted by X."""
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)

    if x.size != y.size:
        raise ValueError("X and Y must have the same length.")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        raise ValueError("At least 3 finite points are required.")

    order = np.argsort(x)
    return x[order], y[order]

def clean_xyzw_volume(
    x_values: Any,
    y_values: Any,
    z_values: Any,
    raw_values: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 3D volume/grid or scattered XYZW arrays."""
    if z_values is None or raw_values is None:
        raise ValueError("3D smoothing requires X, Y, Z and values.")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    z = np.asarray(z_values, dtype=float)
    values = np.asarray(raw_values, dtype=float)

    if values.ndim == 3:
        return x.reshape(-1), y.reshape(-1), z.reshape(-1), values

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    values = values.reshape(-1)

    if not (x.size == y.size == z.size == values.size):
        raise ValueError("Scattered X, Y, Z and value arrays must match.")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(values)
    return x[mask], y[mask], z[mask], values[mask]

def clean_xyz_surface(
    x_values: Any,
    y_values: Any,
    z_values: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2D surface/grid or scattered XYZ arrays."""
    if z_values is None:
        raise ValueError("2D smoothing requires Z values.")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    z = np.asarray(z_values, dtype=float)

    if z.ndim == 2:
        return x.reshape(-1), y.reshape(-1), z

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    if not (x.size == y.size == z.size):
        raise ValueError("Scattered X, Y and Z arrays must have the same length.")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[mask], y[mask], z[mask]


def smooth_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    method: str,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth a 2D surface or scattered XYZ field."""
    x_clean, y_clean, z_clean = clean_xyz_surface(x_values, y_values, z_values)

    if method == SMOOTH2D_GAUSSIAN:
        if z_clean.ndim != 2:
            raise ValueError("2D Gaussian expects gridded Z data.")
        return x_clean, y_clean, gaussian_filter(
            z_clean,
            sigma=(
                float(params.get("sigma_y", 1.5)),
                float(params.get("sigma_x", 1.5)),
            ),
            mode=str(params.get("mode", "nearest")),
            truncate=float(params.get("truncate", 4.0)),
        )

    if method == SMOOTH2D_MEDIAN:
        if z_clean.ndim != 2:
            raise ValueError("2D median expects gridded Z data.")
        kernel_y = _odd_window(int(params.get("kernel_y", 3)), z_clean.shape[0])
        kernel_x = _odd_window(int(params.get("kernel_x", 3)), z_clean.shape[1])
        return x_clean, y_clean, median_filter(
            z_clean,
            size=(kernel_y, kernel_x),
            mode=str(params.get("mode", "nearest")),
        )

    if method == SMOOTH2D_SPLINE:
        if z_clean.ndim == 2:
            xx_mesh, yy_mesh = np.meshgrid(x_clean, y_clean)
            xs = xx_mesh.reshape(-1)
            ys = yy_mesh.reshape(-1)
            zs = z_clean.reshape(-1)
        else:
            xs = x_clean.reshape(-1)
            ys = y_clean.reshape(-1)
            zs = z_clean.reshape(-1)
        grid_x = np.unique(xs)
        grid_y = np.unique(ys)
        spline = SmoothBivariateSpline(
            xs,
            ys,
            zs,
            s=float(params.get("spline_s", len(zs))),
            kx=int(params.get("kx", 3)),
            ky=int(params.get("ky", 3)),
        )
        return grid_x, grid_y, spline(grid_x, grid_y).T

    if method == SMOOTH2D_RECT_SPLINE:
        if z_clean.ndim != 2:
            raise ValueError("RectBivariateSpline expects gridded data.")
        z_input = z_clean.T if z_clean.shape == (y_clean.size, x_clean.size) else z_clean
        spline = cast(Any, RectBivariateSpline)(
            x_clean,
            y_clean,
            z_input,
            kx=int(params.get("kx", 3)),
            ky=int(params.get("ky", 3)),
            s=float(params.get("spline_s", 0.0)),
        )
        return x_clean, y_clean, spline(x_clean, y_clean).T

    if method == SMOOTH2D_RBF:
        if z_clean.ndim == 2:
            xx_mesh, yy_mesh = np.meshgrid(x_clean, y_clean)
            points = np.column_stack([xx_mesh.reshape(-1), yy_mesh.reshape(-1)])
            values = z_clean.reshape(-1)
            grid_x = x_clean
            grid_y = y_clean
        else:
            points = np.column_stack([x_clean.reshape(-1), y_clean.reshape(-1)])
            values = z_clean.reshape(-1)
            grid_x = np.unique(x_clean)
            grid_y = np.unique(y_clean)
        epsilon = params.get("rbf_epsilon") or None
        neighbors = int(params.get("rbf_neighbors", 0)) or None
        rbf = RBFInterpolator(
            points,
            values,
            kernel=str(params.get("rbf_kernel", "thin_plate_spline")),
            smoothing=float(params.get("rbf_smoothing", 0.1)),
            epsilon=epsilon,
            neighbors=neighbors,
        )
        xx_eval, yy_eval = np.meshgrid(grid_x, grid_y)
        eval_points = np.column_stack([xx_eval.reshape(-1), yy_eval.reshape(-1)])
        return grid_x, grid_y, rbf(eval_points).reshape(xx_eval.shape)

    if method == SMOOTH2D_TV:
        if z_clean.ndim != 2:
            raise ValueError("2D total variation expects gridded data.")
        denoise_fn = _require_dependency(denoise_tv_chambolle, "scikit-image")
        return x_clean, y_clean, np.asarray(
            denoise_fn(
                z_clean,
                weight=float(params.get("tv_weight", 0.15)),
            ),
            dtype=float,
        )

    raise ValueError(f"Unsupported 2D smoothing method: {method}")


def smooth_3d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    raw_values: np.ndarray,
    method: str,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth a 3D volume or scattered XYZW field."""
    x_clean, y_clean, z_clean, values = clean_xyzw_volume(
        x_values,
        y_values,
        z_values,
        raw_values,
    )

    if method == SMOOTH3D_GAUSSIAN:
        if values.ndim != 3:
            raise ValueError("3D Gaussian expects gridded volume data.")
        return x_clean, y_clean, z_clean, gaussian_filter(
            values,
            sigma=(
                float(params.get("sigma_z", 1.5)),
                float(params.get("sigma_y", 1.5)),
                float(params.get("sigma_x", 1.5)),
            ),
            mode=str(params.get("mode", "nearest")),
            truncate=float(params.get("truncate", 4.0)),
        )

    if method == SMOOTH3D_MEDIAN:
        if values.ndim != 3:
            raise ValueError("3D median expects gridded volume data.")
        kernel_z = _odd_window(int(params.get("kernel_z", 3)), values.shape[0])
        kernel_y = _odd_window(int(params.get("kernel_y", 3)), values.shape[1])
        kernel_x = _odd_window(int(params.get("kernel_x", 3)), values.shape[2])
        return x_clean, y_clean, z_clean, median_filter(
            values,
            size=(kernel_z, kernel_y, kernel_x),
            mode=str(params.get("mode", "nearest")),
        )

    if method == SMOOTH3D_FFT:
        if values.ndim != 3:
            raise ValueError("3D FFT expects gridded volume data.")
        return x_clean, y_clean, z_clean, _fft_low_pass_nd(
            values,
            float(params.get("fft_cutoff_ratio", 0.20)),
        )

    if method == SMOOTH3D_RBF:
        if values.ndim == 3:
            zz_mesh, yy_mesh, xx_mesh = np.meshgrid(
                z_clean,
                y_clean,
                x_clean,
                indexing="ij",
            )
            points = np.column_stack(
                [xx_mesh.reshape(-1), yy_mesh.reshape(-1), zz_mesh.reshape(-1)]
            )
            scalar_values = values.reshape(-1)
            grid_x = x_clean
            grid_y = y_clean
            grid_z = z_clean
        else:
            points = np.column_stack(
                [x_clean.reshape(-1), y_clean.reshape(-1), z_clean.reshape(-1)]
            )
            scalar_values = values.reshape(-1)
            grid_x = np.unique(x_clean)
            grid_y = np.unique(y_clean)
            grid_z = np.unique(z_clean)
        epsilon = params.get("rbf_epsilon") or None
        neighbors = int(params.get("rbf_neighbors", 0)) or None
        rbf = RBFInterpolator(
            points,
            scalar_values,
            kernel=str(params.get("rbf_kernel", "thin_plate_spline")),
            smoothing=float(params.get("rbf_smoothing", 0.1)),
            epsilon=epsilon,
            neighbors=neighbors,
        )
        zz_eval, yy_eval, xx_eval = np.meshgrid(
            grid_z,
            grid_y,
            grid_x,
            indexing="ij",
        )
        eval_points = np.column_stack(
            [xx_eval.reshape(-1), yy_eval.reshape(-1), zz_eval.reshape(-1)]
        )
        output = rbf(eval_points).reshape(xx_eval.shape)
        return grid_x, grid_y, grid_z, output

    if method == SMOOTH3D_TV:
        if values.ndim != 3:
            raise ValueError("3D total variation expects gridded volume data.")
        denoise_fn = _require_dependency(denoise_tv_chambolle, "scikit-image")
        return x_clean, y_clean, z_clean, np.asarray(
            denoise_fn(
                values,
                weight=float(params.get("tv_weight", 0.15)),
            ),
            dtype=float,
        )

    raise ValueError(f"Unsupported 3D smoothing method: {method}")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class Smoothed(NamedTuple):
    """What a smoothing gives back: the coordinates, and the smoothed values.

    A 1D result has only ``x`` and ``y``; a 2D surface adds ``z``; a 3D
    volume adds ``values``. Whatever the method did not produce is None.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    values: np.ndarray | None = None


def smooth_series(
    dimension: int,
    method: str,
    params: Mapping[str, Any],
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
    values: np.ndarray | None = None,
) -> Smoothed:
    """Smooth one series of *dimension* 1, 2 or 3 with *method*.

    The one entry point the dialog needs: it picks the 1D, 2D or 3D
    function, and says in a ``ValueError`` what is missing when the series
    does not have the data that dimension needs.
    """
    if dimension == 3:
        if z is None or values is None:
            raise ValueError("Selected series has no 3D values.")
        x_out, y_out, z_out, values_out = smooth_3d(x, y, z, values, method, params)
        return Smoothed(
            np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float),
            np.asarray(z_out, dtype=float), np.asarray(values_out, dtype=float),
        )
    if dimension == 2:
        if z is None:
            raise ValueError("Selected series has no Z values.")
        x_out, y_out, z_out = smooth_2d(x, y, z, method, params)
        return Smoothed(
            np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float),
            np.asarray(z_out, dtype=float),
        )
    x_clean, y_clean = clean_xy(x, y)
    return Smoothed(x_clean, np.asarray(smooth_1d(x_clean, y_clean, method, params), dtype=float))
