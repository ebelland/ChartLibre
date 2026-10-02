"""Least-squares curve fitting on plain arrays: estimates, errors, inference.

The engine behind the Fit series operation. A model is any callable
``model(x, p) -> y`` - the scanned fit functions, the multi-peak builder, a
user formula - and the result carries everything the report prints:
parameters, standard errors, covariance and correlation, t and p values,
95% confidence intervals and the goodness-of-fit measures.

Nothing here knows about Qt or the repository, so it runs in a test, a
script or a worker thread exactly as it runs behind the dialog.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from app.analysis import Stopped
from app.functions.optimizers import BY_KEY, DEFAULT_OPTIMIZER, RELATIVE_STEP, run_optimizer

Model = Callable[[np.ndarray, np.ndarray], np.ndarray]


class FitStopped(Stopped):
    """The fit was asked to stop before it finished. See app.analysis.Stopped."""


@dataclass(slots=True)
class CurveFit:
    """One fitted (or merely evaluated) model."""

    params: np.ndarray
    std: np.ndarray
    cov: np.ndarray
    corr: np.ndarray
    fit_values: np.ndarray
    residual: np.ndarray
    metrics: dict[str, float]
    success: bool
    message: str
    #: Residual degrees of freedom: points minus parameters estimated.
    dof: int
    tvalues: np.ndarray
    pvalues: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def primary_x(value: np.ndarray) -> np.ndarray:
    """The first independent variable of a 1D or 2D model input."""
    arr = np.asarray(value, dtype=float)
    return arr[:, 0] if arr.ndim == 2 else arr


def multi_peak_model(
    family: str = "Gaussian",
    count: int = 1,
    *,
    tie_width: bool = False,
    tie_eta: bool = False,
) -> Model:
    """A sum of *count* Gaussian, Lorentzian or pseudo-Voigt peaks plus an offset.

    Parameters, in order: for each peak amplitude, centre, then its width
    unless widths are tied, then its eta (pseudo-Voigt) unless tied; then
    the shared width and shared eta when tied; the offset last.
    """

    def model(x_or_xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x = primary_x(x_or_xy)
        values = np.asarray(p, dtype=float)
        idx = 0
        peaks: list[tuple[float, float, float, float]] = []
        for _i in range(count):
            amp = float(values[idx]); idx += 1
            center = float(values[idx]); idx += 1
            width_value = 1.0
            eta_value = 0.5
            if not tie_width:
                width_value = max(abs(float(values[idx])), 1e-12); idx += 1
            if family == "Pseudo-Voigt" and not tie_eta:
                eta_value = float(np.clip(values[idx], 0.0, 1.0)); idx += 1
            peaks.append((amp, center, width_value, eta_value))

        shared_width = 1.0
        if tie_width:
            shared_width = max(abs(float(values[idx])), 1e-12)
            idx += 1
        shared_eta = 0.5
        if family == "Pseudo-Voigt" and tie_eta:
            shared_eta = float(np.clip(values[idx], 0.0, 1.0))
            idx += 1
        offset = float(values[idx]) if idx < values.size else 0.0

        y = np.full_like(x, offset, dtype=float)
        for amp, center, width_value, eta_value in peaks:
            w = max(shared_width if tie_width else width_value, 1e-12)
            if family == "Gaussian":
                y += amp * np.exp(-((x - center) ** 2) / (2.0 * w * w))
            elif family == "Lorentzian":
                g2 = (0.5 * w) ** 2
                y += amp * g2 / (((x - center) ** 2) + g2)
            else:
                e = float(np.clip(shared_eta if tie_eta else eta_value, 0.0, 1.0))
                g = np.exp(-((x - center) ** 2) / (2.0 * w * w))
                lor = (0.5 * w) ** 2 / (((x - center) ** 2) + (0.5 * w) ** 2)
                y += amp * (e * lor + (1.0 - e) * g)
        return y

    return model


def model_jacobian(model: Model, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Central-difference derivative of the model with respect to each parameter.

    The step is relative to each parameter (RELATIVE_STEP): the absolute
    1e-6 it used to have for anything below 1 was larger than a parameter of
    1e-7, and the NIST Hahn1 errors came out with no correct digit.
    """
    params = np.asarray(p, dtype=float)
    base = np.asarray(model(x, params), dtype=float)
    jac = np.empty((base.size, params.size), dtype=float)
    for col in range(params.size):
        value = float(params[col])
        step = RELATIVE_STEP * (abs(value) if value != 0.0 else 1.0)
        p_plus = params.copy(); p_plus[col] += step
        p_minus = params.copy(); p_minus[col] -= step
        jac[:, col] = (
            np.asarray(model(x, p_plus), dtype=float) - np.asarray(model(x, p_minus), dtype=float)
        ) / (2.0 * step)
    return jac


def residual_jacobian(model: Model, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Jacobian of the residuals (target - model): the model's, negated."""
    return -model_jacobian(model, x, p)


def weights_sigma(data: np.ndarray) -> np.ndarray:
    """Relative weights: each residual divided by the size of its target."""
    return np.maximum(np.abs(data), np.nanmedian(np.abs(data)) * 1e-6 + 1e-12).astype(float)


def fit_metrics(target: np.ndarray, fit: np.ndarray, p_count: int) -> dict[str, float]:
    """R2, RMSE, residual sum of squares, AIC and BIC."""
    residual = target - fit
    n = int(target.size)
    ss_res = float(np.nansum(np.square(residual)))
    ss_tot = float(np.nansum(np.square(target - np.nanmean(target))))
    return {
        "rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
        "ss_res": ss_res,
        "aic": n * math.log(max(ss_res / max(n, 1), 1e-300)) + 2.0 * p_count,
        "bic": n * math.log(max(ss_res / max(n, 1), 1e-300)) + p_count * math.log(max(n, 1)),
    }


def param_uncertainty(
    jac_free: np.ndarray,
    residual: np.ndarray,
    free: np.ndarray,
    n_params: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard errors, covariance and correlation of the free parameters.

    The usual least-squares estimate: (J'J)^-1 scaled by the residual
    variance. Fixed parameters get NaN everywhere - they have no error.

    (J'J)^-1 is taken through the singular values of J rather than by
    inverting J'J: forming J'J squares the condition number, and with
    parameters of very different sizes (NIST Misra1c: 636 and 0.0002) the
    pseudo-inverse then dropped a direction and reported an error 1e14 times
    too small. Only directions at rounding level are dropped here, as
    SciPy's curve_fit does.
    """
    std = np.full(n_params, np.nan, dtype=float)
    cov_full = np.full((n_params, n_params), np.nan, dtype=float)
    corr_full = np.full((n_params, n_params), np.nan, dtype=float)
    if jac_free.size == 0 or jac_free.shape[1] == 0:
        return std, cov_full, corr_full
    try:
        dof = max(1, residual.size - jac_free.shape[1])
        # Columns to unit length first, then back: on NIST Filip the columns
        # (x^0 ... x^10) differ by ten orders of magnitude and, unscaled, the
        # rounding-level cut below dropped real directions.
        norms = np.linalg.norm(jac_free, axis=0)
        norms[norms == 0.0] = 1.0
        _u, singular, vt = np.linalg.svd(jac_free / norms, full_matrices=False)
        keep = singular > np.finfo(float).eps * max(jac_free.shape) * singular[0]
        singular, vt = singular[keep], vt[keep]
        cov = (vt.T / singular**2) @ vt / np.outer(norms, norms) * float(np.dot(residual, residual) / dof)
    except (np.linalg.LinAlgError, ValueError, IndexError):
        return std, cov_full, corr_full
    free_idx = np.flatnonzero(free)
    cov_full[np.ix_(free_idx, free_idx)] = cov
    std_free = np.sqrt(np.maximum(np.diag(cov), 0.0))
    std[free_idx] = std_free
    denom = np.outer(std_free, std_free)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0.0)
    corr_full[np.ix_(free_idx, free_idx)] = corr
    return std, cov_full, corr_full


def parameter_inference(
    params: np.ndarray, std: np.ndarray, dof: int, *, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """t value, two-sided p value (is the parameter zero?) and confidence interval.

    From Student's t with the residual degrees of freedom, as Origin, SPSS
    and statsmodels report them. NaN where a parameter has no error.
    """
    params = np.asarray(params, dtype=float)
    std = np.asarray(std, dtype=float)
    nan = np.full(params.shape, np.nan)
    if dof < 1:
        return nan, nan.copy(), nan.copy(), nan.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = np.where(std > 0, params / std, np.nan)
    pvalues = 2.0 * stats.t.sf(np.abs(tvalues), dof)
    half = stats.t.ppf(0.5 + level / 2.0, dof) * std
    return tvalues, pvalues, params - half, params + half


def confidence_band(
    model: Model,
    x: np.ndarray,
    params: np.ndarray,
    cov: np.ndarray,
    *,
    dof: int,
    residual_variance: float = 0.0,
    level: float = 0.95,
    prediction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper band of the fitted curve at *x* (delta method).

    Confidence: where the true curve lies. Prediction: where a new
    measurement would fall - wider, by the residual variance. Fixed
    parameters (NaN in *cov*) contribute nothing.
    """
    params = np.asarray(params, dtype=float)
    curve = np.asarray(model(x, params), dtype=float)
    gradient = model_jacobian(model, x, params)
    sigma = np.nan_to_num(np.asarray(cov, dtype=float), nan=0.0)
    variance = np.einsum("ij,jk,ik->i", gradient, sigma, gradient)
    if prediction:
        variance = variance + float(residual_variance)
    half = stats.t.ppf(0.5 + level / 2.0, max(int(dof), 1)) * np.sqrt(np.maximum(variance, 0.0))
    return curve - half, curve + half


# ----------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------
def linear_basis(model: Model, x: np.ndarray, n_params: int) -> np.ndarray | None:
    """The columns g_i(x) when ``model(x, p) == sum(p_i * g_i(x))``, else None.

    Checked, not assumed: the model must be zero at p = 0 and reproduce two
    unrelated parameter vectors from the columns to 1e-9. Polynomials of any
    degree, a line through the origin, a sum of fixed sinusoids pass; a
    model with a parameter inside an exp or a denominator does not.
    """
    if n_params == 0:
        return None
    try:
        with np.errstate(all="ignore"):
            zero = np.asarray(model(x, np.zeros(n_params)), dtype=float)
            basis = np.column_stack(
                [np.asarray(model(x, np.eye(n_params)[i]), dtype=float) for i in range(n_params)]
            )
            if not np.all(np.isfinite(basis)) or np.any(zero != 0.0):
                return None
            rng = np.random.default_rng(0)
            for _ in range(2):
                probe = rng.normal(0.0, 1.0, n_params) * 3.0
                value = np.asarray(model(x, probe), dtype=float)
                expected = basis @ probe
                scale = float(np.max(np.abs(expected))) or 1.0
                if not np.allclose(value, expected, rtol=1e-9, atol=1e-12 * scale):
                    return None
    except (ArithmeticError, ValueError, TypeError, IndexError):
        return None
    return basis


def _solve_linear(
    basis: np.ndarray, target: np.ndarray, free: np.ndarray, p0: np.ndarray, sigma: np.ndarray | None
) -> np.ndarray:
    """The exact least-squares parameters of a linear model, fixed ones kept."""
    rhs = target - basis[:, ~free] @ p0[~free]
    columns = basis[:, free]
    if sigma is not None:
        rhs = rhs / sigma
        columns = columns / sigma[:, None]
    # Columns scaled to unit length first: x^10 and 1 differ by ten orders of
    # magnitude on NIST Filip, and an unscaled solve loses those digits.
    norms = np.linalg.norm(columns, axis=0)
    norms[norms == 0.0] = 1.0
    solution, *_ = np.linalg.lstsq(columns / norms, rhs, rcond=None)
    params = p0.copy()
    params[free] = solution / norms
    return params


def fit_curve(
    model: Model,
    x: np.ndarray,
    target: np.ndarray,
    p0: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    fixed: np.ndarray | None = None,
    *,
    optimise: bool = True,
    optimizer: str = DEFAULT_OPTIMIZER,
    loss: str = "linear",
    max_nfev: int = 800,
    weighted: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> CurveFit:
    """Fit *model* to (x, target) starting from *p0*, or just evaluate it.

    ``optimise=False`` evaluates the model at *p0* as it stands - what the
    dialog's Preview and Apply do - with the errors of every parameter
    computed at that point, so a hand-tuned curve still gets error bars.

    *should_stop* is polled before every evaluation of the model; when it
    returns True the fit raises :class:`FitStopped` - how a fit running in
    the background is stopped.
    """
    if optimizer not in BY_KEY:
        # Before the exact linear path too, which never reaches an optimiser.
        raise ValueError(f"Unknown optimizer {optimizer!r}: use one of {', '.join(BY_KEY)}.")
    p0 = np.asarray(p0, dtype=float)
    n_params = p0.size
    lower = np.full(n_params, -np.inf) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full(n_params, np.inf) if upper is None else np.asarray(upper, dtype=float)
    fixed = np.zeros(n_params, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool)
    target = np.asarray(target, dtype=float)

    free = ~fixed if optimise else np.zeros_like(fixed, dtype=bool)
    sigma = weights_sigma(target) if weighted else None
    if not np.any(free):
        p_opt = p0.copy()
        jac_free = np.empty((target.size, 0))
        success = True
        message = (
            "Evaluated at the current parameters."
            if not optimise
            else "All parameters fixed; evaluated model only."
        )
    else:

        def residual_fun(p_free: np.ndarray) -> np.ndarray:
            if should_stop is not None and should_stop():
                raise FitStopped("The fit was stopped.")
            p = p0.copy()
            p[free] = p_free
            r = target - model(x, p)
            if sigma is not None:
                r = r / sigma
            return np.asarray(r, dtype=float)

        # A model linear in its parameters (any polynomial) has an exact
        # least-squares answer: solved directly, not iterated towards. The
        # optimiser worsened a good start on high-degree polynomials (NIST
        # Filip, degree 10: 7.3 correct digits in, 5.4 out, and none in the
        # errors). Only for plain least squares, and only when the answer
        # respects the bounds; otherwise the optimiser runs as before.
        basis = linear_basis(model, x, n_params) if loss == "linear" else None
        exact = _solve_linear(basis, target, free, p0, sigma) if basis is not None else None
        if exact is not None and np.all(np.isfinite(exact)) and np.all(
            (exact[free] >= lower[free]) & (exact[free] <= upper[free])
        ):
            p_opt = exact
            jac_free = -basis[:, free] if basis is not None else np.empty((target.size, 0))
            success = True
            message = "Linear least squares: the model is linear in its parameters, solved exactly."
        else:
            outcome = run_optimizer(
                optimizer,
                residual_fun,
                p0[free],
                lower[free],
                upper[free],
                max_nfev=max(1, int(max_nfev)),
                loss=loss,
            )
            p_opt = p0.copy()
            p_opt[free] = outcome.params
            # The model's own central differences rather than the optimiser's
            # Jacobian: those are forward differences taken at the last step,
            # good enough to steer by and two digits short for error bars.
            jac_free = residual_jacobian(model, x, p_opt)[:, free]
            success = bool(outcome.success)
            message = f"{BY_KEY[outcome.optimizer].label}: {outcome.message}"

    fit_values = np.asarray(model(x, p_opt), dtype=float)
    residual = target - fit_values
    uncertainty_free = free.copy()
    if not optimise and n_params:
        jac_free = residual_jacobian(model, x, p_opt)
        uncertainty_free = np.ones_like(fixed, dtype=bool)
    estimated = int(np.count_nonzero(uncertainty_free))
    metrics = fit_metrics(target, fit_values, estimated)
    if sigma is None:
        std, cov, corr = param_uncertainty(jac_free, residual, uncertainty_free, n_params)
    else:
        # The weighted problem is the one solved, so its errors come from it:
        # Jacobian and residuals both divided by sigma. (The weighted
        # Jacobian used to meet the unweighted residuals here.)
        std, cov, corr = param_uncertainty(
            jac_free / sigma[:, None], residual / sigma, uncertainty_free, n_params
        )
    dof = int(target.size - estimated)
    tvalues, pvalues, ci_low, ci_high = parameter_inference(p_opt, std, dof)
    if dof > 0:
        metrics["reduced_chi2"] = float(metrics["ss_res"] / dof)
    return CurveFit(
        params=p_opt,
        std=std,
        cov=cov,
        corr=corr,
        fit_values=fit_values,
        residual=residual,
        metrics=metrics,
        success=success,
        message=message,
        dof=dof,
        tvalues=tvalues,
        pvalues=pvalues,
        ci_low=ci_low,
        ci_high=ci_high,
    )
