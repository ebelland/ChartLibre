"""The fit dialog's result, its worker job and the helpers its parts share."""
from __future__ import annotations

from html import escape as html_escape
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from app.analysis import fit as fit_engine

from app.series_operations.results import FrameResult
from app.logs.logger import applogger

def _split_xy(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        applogger.error("This model requires 2D input with x and y independent columns.")
    return arr[:, 0], arr[:, 1]


class _SurfaceInitialGuessAdapter:
    """Adapt a surface function's 3-argument ``initial_guess`` to the plain
    ``(x, y)`` shape ``ask_the_function``/``choose_starting_point`` expect.

    Those two (``app/functions/starting_point.py``) are shared with every 1D
    fit function, whose ``initial_guess(x, y)`` reads ``x`` as the sole
    independent variable and ``y`` as the target. A surface function's own
    ``initial_guess(x, y, z)`` needs x and y separately and z as the target -
    so this unpacks the ``(N, 2)`` array this dialog builds for a 2D fit
    before forwarding, rather than teaching the shared starting-point module
    about a minority of functions that take one more argument.
    """

    def __init__(self, cls: Any) -> None:
        self._cls = cls

    def initial_guess(self, xy: np.ndarray, z: np.ndarray) -> list[float] | None:
        arr = np.asarray(xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        return self._cls.initial_guess(arr[:, 0], arr[:, 1], np.asarray(z, dtype=float))

@dataclass(slots=True)
class SeriesFitResult(FrameResult):
    source_table: str
    x_col: str
    target_col: str
    x2_col: str | None
    fit_mode: str
    model: str
    params: np.ndarray
    param_std: np.ndarray
    param_corr: np.ndarray
    param_names: list[str]
    expression: str
    evaluated_expression: str
    metrics: dict[str, float]
    output_table: str
    frame: pd.DataFrame
    message: str
    #: t value, p value (is the parameter zero?) and 95% interval per
    #: parameter, from Student's t with the residual degrees of freedom.
    param_t: np.ndarray | None = None
    param_p: np.ndarray | None = None
    param_ci_low: np.ndarray | None = None
    param_ci_high: np.ndarray | None = None
    dof: int = 0

    @property
    def parameters(self) -> dict[str, Any]:
        return {name: float(value) for name, value in zip(self.param_names, self.params)}

    @property
    def series(self) -> tuple[str, ...]:
        return (self.source_table,)


@dataclass(slots=True)
class _FitJob:
    """Everything one fit needs, read off the window before it starts.

    A fit may run in the background while the window stays usable, so what
    was selected when Fit was pressed - model, columns, names - is captured
    here rather than read again when the result comes back.
    """

    model: Callable[[np.ndarray, np.ndarray], np.ndarray]
    x: np.ndarray
    target: np.ndarray
    clean: pd.DataFrame
    p0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fixed: np.ndarray
    optimizer: str
    loss: str
    max_nfev: int
    weighted: bool
    source_table: str
    x_col: str
    x2_col: str | None
    target_col: str
    is_2d: bool
    model_name: str
    param_names: list[str]
    expression: str
    output_table: str
    #: Fit (optimise the parameters) or merely evaluate them.
    optimise: bool = False
    #: Each point's error, from the series' yerr column, when weighting by it.
    sigma: np.ndarray | None = None
    #: A global fit's (model of one series, GlobalLayout, series names); see global_fit.py.
    global_parts: tuple | None = None

    def run(self, should_stop: Callable[[], bool] | None = None) -> fit_engine.CurveFit:
        """The calculation alone - no Qt, safe on a worker thread."""
        return fit_engine.fit_curve(
            self.model,
            self.x,
            self.target,
            self.p0,
            self.lower,
            self.upper,
            self.fixed,
            optimise=self.optimise,
            optimizer=self.optimizer,
            loss=self.loss,
            max_nfev=self.max_nfev,
            weighted=self.weighted,
            sigma=self.sigma,
            should_stop=should_stop,
        )

def _evaluated_expression_html(
    expression: str, model_name: str, names: Sequence[str], params: np.ndarray
) -> str:
    """The expression plus the value each parameter took."""
    expression = expression or html_escape(model_name)
    rows = [
        f"{html_escape(names[row] if row < len(names) else f'p[{row}]')} = {float(value):.8g}"
        for row, value in enumerate(np.asarray(params, dtype=float))
    ]
    if not rows:
        return expression
    return f"{expression}<br><br><b>Evaluated parameters</b><br>" + "<br>".join(rows)
