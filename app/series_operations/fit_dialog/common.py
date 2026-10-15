"""The fit dialog's result, its worker job and the helpers its parts share."""
from __future__ import annotations

from html import escape as html_escape
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import re

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
    #: Fit every series of a global job on its own (no parameter is shared).
    separate: bool = False
    #: The per-series fits of a separate job, filled by :meth:`run`.
    parts: list[fit_engine.CurveFit] | None = None

    def run(self, should_stop: Callable[[], bool] | None = None) -> fit_engine.CurveFit:
        """The calculation alone - no Qt, safe on a worker thread."""
        if self.separate and self.global_parts is not None:
            base_model, layout, _names = self.global_parts
            outcome, self.parts = fit_engine.fit_separately(
                base_model,
                layout,
                self.x,
                self.target,
                self.p0,
                self.lower,
                self.upper,
                self.fixed,
                sigma=self.sigma,
                optimise=self.optimise,
                optimizer=self.optimizer,
                loss=self.loss,
                max_nfev=self.max_nfev,
                weighted=self.weighted,
                should_stop=should_stop,
            )
            return outcome
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

#: Words the catalogue spells out where the formula writes a letter.
_GREEK = {
    "alpha": "αα", "beta": "ββ", "gamma": "γΓ", "delta": "δΔ", "epsilon": "εϵ", "eta": "ηΗ",
    "theta": "θΘ", "kappa": "κ", "lambda": "λΛ", "mu": "μ", "nu": "ν", "xi": "ξ", "rho": "ρ",
    "sigma": "σΣ", "tau": "τ", "phi": "φΦ", "chi": "χ", "psi": "ψΨ", "omega": "ωΩ",
}
#: Parameter names that are a role, not a symbol, and the symbol the formulas use for it.
_SYMBOL_ALIASES = {
    "intercept": ("b",), "slope": ("m",), "gradient": ("m",),
    "offset": ("C",), "constant": ("C",), "baseline": ("C",),
    "x shift": ("x0", "x₀"),
}
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
#: What continues a symbol: letters, digits and subscripts - not a superscript ("σ²" is σ squared).
_SYMBOL_CHAR = "[A-Za-z0-9_\u0370-\u03ff\u2080-\u2089]"


def _symbol_pattern(alternatives: str) -> str:
    """A regex for one of *alternatives* standing alone, or run into the variable: "k₁x" is k₁ times x."""
    return rf"(?<!{_SYMBOL_CHAR})(?:{alternatives})(?:(?!{_SYMBOL_CHAR})|(?=x(?!{_SYMBOL_CHAR})))"


def _symbol_candidates(name: str) -> list[str]:
    """The spellings a formula may use for a parameter called *name*, best first.

    The whole name; the role it stands for ("offset" is C); then each of its
    words, last first - the catalogue ends a name with its symbol ("rate k",
    "amplitude A", "midpoint x0") - each as written, with a Greek word as the
    letter and with a trailing digit as a subscript (x0 -> x₀, A1 -> A₁).
    """
    name = name.strip()
    candidates = [name, *_SYMBOL_ALIASES.get(name.lower(), ())]
    for word in reversed(name.split()):
        candidates.append(word)
        candidates.extend(_GREEK.get(word.lower(), ()))
        head = word.rstrip("0123456789")
        if head and head != word:
            candidates.append(head + word[len(head):].translate(_SUBSCRIPTS))
    return list(dict.fromkeys(candidates))


def _expression_symbols(visible: str, names: Sequence[str]) -> list[str | None]:
    """The symbol each parameter has in the formula text, None where it has none.

    A symbol stands for one parameter only (the first to claim it), and has to
    stand alone: "k" is found in "exp(-k x)" but not inside "kB".
    """
    taken: set[str] = set()
    symbols: list[str | None] = []
    for name in names:
        found = None
        for candidate in _symbol_candidates(str(name)):
            if not candidate or candidate in taken or candidate in {"x", "y", "z"}:
                continue
            if re.search(_symbol_pattern(re.escape(candidate)), visible):
                found = candidate
                break
        if found is not None:
            taken.add(found)
        symbols.append(found)
    return symbols


def _substitute_expression_parameters(
    expression: str, names: Sequence[str], params: np.ndarray
) -> str:
    """Put each fitted value where its symbol stands in the formula text (not in HTML tags).

    A negative value goes in brackets, so "x - x0" reads "x - (-0.5)".
    """
    values = np.asarray(params, dtype=float).reshape(-1)
    visible = re.sub(r"<[^>]+>", " ", expression)
    replacements: dict[str, str] = {}
    for symbol, value in zip(_expression_symbols(visible, names), values):
        if symbol is not None:
            text = f"{float(value):.8g}"
            replacements[symbol] = f"({text})" if text.startswith("-") else text
    if not replacements:
        return expression
    pattern = re.compile(
        _symbol_pattern("|".join(re.escape(symbol) for symbol in sorted(replacements, key=len, reverse=True)))
    )
    parts = re.split(r"(<[^>]+>)", expression)
    for index in range(0, len(parts), 2):
        parts[index] = pattern.sub(lambda match: replacements[match.group(0)], parts[index])
    return "".join(parts)


def _evaluated_expression_html(
    expression: str, model_name: str, names: Sequence[str], params: np.ndarray
) -> str:
    """Return the expression with evaluated parameter values substituted."""
    source = expression or html_escape(model_name)
    return _substitute_expression_parameters(source, names, params)
