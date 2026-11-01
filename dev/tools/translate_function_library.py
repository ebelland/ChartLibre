"""Write app/functions/library.py: the fit-function classes (app/functions) as SymPy expressions.

Each class's execute() is run once with SymPy symbols in place of x, y and the parameters, and a stand-in
for numpy and scipy.special that answers with SymPy: np.exp -> exp, np.where -> Piecewise, np.clip ->
Min/Max, spsp.erf -> erf, spsp.gammaln -> loggamma... The guards keep a readable name - pos(s), nz(d),
safe(x) - which app.functions.library expands into the same Piecewise the numpy guard computes, so the
expression gives the old function's numbers (dev/tests/test_function_library.py checks each one).

What cannot be SymPy (Voigt: scipy.special.wofz) is written as a "code" entry: the old body, run as Python.

    python -m dev.tools.translate_function_library
"""
from __future__ import annotations

import inspect
import keyword
import pprint
import re
import textwrap
import types
from pathlib import Path

import sympy as sp

import app.functions.functions as basic
import app.functions.nist_functions as nist
import app.functions.surface_functions as surfaces

TARGET = Path(__file__).resolve().parents[2] / "app" / "functions" / "library.py"
POS, NZ, SAFE, FINITE = sp.Function("pos"), sp.Function("nz"), sp.Function("safe"), sp.Function("finite")
EPS = 1e-12
_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
          "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"}


class _NotSymbolic(Exception):
    pass


def _where(condition, a, b):
    return sp.Piecewise((a, condition), (b, True))


_np = types.SimpleNamespace(
    exp=sp.exp, log=sp.log, sqrt=sp.sqrt, sin=sp.sin, cos=sp.cos, tan=sp.tan, tanh=sp.tanh, arctan=sp.atan,
    arcsin=sp.asin, abs=sp.Abs, log1p=lambda v: sp.log(1 + v), power=lambda a, b: a ** b, pi=sp.pi, e=sp.E,
    maximum=lambda a, b: sp.Max(a, b), minimum=lambda a, b: sp.Min(a, b),
    clip=lambda v, lo, hi: sp.Min(sp.Max(v, lo), hi), where=_where,
    full_like=lambda _x, value, dtype=None: sp.sympify(value), ones_like=lambda _x, dtype=None: sp.Integer(1),
    zeros_like=lambda _x, dtype=None: sp.Integer(0), asarray=lambda v, dtype=None: v,
    nan_to_num=lambda v, **_k: FINITE(v),  # the old function's 0 for an infinity at its edge
    real=lambda v: (_ for _ in ()).throw(_NotSymbolic("complex")), ndarray=object,
)
_spsp = types.SimpleNamespace(
    erf=sp.erf, erfc=sp.erfc, gammaln=sp.loggamma, gamma=sp.gamma,
    gammainc=lambda a, v: sp.lowergamma(a, v) / sp.gamma(a),
    betainc=lambda a, b, t: sp.betainc_regularized(a, b, 0, t),
    wofz=lambda _z: (_ for _ in ()).throw(_NotSymbolic("wofz")),
)
_HELPERS = {"np": _np, "spsp": _spsp, "_pos": lambda v, eps=EPS: POS(v), "_eps": lambda v, eps=EPS: NZ(v),
            "_safe_x": lambda v, eps=EPS: SAFE(v), "_positive_x": lambda v, eps=EPS: sp.Max(v, EPS)}
#: Written by hand: a body SymPy cannot run as it is (a Python `if` on a parameter).
_BY_HAND = {
    "gaussian2d": ("a + amp*exp(-((x - x0)**2/(2*nz(sx)**2) + (y - y0)**2/(2*nz(sy)**2)))",
                   ["a", "amp", "x0", "sx", "y0", "sy"]),
}


def _identifier(label: str, taken: set[str]) -> str:
    """A parameter's label as a name an expression can use: "exponent n" -> n, "x0 (centre)" -> x0."""
    words = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", label)
    last = words[-1] if words else "p"
    # The symbol at the end of a label ("scale lambda", "angular frequency omega", "midpoint x0"); else the words.
    name = last if len(words) > 1 and (len(last) <= 3 or last.lower() in _GREEK) else "_".join(words) or "p"
    name = name.lower() if name.isupper() and len(name) > 3 else name
    name = {"lambda": "lam"}.get(name, name)
    if keyword.iskeyword(name) or name in ("x", "y", "z"):
        name += "_"
    base, number = name, 2
    while name in taken:
        name, number = f"{base}{number}", number + 1
    taken.add(name)
    return name


def _translate(cls: type, ndim: int) -> dict:
    labels = [str(label) for label in cls.params]
    taken: set[str] = set()
    names = [_identifier(label, taken) for label in labels]
    entry = {"name": cls.name, "category": cls.category, "description": cls.description,
             "keywords": getattr(cls, "keywords", ""), "goals": list(getattr(cls, "goals", []) or []),
             "params": names, "labels": labels, "values": [float(v) for v in cls.p0], "ndim": ndim,
             "guess": cls.__name__ if "initial_guess" in cls.__dict__ else "", "class_name": cls.__name__}
    if cls.__name__ in _BY_HAND:
        expression, names = _BY_HAND[cls.__name__]
        return {**entry, "params": names, "kind": "expression", "code": expression}
    function = cls.__dict__["execute"].__func__ if isinstance(cls.__dict__["execute"], staticmethod) \
        else cls.execute
    runnable = types.FunctionType(function.__code__, {**vars(inspect.getmodule(cls)), **_HELPERS},
                                  function.__name__, function.__defaults__, function.__closure__)
    symbols = [sp.Symbol(name) for name in names]
    args = (sp.Symbol("x"), symbols) if ndim == 1 else (sp.Symbol("x"), sp.Symbol("y"), symbols)
    try:
        expression = runnable(*args)
    except (_NotSymbolic, TypeError) as error:
        body = textwrap.dedent("".join(inspect.getsourcelines(function)[0][2:]))
        print(f"  {cls.__name__}: kept as code ({error})")
        return {**entry, "kind": "code", "code": body}
    return {**entry, "kind": "expression", "code": str(sp.sympify(expression))}


def main() -> None:
    entries = []
    for module, ndim in ((basic, 1), (surfaces, 2), (nist, 1)):
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and "execute" in cls.__dict__ and getattr(cls, "name", ""):
                entries.append(_translate(cls, ndim))
    source = TARGET.read_text(encoding="utf-8")
    start, end = source.index("# BEGIN GENERATED"), source.index("# END GENERATED")
    generated = "# BEGIN GENERATED - by dev/tools/translate_function_library.py, do not edit by hand\n" \
                + "ENTRIES: tuple[dict, ...] = " + pprint.pformat(tuple(entries), width=118, sort_dicts=False) + "\n"
    TARGET.write_text(source[:start] + generated + source[end:], encoding="utf-8")
    print(f"{len(entries)} functions written to {TARGET}")


if __name__ == "__main__":
    main()
