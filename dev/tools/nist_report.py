"""Print how many digits ChartLibre gets right on the NIST reference datasets.

    python3 dev/tools/nist_report.py

The table behind dev/tests/test_nist_validation.py (todo R-13): for every
dataset, the correct significant digits (LRE) of what ChartLibre computes
against the certified values NIST publishes. 15 (11 for the nonlinear
problems) means every digit certified; 0 means none.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

from app.analysis.fit import fit_curve  # noqa: E402
from app.analysis.interpolation import MODEL_POLYNOMIAL, InterpolationSettings, interpolate  # noqa: E402
from app.analysis.statistics import describe, group_tests  # noqa: E402
from app.functions.optimizers import DOGBOX, LEVENBERG, TRUST_REGION  # noqa: E402
from dev.tests import _nist_strd as nist  # noqa: E402
from dev.tests._nist_models import LOG_RESPONSE, MODELS  # noqa: E402

#: The local least-squares methods; the global ones finish with trust region.
LEAST_SQUARES = [TRUST_REGION, DOGBOX, LEVENBERG]


def nonlinear(budget: int) -> None:
    print(f"Nonlinear least squares (Fit), budget {budget} evaluations: parameters / standard errors")
    header = "".join(f"{key[:12] + ' ' + str(start):>18}" for key in LEAST_SQUARES for start in (1, 2))
    print(f"{'dataset':10}{'difficulty':11}{header}")
    for name, model in MODELS.items():
        data = nist.nonlinear(name)
        target = np.log(data.y) if name in LOG_RESPONSE else data.y
        cells = []
        for key in LEAST_SQUARES:
            for start in (data.start1, data.start2):
                fit = fit_curve(model, data.x, target, start, optimizer=key, max_nfev=budget)
                params = min(nist.lre(a, c, nist.NONLINEAR_DIGITS) for a, c in zip(fit.params, data.params))
                std = min(nist.lre(a, c, nist.NONLINEAR_DIGITS) for a, c in zip(fit.std, data.std))
                cells.append(f"{params:5.1f} / {std:4.1f}")
        print(f"{name:10}{data.difficulty:11}" + "".join(f"{cell:>18}" for cell in cells))


def linear() -> None:
    print("\nPolynomials (Interpolation, NumPy polyfit): coefficients")
    for name in ("Norris", "Pontius", "Filip", "Wampler1", "Wampler2", "Wampler3", "Wampler4", "Wampler5"):
        data = nist.linear(name)
        degree = data.params.size - 1
        order = np.argsort(data.x, kind="stable")
        fitted = interpolate(
            MODEL_POLYNOMIAL, data.x[order], data.y[order], data.x[order], InterpolationSettings(degree=degree)
        )
        coefficients = [fitted.params[f"c{i}"] for i in range(degree + 1)][::-1]
        digits = min(nist.lre(a, c) for a, c in zip(coefficients, data.params))
        print(f"  {name:10} degree {degree:2}  {digits:5.1f}")


def univariate() -> None:
    print("\nSummary statistics (Statistics): mean / standard deviation")
    for name in ("PiDigits", "Lottery", "Lew", "Mavro", "Michelso", "NumAcc1", "NumAcc2", "NumAcc3", "NumAcc4"):
        data = nist.univariate(name)
        summary = describe(data.values)
        print(f"  {name:10} {nist.lre(summary['mean'], data.mean):5.1f} / {nist.lre(summary['std'], data.std):4.1f}")


def anova() -> None:
    print("\nOne-way ANOVA (Statistics): F")
    for name in ("SiRstv", "SmLs01", "SmLs02", "SmLs03", "AtmWtAg", "SmLs04", "SmLs05", "SmLs06",
                 "SmLs07", "SmLs08", "SmLs09"):
        data = nist.anova(name)
        statistic = float(group_tests(data.groups)[0]["statistic"])
        print(f"  {name:10} {nist.lre(statistic, data.f_statistic):5.1f}")


def main() -> None:
    warnings.simplefilter("ignore")
    with np.errstate(all="ignore"):
        nonlinear(800)
        linear()
        univariate()
        anova()


if __name__ == "__main__":
    main()
