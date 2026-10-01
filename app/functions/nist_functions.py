"""NIST reference models: five of the certified nonlinear regression problems.

The models of the NIST Statistical Reference Datasets
(https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml) used by the
"NIST reference datasets" demo project: pick one in the Fit dialog on its
own table, press Fit, and compare the parameters with the certified values
shown under the chart. They start from NIST's first starting point - the
far one, on purpose; Estimate gives the second, nearer one.

What ChartLibre achieves on all 27 NIST problems is measured by
dev/tests/test_nist_validation.py (python3 dev/tools/nist_report.py).
"""
from __future__ import annotations

import numpy as np

from app.functions.base import base_function


class nist_misra1a(base_function):
    name = "Misra1a (NIST, lower difficulty)"
    category = "NIST reference models"
    description = "Dental research, monomolecular adsorption: volume against pressure."
    expression = "<b>Misra1a</b><br>y = b1 (1 - exp(-b2 x))"
    p0 = [500.0, 0.0001]
    params = ["b1", "b2"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * (1.0 - np.exp(-p[1] * x))

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        del x, y
        return [250.0, 0.0005]


class nist_gauss3(base_function):
    name = "Gauss3 (NIST, average difficulty)"
    category = "NIST reference models"
    description = "Two overlapping Gaussian peaks on a decaying exponential background."
    expression = (
        "<b>Gauss3</b><br>y = b1 exp(-b2 x) + b3 exp(-(x - b4)² / b5²) "
        "+ b6 exp(-(x - b7)² / b8²)"
    )
    p0 = [94.9, 0.009, 90.1, 113.0, 20.0, 73.8, 140.0, 20.0]
    params = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (
            p[0] * np.exp(-p[1] * x)
            + p[2] * np.exp(-((x - p[3]) ** 2) / p[4] ** 2)
            + p[5] * np.exp(-((x - p[6]) ** 2) / p[7] ** 2)
        )

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        del x, y
        return [96.0, 0.0096, 80.0, 110.0, 25.0, 74.0, 139.0, 25.0]


class nist_thurber(base_function):
    name = "Thurber (NIST, higher difficulty)"
    category = "NIST reference models"
    description = "Semiconductor electron mobility against the log of density: a rational function."
    expression = "<b>Thurber</b><br>y = (b1 + b2 x + b3 x² + b4 x³) / (1 + b5 x + b6 x² + b7 x³)"
    p0 = [1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03]
    params = ["b1", "b2", "b3", "b4", "b5", "b6", "b7"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3) / (1.0 + p[4] * x + p[5] * x**2 + p[6] * x**3)

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        del x, y
        return [1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05]


class nist_eckerle4(base_function):
    name = "Eckerle4 (NIST, higher difficulty)"
    category = "NIST reference models"
    description = "Circular interference transmittance against wavelength: one Gaussian peak."
    expression = "<b>Eckerle4</b><br>y = (b1 / b2) exp(-½ ((x - b3) / b2)²)"
    p0 = [1.0, 10.0, 500.0]
    params = ["b1", "b2", "b3"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (p[0] / p[1]) * np.exp(-0.5 * ((x - p[2]) / p[1]) ** 2)

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        del x, y
        return [1.5, 5.0, 450.0]


class nist_rat43(base_function):
    name = "Rat43 (NIST, higher difficulty)"
    category = "NIST reference models"
    description = "Onion bulb dry weight against growing time: a generalised logistic."
    expression = "<b>Rat43</b><br>y = b1 / (1 + exp(b2 - b3 x))^(1/b4)"
    p0 = [100.0, 10.0, 1.0, 1.0]
    params = ["b1", "b2", "b3", "b4"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] / ((1.0 + np.exp(p[1] - p[2] * x)) ** (1.0 / p[3]))

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        del x, y
        return [700.0, 5.0, 0.75, 1.3]
