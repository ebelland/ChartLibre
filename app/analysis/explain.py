"""What a fitted model has learned, for any model: permutation importance and partial dependence (AN-18).

Model-agnostic: all either needs is a prediction function over a numeric
matrix, so the same two answers come out of a tree, a forest, a boosted
model or anything else that predicts.

* Permutation importance: how much the model's score drops when one
  column's values are shuffled - the column's link to the response broken,
  everything else left as it was. On held-out rows when there are some, so
  a column the model only uses to fit noise scores nothing.
* Partial dependence: the model's average prediction as one column is set
  to each value of a grid, the other columns left as each row has them -
  the shape of that column's effect, whatever the model.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

Predict = Callable[[np.ndarray], np.ndarray]
Score = Callable[[np.ndarray, np.ndarray], float]


def permutation_importance(predict: Predict, x: np.ndarray, y: np.ndarray, score: Score, *,
                           repeats: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(mean, standard deviation) of the score's drop when each column is shuffled."""
    rng = np.random.default_rng(seed)
    base = score(y, predict(x))
    drops = np.zeros((x.shape[1], repeats))
    for column in range(x.shape[1]):
        for repeat in range(repeats):
            shuffled = x.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            drops[column, repeat] = base - score(y, predict(shuffled))
    return drops.mean(axis=1), drops.std(axis=1, ddof=1) if repeats > 1 else np.zeros(x.shape[1])


def grid_for(values: np.ndarray, points: int = 20) -> np.ndarray:
    """The values to set a column to: each distinct one, or quantiles 5%..95% when there are many."""
    finite = np.unique(values[np.isfinite(values)])
    if finite.size <= points:
        return finite
    return np.unique(np.quantile(finite, np.linspace(0.05, 0.95, points)))


def partial_dependence(predict: Predict, x: np.ndarray, column: int, grid: np.ndarray) -> np.ndarray:
    """The average prediction with *column* set to each value of *grid*."""
    out = np.empty(grid.size)
    for index, value in enumerate(grid):
        fixed = x.copy()
        fixed[:, column] = value
        out[index] = float(np.mean(predict(fixed)))
    return out


def r_squared(y: np.ndarray, predicted: np.ndarray) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - predicted) ** 2)) / total if total > 0 else 0.0


def accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(predicted)))
