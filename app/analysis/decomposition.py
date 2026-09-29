"""Decomposition and manifold learning: the arithmetic behind that operation.

Several series are combined into one feature matrix - columns are series,
rows are a shared grid of x values every series is resampled onto - and
reduced: PCA, ICA and NMF give components that mean something as a curve
over the grid; t-SNE, Isomap and LLE give an exploratory 2D embedding of
the grid samples. Plain arrays in, no Qt, nothing logged (todo R-01); what
had to be adjusted to fit the data comes back as notes for the caller to show.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

KIND_PCA = "pca"
KIND_FASTICA = "fastica"
KIND_NMF = "nmf"
KIND_TSNE = "tsne"
KIND_ISOMAP = "isomap"
KIND_LLE = "lle"

MANIFOLD_KINDS: frozenset[str] = frozenset({KIND_TSNE, KIND_ISOMAP, KIND_LLE})


@dataclass(frozen=True, slots=True)
class Decomposition:
    """Components over the shared grid: ``values`` is ``(grid points, components)``."""

    values: np.ndarray
    component_names: list[str]
    #: PCA only.
    explained_variance_ratio: list[float] | None = None


@dataclass(frozen=True, slots=True)
class Embedding:
    """One 2D point per grid sample, and what was adjusted to make it fit."""

    x: np.ndarray
    y: np.ndarray
    #: E.g. "Perplexity 30 does not fit 20 sampled points; using 5 instead."
    notes: tuple[str, ...] = field(default_factory=tuple)


def shared_grid(xy_pairs: Sequence[tuple[np.ndarray, np.ndarray]], n_grid: int) -> np.ndarray:
    """An evenly spaced grid over the x range every series covers."""
    low = max(float(x.min()) for x, _y in xy_pairs)
    high = min(float(x.max()) for x, _y in xy_pairs)
    if not (high > low):
        raise ValueError(
            "the selected series' x ranges do not overlap; nothing to resample onto a shared grid"
        )
    return np.linspace(low, high, max(2, int(n_grid)))


def feature_matrix(grid: np.ndarray, xy_pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Each series interpolated onto *grid*, one column per series."""
    return np.column_stack([np.interp(grid, x, y) for x, y in xy_pairs])


def decompose(kind: str, matrix: np.ndarray, n_components: int) -> Decomposition:
    """PCA, FastICA or NMF of *matrix*; the component count is clamped to the series."""
    n_series = matrix.shape[1]
    n_components = max(1, min(int(n_components), n_series))

    if kind == KIND_PCA:
        estimator = PCA(n_components=n_components, random_state=0)
        values = estimator.fit_transform(matrix)
        ratio = [float(v) for v in estimator.explained_variance_ratio_]
        prefix = "PC"
    elif kind == KIND_FASTICA:
        values = FastICA(n_components=n_components, random_state=0, max_iter=1000).fit_transform(matrix)
        ratio = None
        prefix = "IC"
    elif kind == KIND_NMF:
        if float(matrix.min()) < 0.0:
            raise ValueError(
                "NMF requires non-negative data; the selected series include negative values"
            )
        values = NMF(n_components=n_components, random_state=0, max_iter=1000).fit_transform(matrix)
        ratio = None
        prefix = "Component"
    else:
        raise ValueError(f"Unsupported decomposition: {kind}")

    return Decomposition(values, [f"{prefix}{index + 1}" for index in range(n_components)], ratio)


def clamp_perplexity(requested: float, n_samples: int) -> tuple[float, str]:
    """A t-SNE perplexity that fits *n_samples* (scikit-learn wants it below), and a note if changed."""
    limit = max(5.0, (n_samples - 1) / 3.0)
    if requested >= n_samples or requested > limit:
        return limit, (
            f"Perplexity {requested:g} does not fit {n_samples} sampled "
            f"points; using {limit:g} instead."
        )
    return requested, ""


def clamp_neighbors(requested: int, n_samples: int) -> tuple[int, str]:
    """A neighbour count that fits *n_samples*, and a note if changed."""
    limit = max(2, n_samples - 1)
    if requested > limit:
        return limit, (
            f"n_neighbors {requested} does not fit {n_samples} sampled "
            f"points; using {limit} instead."
        )
    return requested, ""


def embed(kind: str, matrix: np.ndarray, *, perplexity: float = 30.0, n_neighbors: int = 5) -> Embedding:
    """A 2D embedding of the rows of *matrix* with t-SNE, Isomap or LLE."""
    n_samples = matrix.shape[0]
    notes: list[str] = []

    if kind == KIND_TSNE:
        value, note = clamp_perplexity(float(perplexity), n_samples)
        embedding = TSNE(n_components=2, random_state=0, perplexity=value).fit_transform(matrix)
    elif kind == KIND_ISOMAP:
        neighbors, note = clamp_neighbors(int(n_neighbors), n_samples)
        embedding = Isomap(n_components=2, n_neighbors=neighbors).fit_transform(matrix)
    elif kind == KIND_LLE:
        neighbors, note = clamp_neighbors(int(n_neighbors), n_samples)
        embedding = LocallyLinearEmbedding(
            n_components=2, n_neighbors=neighbors, random_state=0
        ).fit_transform(matrix)
    else:
        raise ValueError(f"Unsupported embedding: {kind}")

    if note:
        notes.append(note)
    return Embedding(
        np.asarray(embedding[:, 0], dtype=float),
        np.asarray(embedding[:, 1], dtype=float),
        tuple(notes),
    )
