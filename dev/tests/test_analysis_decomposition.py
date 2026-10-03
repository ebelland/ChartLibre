"""Decomposition and manifold learning (app.analysis.decomposition).

Series built from known sources - two curves mixed with known weights - so
the number of components, the variance they explain and the shape of the
result can be checked. No dialog, no Qt.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import decomposition as dc
from dev.tests._cases import check_all

GRID = np.linspace(0.0, 10.0, 200)
SOURCE_A = np.sin(GRID)
SOURCE_B = np.exp(-((GRID - 5.0) ** 2))
#: Three series that are mixtures of the two sources: rank 2 in disguise.
PAIRS = [
    (GRID, 1.0 * SOURCE_A + 0.5 * SOURCE_B + 2.0),
    (GRID, 0.3 * SOURCE_A + 2.0 * SOURCE_B + 1.0),
    (GRID, 2.0 * SOURCE_A - 1.0 * SOURCE_B + 3.0),
]
MATRIX = dc.feature_matrix(GRID, PAIRS)


def test_the_shared_grid_covers_only_the_range_every_series_has() -> None:
    pairs = [(np.linspace(0, 10, 50), np.zeros(50)), (np.linspace(4, 14, 50), np.zeros(50))]
    grid = dc.shared_grid(pairs, 25)
    assert grid.size == 25 and grid[0] == pytest.approx(4.0) and grid[-1] == pytest.approx(10.0)


def test_series_that_do_not_overlap_are_refused() -> None:
    pairs = [(np.linspace(0, 1, 10), np.zeros(10)), (np.linspace(2, 3, 10), np.zeros(10))]
    with pytest.raises(ValueError, match="do not overlap"):
        dc.shared_grid(pairs, 20)


def test_the_feature_matrix_has_one_column_per_series_resampled_onto_the_grid() -> None:
    assert MATRIX.shape == (200, 3)
    np.testing.assert_allclose(MATRIX[:, 0], PAIRS[0][1])


def test_pca_of_a_rank_two_mixture_needs_two_components() -> None:
    result = dc.decompose(dc.KIND_PCA, MATRIX, 3)
    ratio = result.explained_variance_ratio
    assert ratio is not None
    assert sum(ratio[:2]) == pytest.approx(1.0, abs=1e-9)  # the third explains nothing
    assert result.component_names == ["PC1", "PC2", "PC3"]
    assert result.values.shape == (200, 3)


def test_the_component_count_is_clamped_to_the_number_of_series() -> None:
    assert dc.decompose(dc.KIND_PCA, MATRIX, 10).values.shape[1] == 3
    assert dc.decompose(dc.KIND_PCA, MATRIX, 0).values.shape[1] == 1


def test_ica_and_nmf_name_their_components() -> None:
    assert dc.decompose(dc.KIND_FASTICA, MATRIX, 2).component_names == ["IC1", "IC2"]
    assert dc.decompose(dc.KIND_NMF, MATRIX, 2).component_names == ["Component1", "Component2"]


def test_nmf_refuses_negative_data() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        dc.decompose(dc.KIND_NMF, MATRIX - 10.0, 2)


def test_the_reconstruction_from_two_pca_components_is_exact_for_a_rank_two_mixture() -> None:
    from sklearn.decomposition import PCA

    model = PCA(n_components=2, random_state=0)
    scores = model.fit_transform(MATRIX)
    np.testing.assert_allclose(model.inverse_transform(scores), MATRIX, atol=1e-9)
    np.testing.assert_allclose(dc.decompose(dc.KIND_PCA, MATRIX, 2).values, scores, atol=1e-9)


_CASES_A_MANIFOLD_GIVES_ONE_2D_POINT_PER_GRID_SAMPLE = [(case,) for case in sorted(dc.MANIFOLD_KINDS)]


def _a_manifold_gives_one_2d_point_per_grid_sample(kind: str) -> None:
    embedding = dc.embed(kind, MATRIX)
    assert embedding.x.shape == embedding.y.shape == (200,)
    assert np.all(np.isfinite(embedding.x)) and np.all(np.isfinite(embedding.y))


def test_a_manifold_gives_one_2d_point_per_grid_sample() -> None:
    check_all(_a_manifold_gives_one_2d_point_per_grid_sample, _CASES_A_MANIFOLD_GIVES_ONE_2D_POINT_PER_GRID_SAMPLE)


def test_a_perplexity_that_does_not_fit_is_clamped_and_reported() -> None:
    small = MATRIX[:20]
    embedding = dc.embed(dc.KIND_TSNE, small, perplexity=30.0)
    assert len(embedding.notes) == 1 and "Perplexity 30" in embedding.notes[0]
    assert dc.clamp_perplexity(5.0, 200) == (5.0, "")
    assert dc.clamp_perplexity(500.0, 200)[0] == pytest.approx(199 / 3.0)


def test_neighbours_that_do_not_fit_are_clamped_and_reported() -> None:
    assert dc.clamp_neighbors(50, 10) == (9, "n_neighbors 50 does not fit 10 sampled points; using 9 instead.")
    assert dc.clamp_neighbors(5, 10) == (5, "")


def test_an_unknown_kind_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported decomposition"):
        dc.decompose("nope", MATRIX, 2)
    with pytest.raises(ValueError, match="Unsupported embedding"):
        dc.embed("nope", MATRIX)
