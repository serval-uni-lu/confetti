"""Edge-case tests for TimeSeriesKNN (no tslearn dependency)."""

from __future__ import annotations

import numpy as np
import pytest

from confetti.distances._neighbors import TimeSeriesKNN


@pytest.fixture
def train_data():
    rng = np.random.default_rng(42)
    return rng.random((6, 8, 2))


@pytest.fixture
def query_data():
    rng = np.random.default_rng(99)
    return rng.random((1, 8, 2))


class TestTimeSeriesKNNEdgeCases:
    def test_kneighbors_before_fit_raises(self, query_data):
        knn = TimeSeriesKNN(n_neighbors=2, metric="euclidean")
        with pytest.raises(RuntimeError, match="Call fit"):
            knn.kneighbors(query_data)

    def test_softdtw_precomputed_metric(self, train_data, query_data):
        knn = TimeSeriesKNN(n_neighbors=2, metric="softdtw")
        knn.fit(train_data)
        distances, indices = knn.kneighbors(query_data)
        assert distances.shape == (1, 2)
        assert indices.shape == (1, 2)
        assert np.all(np.isfinite(distances))

    def test_precomputed_no_distance(self, train_data, query_data):
        knn = TimeSeriesKNN(n_neighbors=2, metric="softdtw")
        knn.fit(train_data)
        result = knn.kneighbors(query_data, return_distance=False)
        assert not isinstance(result, tuple)
        assert result.shape == (1, 2)
