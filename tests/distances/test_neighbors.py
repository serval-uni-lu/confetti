"""Tests for the TimeSeriesKNN neighbor search."""

import numpy as np
import pytest

from confetti.distances._neighbors import TimeSeriesKNN

tslearn_neighbors = pytest.importorskip("tslearn.neighbors")


@pytest.fixture
def train_data():
    """Small training set of (N=6, T=8, C=2) time series."""
    rng = np.random.default_rng(7)
    return rng.random((6, 8, 2))


@pytest.fixture
def query_data():
    """Single query instance of (1, T=8, C=2)."""
    rng = np.random.default_rng(13)
    return rng.random((1, 8, 2))


class TestTimeSeriesKNNEuclidean:
    """Euclidean metric tests."""

    def test_matches_tslearn_indices(self, train_data, query_data):
        knn_ours = TimeSeriesKNN(n_neighbors=3, metric="euclidean")
        knn_ours.fit(train_data)
        dist_ours, ind_ours = knn_ours.kneighbors(query_data, return_distance=True)

        knn_ts = tslearn_neighbors.KNeighborsTimeSeries(n_neighbors=3, metric="euclidean")
        knn_ts.fit(train_data)
        dist_ts, ind_ts = knn_ts.kneighbors(query_data, return_distance=True)

        np.testing.assert_array_equal(ind_ours, ind_ts)

    def test_matches_tslearn_distances(self, train_data, query_data):
        knn_ours = TimeSeriesKNN(n_neighbors=3, metric="euclidean")
        knn_ours.fit(train_data)
        dist_ours, _ = knn_ours.kneighbors(query_data, return_distance=True)

        knn_ts = tslearn_neighbors.KNeighborsTimeSeries(n_neighbors=3, metric="euclidean")
        knn_ts.fit(train_data)
        dist_ts, _ = knn_ts.kneighbors(query_data, return_distance=True)

        np.testing.assert_allclose(dist_ours, dist_ts, atol=1e-10)

    def test_kneighbors_no_distance(self, train_data, query_data):
        knn = TimeSeriesKNN(n_neighbors=2, metric="euclidean")
        knn.fit(train_data)
        result = knn.kneighbors(query_data, return_distance=False)
        assert result.shape == (1, 2)

    def test_single_neighbor(self, train_data, query_data):
        knn = TimeSeriesKNN(n_neighbors=1, metric="euclidean")
        knn.fit(train_data)
        dist, ind = knn.kneighbors(query_data, return_distance=True)
        assert dist.shape == (1, 1)
        assert ind.shape == (1, 1)


class TestTimeSeriesKNNDtw:
    """DTW metric tests."""

    def test_matches_tslearn_indices(self, train_data, query_data):
        knn_ours = TimeSeriesKNN(n_neighbors=3, metric="dtw")
        knn_ours.fit(train_data)
        dist_ours, ind_ours = knn_ours.kneighbors(query_data, return_distance=True)

        knn_ts = tslearn_neighbors.KNeighborsTimeSeries(n_neighbors=3, metric="dtw")
        knn_ts.fit(train_data)
        dist_ts, ind_ts = knn_ts.kneighbors(query_data, return_distance=True)

        np.testing.assert_array_equal(ind_ours, ind_ts)

    def test_matches_tslearn_distances(self, train_data, query_data):
        knn_ours = TimeSeriesKNN(n_neighbors=3, metric="dtw")
        knn_ours.fit(train_data)
        dist_ours, _ = knn_ours.kneighbors(query_data, return_distance=True)

        knn_ts = tslearn_neighbors.KNeighborsTimeSeries(n_neighbors=3, metric="dtw")
        knn_ts.fit(train_data)
        dist_ts, _ = knn_ts.kneighbors(query_data, return_distance=True)

        np.testing.assert_allclose(dist_ours, dist_ts, atol=1e-10)

    def test_with_sakoe_chiba(self, train_data, query_data):
        params = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 2}

        knn_ours = TimeSeriesKNN(n_neighbors=2, metric="dtw", metric_params=params)
        knn_ours.fit(train_data)
        dist_ours, ind_ours = knn_ours.kneighbors(query_data, return_distance=True)

        knn_ts = tslearn_neighbors.KNeighborsTimeSeries(
            n_neighbors=2, metric="dtw", metric_params=params,
        )
        knn_ts.fit(train_data)
        dist_ts, ind_ts = knn_ts.kneighbors(query_data, return_distance=True)

        np.testing.assert_array_equal(ind_ours, ind_ts)
        np.testing.assert_allclose(dist_ours, dist_ts, atol=1e-10)

    def test_all_neighbors(self, train_data, query_data):
        knn = TimeSeriesKNN(n_neighbors=6, metric="dtw")
        knn.fit(train_data)
        dist, ind = knn.kneighbors(query_data, return_distance=True)
        assert dist.shape == (1, 6)
        assert ind.shape == (1, 6)
        assert sorted(ind[0].tolist()) == list(range(6))
