"""Tests for the Global Alignment Kernel (GAK) metric."""

import numpy as np
import pytest

from confetti.distances._gak import cdist_gak, gak

tslearn_metrics = pytest.importorskip("tslearn.metrics")


class TestGakPairwise:
    """Tests for the pairwise gak(x, y) function."""

    def test_self_similarity_is_one(self, ts_a):
        assert gak(ts_a, ts_a) == pytest.approx(1.0, abs=1e-10)

    def test_symmetry(self, ts_a, ts_b):
        assert gak(ts_a, ts_b) == pytest.approx(gak(ts_b, ts_a), abs=1e-10)

    def test_range_zero_to_one(self, ts_a, ts_b):
        result = gak(ts_a, ts_b)
        assert 0.0 <= result <= 1.0 + 1e-10

    def test_matches_tslearn_default_sigma(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_gak(
            ts_a[np.newaxis], ts_b[np.newaxis]
        )[0, 0]
        assert gak(ts_a, ts_b) == pytest.approx(expected, abs=1e-10)

    def test_matches_tslearn_sigma_05(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_gak(
            ts_a[np.newaxis], ts_b[np.newaxis], sigma=0.5
        )[0, 0]
        assert gak(ts_a, ts_b, sigma=0.5) == pytest.approx(expected, abs=1e-10)

    def test_matches_tslearn_sigma_5(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_gak(
            ts_a[np.newaxis], ts_b[np.newaxis], sigma=5.0
        )[0, 0]
        assert gak(ts_a, ts_b, sigma=5.0) == pytest.approx(expected, abs=1e-10)

    def test_single_timestep(self, ts_single_step):
        other = np.array([[3.0, 5.0]])
        expected = tslearn_metrics.cdist_gak(
            ts_single_step[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert gak(ts_single_step, other) == pytest.approx(expected, abs=1e-10)

    def test_single_channel(self, ts_single_channel):
        other = np.zeros((5, 1))
        expected = tslearn_metrics.cdist_gak(
            ts_single_channel[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert gak(ts_single_channel, other) == pytest.approx(expected, abs=1e-10)

    def test_short_series(self, ts_short):
        other = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        expected = tslearn_metrics.cdist_gak(
            ts_short[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert gak(ts_short, other) == pytest.approx(expected, abs=1e-10)

    def test_returns_float(self, ts_a, ts_b):
        assert isinstance(gak(ts_a, ts_b), float)


class TestCdistGak:
    """Tests for the batch cdist_gak(X, Y) function."""

    def test_output_shape(self, batch_a, batch_b):
        result = cdist_gak(batch_a, batch_b)
        assert result.shape == (3, 2)

    def test_values_match_pairwise(self, batch_a, batch_b):
        result = cdist_gak(batch_a, batch_b)
        for i in range(batch_a.shape[0]):
            for j in range(batch_b.shape[0]):
                expected = gak(batch_a[i], batch_b[j])
                assert result[i, j] == pytest.approx(expected, abs=1e-10)

    def test_matches_tslearn_cdist(self, batch_a, batch_b):
        expected = tslearn_metrics.cdist_gak(batch_a, batch_b)
        result = cdist_gak(batch_a, batch_b)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matches_tslearn_cdist_sigma(self, batch_a, batch_b):
        expected = tslearn_metrics.cdist_gak(batch_a, batch_b, sigma=2.0)
        result = cdist_gak(batch_a, batch_b, sigma=2.0)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_self_similarity_diagonal(self, batch_a):
        result = cdist_gak(batch_a, batch_a)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)

    def test_symmetry(self, batch_a, batch_b):
        ab = cdist_gak(batch_a, batch_b)
        ba = cdist_gak(batch_b, batch_a)
        np.testing.assert_allclose(ab, ba.T, atol=1e-10)

    def test_single_vs_batch(self, ts_a, batch_b):
        X = ts_a[np.newaxis, :, :]
        result = cdist_gak(X, batch_b)
        assert result.shape == (1, 2)
