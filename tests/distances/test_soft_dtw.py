"""Tests for the Soft-DTW distance metric."""

import numpy as np
import pytest

from confetti.distances._soft_dtw import cdist_soft_dtw, soft_dtw

tslearn_metrics = pytest.importorskip("tslearn.metrics")


class TestSoftDtwPairwise:
    """Tests for the pairwise soft_dtw(x, y) function."""

    def test_symmetry(self, ts_a, ts_b):
        assert soft_dtw(ts_a, ts_b) == pytest.approx(soft_dtw(ts_b, ts_a), abs=1e-10)

    def test_matches_tslearn_default_gamma(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_soft_dtw(
            ts_a[np.newaxis], ts_b[np.newaxis]
        )[0, 0]
        assert soft_dtw(ts_a, ts_b) == pytest.approx(expected, abs=1e-10)

    def test_matches_tslearn_gamma_05(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_soft_dtw(
            ts_a[np.newaxis], ts_b[np.newaxis], gamma=0.5
        )[0, 0]
        assert soft_dtw(ts_a, ts_b, gamma=0.5) == pytest.approx(expected, abs=1e-10)

    def test_matches_tslearn_gamma_10(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_soft_dtw(
            ts_a[np.newaxis], ts_b[np.newaxis], gamma=10.0
        )[0, 0]
        assert soft_dtw(ts_a, ts_b, gamma=10.0) == pytest.approx(expected, abs=1e-10)

    def test_single_timestep(self, ts_single_step):
        other = np.array([[3.0, 5.0]])
        expected = tslearn_metrics.cdist_soft_dtw(
            ts_single_step[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert soft_dtw(ts_single_step, other) == pytest.approx(expected, abs=1e-10)

    def test_single_channel(self, ts_single_channel):
        other = np.zeros((5, 1))
        expected = tslearn_metrics.cdist_soft_dtw(
            ts_single_channel[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert soft_dtw(ts_single_channel, other) == pytest.approx(expected, abs=1e-10)

    def test_short_series(self, ts_short):
        other = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        expected = tslearn_metrics.cdist_soft_dtw(
            ts_short[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert soft_dtw(ts_short, other) == pytest.approx(expected, abs=1e-10)

    def test_returns_float(self, ts_a, ts_b):
        assert isinstance(soft_dtw(ts_a, ts_b), float)

    def test_self_alignment(self, ts_a):
        """Self-alignment gives a non-positive value (sum of zero costs through softmin)."""
        result = soft_dtw(ts_a, ts_a)
        expected = tslearn_metrics.cdist_soft_dtw(
            ts_a[np.newaxis], ts_a[np.newaxis]
        )[0, 0]
        assert result == pytest.approx(expected, abs=1e-10)


class TestCdistSoftDtw:
    """Tests for the batch cdist_soft_dtw(X, Y) function."""

    def test_output_shape(self, batch_a, batch_b):
        result = cdist_soft_dtw(batch_a, batch_b)
        assert result.shape == (3, 2)

    def test_values_match_pairwise(self, batch_a, batch_b):
        result = cdist_soft_dtw(batch_a, batch_b)
        for i in range(batch_a.shape[0]):
            for j in range(batch_b.shape[0]):
                expected = soft_dtw(batch_a[i], batch_b[j])
                assert result[i, j] == pytest.approx(expected, abs=1e-10)

    def test_matches_tslearn_cdist(self, batch_a, batch_b):
        expected = tslearn_metrics.cdist_soft_dtw(batch_a, batch_b)
        result = cdist_soft_dtw(batch_a, batch_b)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matches_tslearn_cdist_gamma(self, batch_a, batch_b):
        expected = tslearn_metrics.cdist_soft_dtw(batch_a, batch_b, gamma=2.0)
        result = cdist_soft_dtw(batch_a, batch_b, gamma=2.0)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_symmetry(self, batch_a, batch_b):
        ab = cdist_soft_dtw(batch_a, batch_b)
        ba = cdist_soft_dtw(batch_b, batch_a)
        np.testing.assert_allclose(ab, ba.T, atol=1e-10)

    def test_single_vs_batch(self, ts_a, batch_b):
        X = ts_a[np.newaxis, :, :]
        result = cdist_soft_dtw(X, batch_b)
        assert result.shape == (1, 2)
