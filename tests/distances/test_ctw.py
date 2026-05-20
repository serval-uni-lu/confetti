"""Tests for the Canonical Time Warping distance metric.

tslearn's CTW implementation has a convergence bug
(``np.array_equal(path, path)`` always True), so these tests validate
correctness properties rather than matching tslearn output.
"""

import numpy as np
import pytest

from confetti.distances._ctw import cdist_ctw, ctw
from confetti.distances._dtw import dtw


class TestCtwPairwise:
    """Tests for the pairwise ctw(x, y) function."""

    def test_identity(self, ts_a):
        assert ctw(ts_a, ts_a) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self, ts_a, ts_b):
        assert ctw(ts_a, ts_b) == pytest.approx(ctw(ts_b, ts_a), abs=1e-6)

    def test_non_negative(self, ts_a, ts_b):
        assert ctw(ts_a, ts_b) >= 0.0

    def test_at_most_dtw(self, ts_a, ts_b):
        """CTW should find a distance <= plain DTW since CCA can only improve alignment."""
        ctw_dist = ctw(ts_a, ts_b)
        dtw_dist = dtw(ts_a, ts_b)
        assert ctw_dist <= dtw_dist + 1e-10

    def test_single_timestep(self, ts_single_step):
        other = np.array([[3.0, 5.0]])
        result = ctw(ts_single_step, other)
        expected_dtw = dtw(ts_single_step, other)
        assert result == pytest.approx(expected_dtw, abs=1e-10)

    def test_single_channel(self, ts_single_channel):
        other = np.zeros((5, 1))
        result = ctw(ts_single_channel, other)
        assert result >= 0.0

    def test_short_series(self, ts_short):
        other = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        result = ctw(ts_short, other)
        assert result >= 0.0
        assert result <= dtw(ts_short, other) + 1e-10

    def test_returns_float(self, ts_a, ts_b):
        assert isinstance(ctw(ts_a, ts_b), float)

    def test_max_iter_one_equals_dtw(self, ts_a, ts_b):
        """With max_iter=1 there is no CCA step, so CTW == DTW on identity-projected inputs."""
        result = ctw(ts_a, ts_b, max_iter=1)
        expected = dtw(ts_a, ts_b)
        assert result == pytest.approx(expected, abs=1e-10)


class TestCdistCtw:
    """Tests for the batch cdist_ctw(X, Y) function."""

    def test_output_shape(self, batch_a, batch_b):
        result = cdist_ctw(batch_a, batch_b)
        assert result.shape == (3, 2)

    def test_values_match_pairwise(self, batch_a, batch_b):
        result = cdist_ctw(batch_a, batch_b)
        for i in range(batch_a.shape[0]):
            for j in range(batch_b.shape[0]):
                expected = ctw(batch_a[i], batch_b[j])
                assert result[i, j] == pytest.approx(expected, abs=1e-10)

    def test_self_distance_diagonal(self, batch_a):
        result = cdist_ctw(batch_a, batch_a)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-10)

    def test_symmetry(self, batch_a, batch_b):
        """CTW symmetry is approximate because CCA convergence is order-dependent."""
        ab = cdist_ctw(batch_a, batch_b)
        ba = cdist_ctw(batch_b, batch_a)
        np.testing.assert_allclose(ab, ba.T, atol=0.2)

    def test_single_vs_batch(self, ts_a, batch_b):
        X = ts_a[np.newaxis, :, :]
        result = cdist_ctw(X, batch_b)
        assert result.shape == (1, 2)

    def test_non_negative(self, batch_a, batch_b):
        result = cdist_ctw(batch_a, batch_b)
        assert np.all(result >= -1e-10)
