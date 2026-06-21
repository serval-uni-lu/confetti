"""Tests for the Manhattan (L1) distance metric."""

import numpy as np
import pytest

from confetti.distances._manhattan import cdist_manhattan, manhattan


class TestManhattanPairwise:
    """Tests for the pairwise manhattan(x, y) function."""

    def test_known_values(self, ts_a, ts_b):
        expected = float(np.sum(np.abs(ts_a - ts_b)))
        assert manhattan(ts_a, ts_b) == pytest.approx(expected)

    def test_identity(self, ts_a):
        assert manhattan(ts_a, ts_a) == 0.0

    def test_symmetry(self, ts_a, ts_b):
        assert manhattan(ts_a, ts_b) == pytest.approx(manhattan(ts_b, ts_a))

    def test_non_negative(self, ts_a, ts_b):
        assert manhattan(ts_a, ts_b) >= 0.0

    def test_single_timestep(self, ts_single_step):
        other = np.array([[3.0, 5.0]])
        assert manhattan(ts_single_step, other) == pytest.approx(abs(1.0 - 3.0) + abs(2.0 - 5.0))

    def test_single_channel(self, ts_single_channel):
        other = np.zeros((5, 1))
        expected = float(np.sum(np.abs(ts_single_channel)))
        assert manhattan(ts_single_channel, other) == pytest.approx(expected)

    def test_returns_float(self, ts_a, ts_b):
        result = manhattan(ts_a, ts_b)
        assert isinstance(result, float)


class TestCdistManhattan:
    """Tests for the batch cdist_manhattan(X, Y) function."""

    def test_output_shape(self, batch_a, batch_b):
        result = cdist_manhattan(batch_a, batch_b)
        assert result.shape == (3, 2)

    def test_values_match_pairwise(self, batch_a, batch_b):
        result = cdist_manhattan(batch_a, batch_b)
        for i in range(batch_a.shape[0]):
            for j in range(batch_b.shape[0]):
                expected = manhattan(batch_a[i], batch_b[j])
                assert result[i, j] == pytest.approx(expected)

    def test_self_distance_diagonal(self, batch_a):
        result = cdist_manhattan(batch_a, batch_a)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-15)

    def test_single_vs_batch(self, ts_a, batch_b):
        X = ts_a[np.newaxis, :, :]
        result = cdist_manhattan(X, batch_b)
        assert result.shape == (1, 2)

    def test_symmetry(self, batch_a, batch_b):
        ab = cdist_manhattan(batch_a, batch_b)
        ba = cdist_manhattan(batch_b, batch_a)
        np.testing.assert_allclose(ab, ba.T, atol=1e-15)

    def test_non_negative(self, batch_a, batch_b):
        result = cdist_manhattan(batch_a, batch_b)
        assert np.all(result >= 0.0)
