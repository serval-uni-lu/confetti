"""Tests for the Dynamic Time Warping distance metric."""

import numpy as np
import pytest

from confetti.distances._dtw import cdist_dtw, dtw

tslearn_metrics = pytest.importorskip("tslearn.metrics")


class TestDtwPairwise:
    """Tests for the pairwise dtw(x, y) function."""

    def test_identity(self, ts_a):
        assert dtw(ts_a, ts_a) == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self, ts_a, ts_b):
        assert dtw(ts_a, ts_b) == pytest.approx(dtw(ts_b, ts_a))

    def test_non_negative(self, ts_a, ts_b):
        assert dtw(ts_a, ts_b) >= 0.0

    def test_matches_tslearn(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_dtw(
            ts_a[np.newaxis], ts_b[np.newaxis]
        )[0, 0]
        assert dtw(ts_a, ts_b) == pytest.approx(expected, abs=1e-10)

    def test_matches_tslearn_with_sakoe_chiba(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_dtw(
            ts_a[np.newaxis],
            ts_b[np.newaxis],
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=2,
        )[0, 0]
        result = dtw(ts_a, ts_b, sakoe_chiba_radius=2)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_single_timestep(self, ts_single_step):
        other = np.array([[3.0, 5.0]])
        expected = np.sqrt((1.0 - 3.0) ** 2 + (2.0 - 5.0) ** 2)
        assert dtw(ts_single_step, other) == pytest.approx(expected, abs=1e-10)

    def test_single_channel(self, ts_single_channel):
        other = np.zeros((5, 1))
        expected = tslearn_metrics.cdist_dtw(
            ts_single_channel[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert dtw(ts_single_channel, other) == pytest.approx(expected, abs=1e-10)

    def test_short_series(self, ts_short):
        other = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        expected = tslearn_metrics.cdist_dtw(
            ts_short[np.newaxis], other[np.newaxis]
        )[0, 0]
        assert dtw(ts_short, other) == pytest.approx(expected, abs=1e-10)

    def test_returns_float(self, ts_a, ts_b):
        assert isinstance(dtw(ts_a, ts_b), float)

    def test_sakoe_chiba_large_radius_matches_unconstrained(self, ts_a, ts_b):
        """A radius >= T-1 should give the same result as no constraint."""
        T = ts_a.shape[0]
        constrained = dtw(ts_a, ts_b, sakoe_chiba_radius=T)
        unconstrained = dtw(ts_a, ts_b)
        assert constrained == pytest.approx(unconstrained, abs=1e-12)

    def test_sakoe_chiba_radius_one(self, ts_a, ts_b):
        expected = tslearn_metrics.cdist_dtw(
            ts_a[np.newaxis],
            ts_b[np.newaxis],
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=1,
        )[0, 0]
        result = dtw(ts_a, ts_b, sakoe_chiba_radius=1)
        assert result == pytest.approx(expected, abs=1e-10)


class TestCdistDtw:
    """Tests for the batch cdist_dtw(X, Y) function."""

    def test_output_shape(self, batch_a, batch_b):
        result = cdist_dtw(batch_a, batch_b)
        assert result.shape == (3, 2)

    def test_values_match_pairwise(self, batch_a, batch_b):
        result = cdist_dtw(batch_a, batch_b)
        for i in range(batch_a.shape[0]):
            for j in range(batch_b.shape[0]):
                expected = dtw(batch_a[i], batch_b[j])
                assert result[i, j] == pytest.approx(expected)

    def test_matches_tslearn_cdist(self, batch_a, batch_b):
        expected = tslearn_metrics.cdist_dtw(batch_a, batch_b)
        result = cdist_dtw(batch_a, batch_b)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matches_tslearn_cdist_with_sakoe_chiba(self, batch_a, batch_b):
        expected = tslearn_metrics.cdist_dtw(
            batch_a, batch_b,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=3,
        )
        result = cdist_dtw(
            batch_a, batch_b,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=3,
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_self_distance_diagonal(self, batch_a):
        result = cdist_dtw(batch_a, batch_a)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-12)

    def test_symmetry(self, batch_a, batch_b):
        ab = cdist_dtw(batch_a, batch_b)
        ba = cdist_dtw(batch_b, batch_a)
        np.testing.assert_allclose(ab, ba.T, atol=1e-12)

    def test_single_vs_batch(self, ts_a, batch_b):
        X = ts_a[np.newaxis, :, :]
        result = cdist_dtw(X, batch_b)
        assert result.shape == (1, 2)
