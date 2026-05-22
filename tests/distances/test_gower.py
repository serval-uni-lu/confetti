"""Tests for the Gower distance metric."""

import numpy as np
import pytest

from confetti.distances._gower import cdist_gower, gower


@pytest.fixture
def mixed_x():
    """Feature vector: 2 numerical + 2 categorical features."""
    return np.array([0.2, 0.8, 1.0, 0.0])


@pytest.fixture
def mixed_y():
    """Feature vector differing from mixed_x."""
    return np.array([0.5, 0.2, 1.0, 1.0])


@pytest.fixture
def cat_mask_mixed():
    """cat_mask: last two features are categorical."""
    return np.array([False, False, True, True])


@pytest.fixture
def ranges_mixed():
    """Ranges for numerical features (indices 0 and 1)."""
    return np.array([1.0, 1.0, 0.0, 0.0])


@pytest.fixture
def batch_x():
    """Batch of 3 feature vectors, shape (3, 4)."""
    return np.array([
        [0.2, 0.8, 1.0, 0.0],
        [0.5, 0.5, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])


@pytest.fixture
def batch_y():
    """Batch of 2 feature vectors, shape (2, 4)."""
    return np.array([
        [0.5, 0.2, 1.0, 1.0],
        [0.1, 0.9, 0.0, 0.0],
    ])


class TestGowerPairwise:
    """Tests for the pairwise gower(x, y) function."""

    def test_known_values(self, mixed_x, mixed_y, cat_mask_mixed, ranges_mixed):
        # num feat 0: |0.2-0.5|/1.0 = 0.3
        # num feat 1: |0.8-0.2|/1.0 = 0.6
        # cat feat 2: 1.0 == 1.0 → 0.0
        # cat feat 3: 0.0 != 1.0 → 1.0
        # mean = (0.3 + 0.6 + 0.0 + 1.0) / 4 = 0.475
        expected = 0.475
        assert gower(mixed_x, mixed_y, cat_mask_mixed, ranges_mixed) == pytest.approx(expected)

    def test_identity(self, mixed_x, cat_mask_mixed, ranges_mixed):
        assert gower(mixed_x, mixed_x, cat_mask_mixed, ranges_mixed) == 0.0

    def test_symmetry(self, mixed_x, mixed_y, cat_mask_mixed, ranges_mixed):
        assert gower(mixed_x, mixed_y, cat_mask_mixed, ranges_mixed) == pytest.approx(
            gower(mixed_y, mixed_x, cat_mask_mixed, ranges_mixed)
        )

    def test_non_negative(self, mixed_x, mixed_y, cat_mask_mixed, ranges_mixed):
        assert gower(mixed_x, mixed_y, cat_mask_mixed, ranges_mixed) >= 0.0

    def test_returns_float(self, mixed_x, mixed_y, cat_mask_mixed, ranges_mixed):
        result = gower(mixed_x, mixed_y, cat_mask_mixed, ranges_mixed)
        assert isinstance(result, float)

    def test_all_categorical(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 1.0])
        cat_mask = np.array([True, True, True])
        ranges = np.array([0.0, 0.0, 0.0])
        # 1 mismatch out of 3
        assert gower(x, y, cat_mask, ranges) == pytest.approx(1.0 / 3.0)

    def test_all_numerical(self):
        x = np.array([0.0, 0.5])
        y = np.array([1.0, 0.0])
        cat_mask = np.array([False, False])
        ranges = np.array([2.0, 1.0])
        # feat 0: |0-1|/2 = 0.5, feat 1: |0.5-0|/1 = 0.5
        assert gower(x, y, cat_mask, ranges) == pytest.approx(0.5)

    def test_zero_range_feature(self):
        x = np.array([5.0, 0.3])
        y = np.array([5.0, 0.7])
        cat_mask = np.array([False, False])
        ranges = np.array([0.0, 1.0])
        # feat 0: range=0 → 0.0, feat 1: |0.3-0.7|/1.0 = 0.4
        assert gower(x, y, cat_mask, ranges) == pytest.approx(0.2)

    def test_maximum_distance(self):
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 1.0])
        cat_mask = np.array([True, True])
        ranges = np.array([0.0, 0.0])
        assert gower(x, y, cat_mask, ranges) == pytest.approx(1.0)

    def test_binary_features(self):
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([1.0, 1.0, 1.0])
        cat_mask = np.array([True, True, True])
        ranges = np.array([0.0, 0.0, 0.0])
        # 2 mismatches out of 3
        assert gower(x, y, cat_mask, ranges) == pytest.approx(2.0 / 3.0)


class TestCdistGower:
    """Tests for the batch cdist_gower(X, Y) function."""

    def test_output_shape(self, batch_x, batch_y, cat_mask_mixed, ranges_mixed):
        result = cdist_gower(batch_x, batch_y, cat_mask_mixed, ranges_mixed)
        assert result.shape == (3, 2)

    def test_values_match_pairwise(self, batch_x, batch_y, cat_mask_mixed, ranges_mixed):
        result = cdist_gower(batch_x, batch_y, cat_mask_mixed, ranges_mixed)
        for i in range(batch_x.shape[0]):
            for j in range(batch_y.shape[0]):
                expected = gower(batch_x[i], batch_y[j], cat_mask_mixed, ranges_mixed)
                assert result[i, j] == pytest.approx(expected)

    def test_self_distance_diagonal(self, batch_x, cat_mask_mixed, ranges_mixed):
        result = cdist_gower(batch_x, batch_x, cat_mask_mixed, ranges_mixed)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-15)

    def test_symmetry(self, batch_x, batch_y, cat_mask_mixed, ranges_mixed):
        ab = cdist_gower(batch_x, batch_y, cat_mask_mixed, ranges_mixed)
        ba = cdist_gower(batch_y, batch_x, cat_mask_mixed, ranges_mixed)
        np.testing.assert_allclose(ab, ba.T, atol=1e-15)

    def test_non_negative(self, batch_x, batch_y, cat_mask_mixed, ranges_mixed):
        result = cdist_gower(batch_x, batch_y, cat_mask_mixed, ranges_mixed)
        assert np.all(result >= 0.0)
