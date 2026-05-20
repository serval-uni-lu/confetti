"""
Tests for the Das-Dennis reference-directions sampler used by NSGA-III.

These lock in the deterministic simplex values and cardinality formula as a
specification for a future hand-rolled re-implementation. Das-Dennis is purely
combinatorial — no RNG is involved — so *exact* numeric comparisons are stable.
"""

import math

import numpy as np
import pytest

from pymoo.util.ref_dirs import get_reference_directions


@pytest.mark.parametrize(
    "n_dim, n_partitions",
    [(2, 12), (3, 3), (3, 12), (4, 5), (5, 4)],
)
def test_das_dennis_shape_matches_formula(n_dim, n_partitions):
    """The number of Das-Dennis points is C(n_dim + n_partitions - 1, n_partitions)."""
    rd = get_reference_directions("das-dennis", n_dim, n_partitions=n_partitions)
    expected_count = math.comb(n_dim + n_partitions - 1, n_partitions)
    assert rd.shape == (expected_count, n_dim)


@pytest.mark.parametrize(
    "n_dim, n_partitions",
    [(2, 12), (3, 3), (4, 5)],
)
def test_das_dennis_rows_sum_to_one(n_dim, n_partitions):
    rd = get_reference_directions("das-dennis", n_dim, n_partitions=n_partitions)
    sums = rd.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-9)


@pytest.mark.parametrize(
    "n_dim, n_partitions",
    [(2, 12), (3, 3), (4, 5)],
)
def test_das_dennis_non_negative(n_dim, n_partitions):
    rd = get_reference_directions("das-dennis", n_dim, n_partitions=n_partitions)
    assert (rd >= 0.0).all()


def test_das_dennis_2d_p12_exact_values():
    """Exact lexicographic layout for (n_dim=2, n_partitions=12)."""
    rd = get_reference_directions("das-dennis", 2, n_partitions=12)
    expected = np.array(
        [[k / 12.0, 1.0 - k / 12.0] for k in range(13)],
        dtype=float,
    )
    np.testing.assert_allclose(rd, expected, atol=1e-12)


def test_das_dennis_3d_p3_exact_values():
    """The 10 canonical Das-Dennis points on the 3D simplex at thirds."""
    rd = get_reference_directions("das-dennis", 3, n_partitions=3)
    third = 1.0 / 3.0
    two_thirds = 2.0 / 3.0
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, third, two_thirds],
            [0.0, two_thirds, third],
            [0.0, 1.0, 0.0],
            [third, 0.0, two_thirds],
            [third, third, third],
            [third, two_thirds, 0.0],
            [two_thirds, 0.0, third],
            [two_thirds, third, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(rd, expected, atol=1e-12)


def test_das_dennis_determinism():
    """Two calls with the same args return identical arrays (no RNG)."""
    a = get_reference_directions("das-dennis", 3, n_partitions=6)
    b = get_reference_directions("das-dennis", 3, n_partitions=6)
    np.testing.assert_array_equal(a, b)


def test_das_dennis_1d_1partition_edge():
    """Minimal edge case: ``n_dim=1, n_partitions=1`` → single point ``[[1.0]]``."""
    rd = get_reference_directions("das-dennis", 1, n_partitions=1)
    assert rd.shape == (1, 1)
    np.testing.assert_allclose(rd, np.array([[1.0]]), atol=1e-12)
