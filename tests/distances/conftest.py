"""Shared fixtures for the distance-metrics test suite."""

import numpy as np
import pytest


@pytest.fixture
def ts_a():
    """Deterministic (T=8, C=2) time series A."""
    return np.arange(16, dtype=float).reshape(8, 2) / 16.0


@pytest.fixture
def ts_b():
    """Deterministic (T=8, C=2) time series B, visibly different from A."""
    return np.full((8, 2), 2.0, dtype=float)


@pytest.fixture
def ts_short():
    """Short (T=3, C=2) time series for edge-case testing."""
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def ts_single_step():
    """Single-timestep (T=1, C=2) time series."""
    return np.array([[1.0, 2.0]])


@pytest.fixture
def ts_single_channel():
    """Single-channel (T=5, C=1) time series."""
    return np.arange(5, dtype=float).reshape(5, 1)


@pytest.fixture
def batch_a(ts_a, ts_short):
    """Batch of (N=3, T=8, C=2) time series built from ts_a with perturbations."""
    rng = np.random.default_rng(42)
    return np.stack([ts_a, ts_a + rng.normal(0, 0.1, ts_a.shape), ts_a * 1.5])


@pytest.fixture
def batch_b(ts_b):
    """Batch of (M=2, T=8, C=2) time series built from ts_b with perturbations."""
    rng = np.random.default_rng(99)
    return np.stack([ts_b, ts_b + rng.normal(0, 0.1, ts_b.shape)])
