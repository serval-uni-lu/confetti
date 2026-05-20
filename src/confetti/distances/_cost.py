"""Shared cost-matrix utilities for time-series distance metrics."""

from __future__ import annotations

import numpy as np


def squared_euclidean_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distance between all pairs of time points.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C)``.

    Returns
    -------
    np.ndarray
        Cost matrix of shape ``(T1, T2)`` where entry ``(i, j)`` is
        ``||x[i] - y[j]||^2``.
    """
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)
