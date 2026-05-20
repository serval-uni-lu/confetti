"""Manhattan (L1) distance for multivariate time series."""

from __future__ import annotations

import numpy as np


def manhattan(x: np.ndarray, y: np.ndarray) -> float:
    """
    Manhattan distance between two time series.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T, C)``.

    Returns
    -------
    float
        Sum of element-wise absolute differences.
    """
    return float(np.sum(np.abs(x - y)))


def cdist_manhattan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Pairwise Manhattan distance between two sets of time series.

    Parameters
    ----------
    ``X`` : np.ndarray
        First dataset of shape ``(N, T, C)``.
    ``Y`` : np.ndarray
        Second dataset of shape ``(M, T, C)``.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(N, M)``.
    """
    diff = X[:, np.newaxis, :, :] - Y[np.newaxis, :, :, :]
    return np.sum(np.abs(diff), axis=(2, 3))
