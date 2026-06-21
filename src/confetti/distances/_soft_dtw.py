"""Soft-DTW distance for multivariate time series."""

from __future__ import annotations

import numpy as np

from confetti.distances._cost import squared_euclidean_cost_matrix

try:
    from confetti._rust_core import (
        soft_dtw as _rs_soft_dtw,
        cdist_soft_dtw as _rs_cdist_soft_dtw,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _ensure_f64_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    """
    Numerically stable soft-minimum of three values.

    Uses the log-sum-exp trick to avoid overflow/underflow.
    """
    min_val = min(a, b, c)
    exp_sum = np.exp((min_val - a) / gamma) + np.exp((min_val - b) / gamma) + np.exp((min_val - c) / gamma)
    return min_val - gamma * np.log(exp_sum)


def soft_dtw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float = 1.0,
) -> float:
    """
    Soft-DTW similarity between two time series.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C)``.
    ``gamma`` : float, default=1.0
        Smoothing parameter.  Smaller values approximate hard DTW.

    Returns
    -------
    float
        Soft-DTW value.  Unlike standard DTW this is a similarity score
        that can be negative.
    """
    if _HAS_RUST:
        return _rs_soft_dtw(_ensure_f64_c(x), _ensure_f64_c(y), gamma)

    C = squared_euclidean_cost_matrix(x, y)
    T1, T2 = C.shape

    R = np.full((T1 + 1, T2 + 1), np.inf)
    R[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            R[i, j] = C[i - 1, j - 1] + _softmin3(R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma)

    return float(R[T1, T2])


def cdist_soft_dtw(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Pairwise Soft-DTW between two sets of time series.

    Parameters
    ----------
    ``X`` : np.ndarray
        First dataset of shape ``(N, T, C)``.
    ``Y`` : np.ndarray
        Second dataset of shape ``(M, T, C)``.
    ``gamma`` : float, default=1.0
        Smoothing parameter.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape ``(N, M)``.
    """
    if _HAS_RUST:
        return np.asarray(_rs_cdist_soft_dtw(_ensure_f64_c(X), _ensure_f64_c(Y), gamma))

    N = X.shape[0]
    M = Y.shape[0]
    result = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            result[i, j] = soft_dtw(X[i], Y[j], gamma=gamma)
    return result
