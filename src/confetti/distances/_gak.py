"""Global Alignment Kernel (GAK) for multivariate time series."""

from __future__ import annotations

import math

import numpy as np

from confetti.distances._cost import squared_euclidean_cost_matrix

try:
    from confetti._rust_core import (
        gak as _rs_gak,
        unnormalized_gak_py as _rs_unnormalized_gak,
        cdist_gak as _rs_cdist_gak,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _ensure_f64_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _gak_gram_matrix(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the GAK Gram matrix between two time series.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C)``.
    ``sigma`` : float
        Bandwidth of the internal Gaussian kernel.

    Returns
    -------
    np.ndarray
        Gram matrix of shape ``(T1, T2)``.
    """
    gram = -squared_euclidean_cost_matrix(x, y) / (2.0 * sigma ** 2)
    gram -= np.log(2.0 - np.exp(gram))
    return np.exp(gram)


def _unnormalized_gak(x: np.ndarray, y: np.ndarray, *, sigma: float = 1.0) -> float:
    """
    Unnormalized Global Alignment Kernel value.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C)``.
    ``sigma`` : float, default=1.0
        Bandwidth of the internal Gaussian kernel.

    Returns
    -------
    float
        Unnormalized kernel value.
    """
    if _HAS_RUST:
        return _rs_unnormalized_gak(_ensure_f64_c(x), _ensure_f64_c(y), sigma)

    gram = _gak_gram_matrix(x, y, sigma)
    T1, T2 = gram.shape

    cum_sum = np.zeros((T1 + 1, T2 + 1))
    cum_sum[0, 0] = 1.0

    for i in range(T1):
        for j in range(T2):
            cum_sum[i + 1, j + 1] = (
                cum_sum[i, j + 1] + cum_sum[i + 1, j] + cum_sum[i, j]
            ) * gram[i, j]

    return float(cum_sum[T1, T2])


def gak(x: np.ndarray, y: np.ndarray, *, sigma: float = 1.0) -> float:
    """
    Normalized Global Alignment Kernel between two time series.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C)``.
    ``sigma`` : float, default=1.0
        Bandwidth of the internal Gaussian kernel.

    Returns
    -------
    float
        Normalized kernel value in ``[0, 1]``.  A value of 1 means the
        two series are identical.
    """
    if _HAS_RUST:
        return _rs_gak(_ensure_f64_c(x), _ensure_f64_c(y), sigma)

    k_xy = _unnormalized_gak(x, y, sigma=sigma)
    k_xx = _unnormalized_gak(x, x, sigma=sigma)
    k_yy = _unnormalized_gak(y, y, sigma=sigma)
    return float(k_xy / math.sqrt(k_xx * k_yy))


def cdist_gak(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Pairwise normalized GAK between two sets of time series.

    Parameters
    ----------
    ``X`` : np.ndarray
        First dataset of shape ``(N, T1, C)``.
    ``Y`` : np.ndarray
        Second dataset of shape ``(M, T2, C)``.
    ``sigma`` : float, default=1.0
        Bandwidth of the internal Gaussian kernel.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape ``(N, M)`` with values in ``[0, 1]``.
    """
    if _HAS_RUST:
        return np.asarray(_rs_cdist_gak(_ensure_f64_c(X), _ensure_f64_c(Y), sigma))

    N = X.shape[0]
    M = Y.shape[0]

    k_xx = np.array([_unnormalized_gak(X[i], X[i], sigma=sigma) for i in range(N)])
    k_yy = np.array([_unnormalized_gak(Y[j], Y[j], sigma=sigma) for j in range(M)])

    result = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            k_xy = _unnormalized_gak(X[i], Y[j], sigma=sigma)
            result[i, j] = k_xy / math.sqrt(k_xx[i] * k_yy[j])
    return result
