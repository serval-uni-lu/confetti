"""Manhattan (L1) distance for multivariate time series."""

from __future__ import annotations

import numpy as np

try:
    from confetti._rust_core import (
        manhattan as _rs_manhattan,
        cdist_manhattan as _rs_cdist_manhattan,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _ensure_f64_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


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
    if _HAS_RUST:
        return _rs_manhattan(_ensure_f64_c(x), _ensure_f64_c(y))

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
    if _HAS_RUST:
        return np.asarray(_rs_cdist_manhattan(_ensure_f64_c(X), _ensure_f64_c(Y)))

    diff = X[:, np.newaxis, :, :] - Y[np.newaxis, :, :, :]
    return np.sum(np.abs(diff), axis=(2, 3))
