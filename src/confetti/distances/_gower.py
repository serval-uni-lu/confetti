"""Gower distance for mixed-type tabular data."""

from __future__ import annotations

import numpy as np

try:
    from confetti._rust_core import (
        gower as _rs_gower,
        cdist_gower as _rs_cdist_gower,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _ensure_f64_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def gower(
    x: np.ndarray,
    y: np.ndarray,
    cat_mask: np.ndarray,
    ranges: np.ndarray,
) -> float:
    """
    Gower distance between two feature vectors.

    Parameters
    ----------
    ``x`` : np.ndarray
        First feature vector of shape ``(F,)``.
    ``y`` : np.ndarray
        Second feature vector of shape ``(F,)``.
    ``cat_mask`` : np.ndarray
        Boolean array of shape ``(F,)``.  ``True`` for categorical
        features (binary match/mismatch), ``False`` for numerical
        features (range-normalized absolute difference).
    ``ranges`` : np.ndarray
        Per-feature range (max - min) of shape ``(F,)``.  Only used
        for numerical features; entries for categorical features are
        ignored.

    Returns
    -------
    float
        Gower distance in [0, 1].
    """
    if _HAS_RUST:
        return _rs_gower(
            _ensure_f64_c(x),
            _ensure_f64_c(y),
            np.ascontiguousarray(cat_mask, dtype=np.bool_),
            _ensure_f64_c(ranges),
        )

    n_features = x.shape[0]
    diff = np.abs(x - y)

    d = np.empty(n_features, dtype=np.float64)
    d[cat_mask] = (x[cat_mask] != y[cat_mask]).astype(np.float64)

    num_mask = ~cat_mask
    safe_ranges = np.where(ranges > 0, ranges, 1.0)
    d[num_mask] = diff[num_mask] / safe_ranges[num_mask]
    d[num_mask & (ranges == 0)] = 0.0

    return float(np.mean(d))


def cdist_gower(
    X: np.ndarray,
    Y: np.ndarray,
    cat_mask: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    """
    Pairwise Gower distance between two sets of feature vectors.

    Parameters
    ----------
    ``X`` : np.ndarray
        First dataset of shape ``(N, F)``.
    ``Y`` : np.ndarray
        Second dataset of shape ``(M, F)``.
    ``cat_mask`` : np.ndarray
        Boolean array of shape ``(F,)``.
    ``ranges`` : np.ndarray
        Per-feature range of shape ``(F,)``.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(N, M)``.
    """
    if _HAS_RUST:
        return np.asarray(
            _rs_cdist_gower(
                _ensure_f64_c(X),
                _ensure_f64_c(Y),
                np.ascontiguousarray(cat_mask, dtype=np.bool_),
                _ensure_f64_c(ranges),
            )
        )

    n = X.shape[0]
    m = Y.shape[0]
    n_features = X.shape[1]
    result = np.empty((n, m), dtype=np.float64)

    num_mask = ~cat_mask
    safe_ranges = np.where(ranges > 0, ranges, 1.0)
    zero_range = ranges == 0

    for i in range(n):
        diff = np.abs(X[i] - Y)

        d = np.empty((m, n_features), dtype=np.float64)
        d[:, cat_mask] = (X[i, cat_mask] != Y[:, cat_mask]).astype(np.float64)
        d[:, num_mask] = diff[:, num_mask] / safe_ranges[num_mask]
        d[:, num_mask & zero_range] = 0.0

        result[i] = np.mean(d, axis=1)

    return result
