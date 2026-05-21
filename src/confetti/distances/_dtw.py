"""Dynamic Time Warping distance for multivariate time series."""

from __future__ import annotations

import math

import numpy as np

from confetti.distances._cost import squared_euclidean_cost_matrix

try:
    from confetti._rust_core import (
        dtw as _rs_dtw,
        dtw_with_path_py as _rs_dtw_with_path,
        cdist_dtw as _rs_cdist_dtw,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _ensure_f64_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _sakoe_chiba_mask(T1: int, T2: int, radius: int) -> np.ndarray:
    """
    Boolean mask for the Sakoe-Chiba band constraint.

    Parameters
    ----------
    ``T1`` : int
        Length of the first time series.
    ``T2`` : int
        Length of the second time series.
    ``radius`` : int
        Band radius.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(T1, T2)`` where ``True`` means the cell
        is **allowed**.
    """
    i_idx = np.arange(T1)[:, np.newaxis]
    j_idx = np.arange(T2)[np.newaxis, :]
    diag_shift = (T2 - 1) * i_idx / max(T1 - 1, 1)
    return np.abs(j_idx - diag_shift) <= radius


def _dtw_accumulated_cost(
    C: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the DTW accumulated cost matrix.

    Parameters
    ----------
    ``C`` : np.ndarray
        Squared Euclidean cost matrix of shape ``(T1, T2)``.
    ``mask`` : np.ndarray or None
        Boolean mask of shape ``(T1, T2)``.  Cells where mask is
        ``False`` are set to infinity.

    Returns
    -------
    np.ndarray
        Accumulated cost matrix of shape ``(T1 + 1, T2 + 1)``.
    """
    T1, T2 = C.shape
    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0.0

    if mask is not None:
        C = np.where(mask, C, np.inf)

    for i in range(T1):
        for j in range(T2):
            D[i + 1, j + 1] = C[i, j] + min(D[i, j + 1], D[i + 1, j], D[i, j])

    return D


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sakoe_chiba_radius: int | None = None,
) -> float:
    """
    Dynamic Time Warping distance between two time series.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C)``.
    ``sakoe_chiba_radius`` : int or None, default=None
        Sakoe-Chiba band radius.  If ``None``, no constraint is applied.

    Returns
    -------
    float
        DTW distance (square root of the accumulated squared Euclidean cost).
    """
    if _HAS_RUST:
        return _rs_dtw(_ensure_f64_c(x), _ensure_f64_c(y), sakoe_chiba_radius)

    C = squared_euclidean_cost_matrix(x, y)
    mask = None
    if sakoe_chiba_radius is not None:
        mask = _sakoe_chiba_mask(C.shape[0], C.shape[1], sakoe_chiba_radius)
    D = _dtw_accumulated_cost(C, mask)
    return math.sqrt(D[-1, -1])


def _dtw_with_path(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sakoe_chiba_radius: int | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """
    DTW distance together with the optimal warping path.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C)``.
    ``sakoe_chiba_radius`` : int or None, default=None
        Sakoe-Chiba band radius.

    Returns
    -------
    distance : float
        DTW distance.
    path : list[tuple[int, int]]
        Warping path as a list of ``(i, j)`` index pairs, from ``(0, 0)``
        to ``(T1-1, T2-1)``.
    """
    if _HAS_RUST:
        dist, path = _rs_dtw_with_path(_ensure_f64_c(x), _ensure_f64_c(y), sakoe_chiba_radius)
        return dist, [(int(i), int(j)) for i, j in path]

    C = squared_euclidean_cost_matrix(x, y)
    mask = None
    if sakoe_chiba_radius is not None:
        mask = _sakoe_chiba_mask(C.shape[0], C.shape[1], sakoe_chiba_radius)
    D = _dtw_accumulated_cost(C, mask)

    T1, T2 = C.shape
    i, j = T1, T2
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (D[i - 1, j - 1], i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda t: t[0])
    path.reverse()
    return math.sqrt(D[-1, -1]), path


def cdist_dtw(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    global_constraint: str | None = None,
    sakoe_chiba_radius: int | None = None,
) -> np.ndarray:
    """
    Pairwise DTW distance between two sets of time series.

    Parameters
    ----------
    ``X`` : np.ndarray
        First dataset of shape ``(N, T1, C)``.
    ``Y`` : np.ndarray
        Second dataset of shape ``(M, T2, C)``.
    ``global_constraint`` : str or None, default=None
        If ``"sakoe_chiba"``, applies the Sakoe-Chiba band constraint.
    ``sakoe_chiba_radius`` : int or None, default=None
        Radius for the Sakoe-Chiba band.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(N, M)``.
    """
    radius = None
    if global_constraint == "sakoe_chiba" and sakoe_chiba_radius is not None:
        radius = int(sakoe_chiba_radius)

    if _HAS_RUST:
        return np.asarray(_rs_cdist_dtw(_ensure_f64_c(X), _ensure_f64_c(Y), radius))

    N = X.shape[0]
    M = Y.shape[0]
    result = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            result[i, j] = dtw(X[i], Y[j], sakoe_chiba_radius=radius)
    return result
