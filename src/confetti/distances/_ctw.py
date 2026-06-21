"""Canonical Time Warping distance for multivariate time series."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cross_decomposition import CCA

from confetti.distances._dtw import _dtw_with_path


def _get_warp_matrices(
    path: list[tuple[int, int]],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a warping path into binary indicator matrices.

    Parameters
    ----------
    ``path`` : list[tuple[int, int]]
        Warping path as ``(i, j)`` index pairs.
    ``nx`` : int
        Length of the first time series.
    ``ny`` : int
        Length of the second time series.

    Returns
    -------
    Wx : np.ndarray
        Indicator matrix of shape ``(m, nx)``.
    Wy : np.ndarray
        Indicator matrix of shape ``(m, ny)``.
    """
    m = len(path)
    Wx = np.zeros((m, nx), dtype=np.float64)
    Wy = np.zeros((m, ny), dtype=np.float64)
    for k, (i, j) in enumerate(path):
        Wx[k, i] = 1.0
        Wy[k, j] = 1.0
    return Wx, Wy


def ctw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 100,
    n_components: int | None = None,
    sakoe_chiba_radius: int | None = None,
) -> float:
    """
    Canonical Time Warping distance between two time series.

    Aligns the feature spaces of two time series via Canonical Correlation
    Analysis and then computes DTW in the shared canonical space.

    Parameters
    ----------
    ``x`` : np.ndarray
        First time series of shape ``(T1, C1)``.
    ``y`` : np.ndarray
        Second time series of shape ``(T2, C2)``.
    ``max_iter`` : int, default=100
        Maximum number of CTW iterations.
    ``n_components`` : int or None, default=None
        Number of CCA components.  Defaults to ``min(C1, C2)``.
    ``sakoe_chiba_radius`` : int or None, default=None
        Sakoe-Chiba band radius for the internal DTW calls.

    Returns
    -------
    float
        CTW distance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if n_components is None:
        n_components = min(x.shape[1], y.shape[1])

    x_proj = x @ np.eye(x.shape[1], n_components, dtype=np.float64)
    y_proj = y @ np.eye(y.shape[1], n_components, dtype=np.float64)

    current_score, current_path = _dtw_with_path(
        x_proj,
        y_proj,
        sakoe_chiba_radius=sakoe_chiba_radius,
    )

    if max_iter <= 1 or len(current_path) < 2:
        return current_score

    cca = CCA(n_components=n_components)
    for _ in range(max_iter - 1):
        Wx, Wy = _get_warp_matrices(current_path, x.shape[0], y.shape[0])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cca.fit(Wx @ x, Wy @ y)
                x_tr, y_tr = cca.transform(x, y)
        except (ValueError, np.linalg.LinAlgError):
            break

        if np.any(np.isnan(x_tr)) or np.any(np.isnan(y_tr)):
            break

        new_score, new_path = _dtw_with_path(
            x_tr,
            y_tr,
            sakoe_chiba_radius=sakoe_chiba_radius,
        )

        if new_path == current_path:
            break

        current_path = new_path
        current_score = new_score

    return current_score


def cdist_ctw(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    max_iter: int = 100,
    n_components: int | None = None,
    global_constraint: str | None = None,
    sakoe_chiba_radius: int | None = None,
) -> np.ndarray:
    """
    Pairwise CTW distance between two sets of time series.

    Parameters
    ----------
    ``X`` : np.ndarray
        First dataset of shape ``(N, T1, C1)``.
    ``Y`` : np.ndarray
        Second dataset of shape ``(M, T2, C2)``.
    ``max_iter`` : int, default=100
        Maximum number of CTW iterations per pair.
    ``n_components`` : int or None, default=None
        Number of CCA components.
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

    N = X.shape[0]
    M = Y.shape[0]
    result = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            result[i, j] = ctw(
                X[i],
                Y[j],
                max_iter=max_iter,
                n_components=n_components,
                sakoe_chiba_radius=radius,
            )
    return result
