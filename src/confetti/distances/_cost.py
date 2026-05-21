"""Shared cost-matrix utilities for time-series distance metrics."""

from __future__ import annotations

import numpy as np

try:
    from confetti._rust_core import squared_euclidean_cost_matrix as _rs_cost_matrix

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
        return _rs_cost_matrix(
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
        )

except ImportError:

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
