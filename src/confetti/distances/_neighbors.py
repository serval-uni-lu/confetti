"""KNN for time series, replacing tslearn.neighbors.KNeighborsTimeSeries."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from confetti.distances._dtw import cdist_dtw
from confetti.distances._registry import get_cdist_function


class TimeSeriesKNN:
    """
    K-nearest-neighbors search for multivariate time series.

    Drop-in replacement for ``tslearn.neighbors.KNeighborsTimeSeries``.
    For Euclidean distance the search is done via sklearn on flattened
    vectors.  For all other metrics a precomputed distance matrix is
    built with the corresponding ``cdist_*`` function.

    Parameters
    ----------
    ``n_neighbors`` : int, default=1
        Number of neighbours to return.
    ``metric`` : str, default="euclidean"
        Distance metric name (``"euclidean"``, ``"dtw"``, etc.).
    ``metric_params`` : dict or None, default=None
        Extra keyword arguments forwarded to the underlying ``cdist_*``
        function (e.g. ``{"global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius": 2}``).
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = "euclidean",
        metric_params: dict[str, Any] | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_params = metric_params or {}
        self._train_data: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> TimeSeriesKNN:
        """
        Store the training dataset.

        Parameters
        ----------
        ``X`` : np.ndarray
            Training time series of shape ``(N, T, C)``.

        Returns
        -------
        TimeSeriesKNN
            self
        """
        self._train_data = np.asarray(X, dtype=np.float64)
        return self

    def kneighbors(
        self,
        X: np.ndarray,
        return_distance: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Find the k nearest neighbors for each query.

        Parameters
        ----------
        ``X`` : np.ndarray
            Query time series of shape ``(Q, T, C)``.
        ``return_distance`` : bool, default=True
            If ``True``, return ``(distances, indices)``; otherwise
            return only ``indices``.

        Returns
        -------
        distances : np.ndarray
            Shape ``(Q, n_neighbors)``.  Only returned when
            ``return_distance`` is ``True``.
        indices : np.ndarray
            Shape ``(Q, n_neighbors)``.
        """
        if self._train_data is None:
            raise RuntimeError("Call fit() before kneighbors().")

        X = np.asarray(X, dtype=np.float64)

        if self.metric == "euclidean":
            return self._kneighbors_euclidean(X, return_distance)

        return self._kneighbors_precomputed(X, return_distance)

    def _kneighbors_euclidean(
        self,
        X: np.ndarray,
        return_distance: bool,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        assert self._train_data is not None
        N, T, C = self._train_data.shape
        train_flat = self._train_data.reshape(N, T * C)
        query_flat = X.reshape(X.shape[0], T * C)

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        nn.fit(train_flat)
        return nn.kneighbors(query_flat, return_distance=return_distance)

    def _kneighbors_precomputed(
        self,
        X: np.ndarray,
        return_distance: bool,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        assert self._train_data is not None

        if self.metric == "dtw":
            dist_matrix = cdist_dtw(X, self._train_data, **self.metric_params)
        else:
            cdist_fn = get_cdist_function(self.metric)
            dist_matrix = cdist_fn(X, self._train_data, **self.metric_params)

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="precomputed")
        nn.fit(np.zeros((self._train_data.shape[0], self._train_data.shape[0])))

        if return_distance:
            indices = np.argsort(dist_matrix, axis=1)[:, : self.n_neighbors]
            distances = np.take_along_axis(dist_matrix, indices, axis=1)
            return distances, indices
        else:
            indices = np.argsort(dist_matrix, axis=1)[:, : self.n_neighbors]
            return indices
