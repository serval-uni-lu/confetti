"""Metric name to cdist function dispatch."""

from __future__ import annotations

from typing import Any, Callable

from confetti.distances._ctw import cdist_ctw
from confetti.distances._dtw import cdist_dtw
from confetti.distances._gak import cdist_gak
from confetti.distances._manhattan import cdist_manhattan
from confetti.distances._soft_dtw import cdist_soft_dtw
from confetti.errors import CONFETTIConfigurationError

_CDIST_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "dtw": cdist_dtw,
    "ctw": cdist_ctw,
    "softdtw": cdist_soft_dtw,
    "gak": cdist_gak,
    "manhattan": cdist_manhattan,
}


def get_cdist_function(metric_name: str) -> Callable[..., Any]:
    """
    Look up a pairwise distance function by name.

    Parameters
    ----------
    ``metric_name`` : str
        One of ``"dtw"``, ``"ctw"``, ``"softdtw"``, ``"gak"``, or
        ``"manhattan"`` (case-insensitive).

    Returns
    -------
    Callable
        The corresponding ``cdist_*`` function.

    Raises
    ------
    CONFETTIConfigurationError
        If the metric name is not recognized.
    """
    fn = _CDIST_FUNCTIONS.get(metric_name.lower())
    if fn is not None:
        return fn
    raise CONFETTIConfigurationError(
        message="The specified proximity distance metric is not supported.",
        param="proximity_distance",
        config={"proximity_distance": metric_name},
        hint="Choose from 'manhattan', 'dtw', 'ctw', 'softdtw', or 'gak'.",
    )
