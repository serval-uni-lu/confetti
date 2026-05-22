"""
Pure-numpy distance metrics for multivariate time series.

This package replaces tslearn's distance metric implementations with
hand-rolled versions whose only external dependency is scikit-learn
(for CCA in CTW and NearestNeighbors in TimeSeriesKNN).
"""

from confetti.distances._ctw import cdist_ctw, ctw
from confetti.distances._dtw import cdist_dtw, dtw
from confetti.distances._gak import cdist_gak, gak
from confetti.distances._gower import cdist_gower, gower
from confetti.distances._manhattan import cdist_manhattan, manhattan
from confetti.distances._neighbors import TimeSeriesKNN
from confetti.distances._registry import get_cdist_function
from confetti.distances._soft_dtw import cdist_soft_dtw, soft_dtw

__all__ = [
    "TimeSeriesKNN",
    "cdist_ctw",
    "cdist_dtw",
    "cdist_gak",
    "cdist_gower",
    "cdist_manhattan",
    "cdist_soft_dtw",
    "ctw",
    "dtw",
    "gak",
    "get_cdist_function",
    "gower",
    "manhattan",
    "soft_dtw",
]
