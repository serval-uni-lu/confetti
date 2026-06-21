"""Hyperplane normalization for NSGA-III objective-space scaling."""

from __future__ import annotations

import numpy as np

try:
    from confetti._rust_core import normalization_update_py as _rs_norm_update

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class HyperplaneNormalization:
    """
    Track ideal / nadir points across generations and normalise
    objective vectors into ``[0, 1]`` for reference-direction association.

    Parameters
    ----------
    ``n_dim`` : int
        Number of objectives.
    """

    def __init__(self, n_dim: int) -> None:
        self.n_dim = n_dim
        self.ideal_point = np.full(n_dim, np.inf)
        self.nadir_point: np.ndarray | None = None

    def update(self, F: np.ndarray, first_front: np.ndarray) -> None:
        """
        Recompute ideal and nadir points from the current population.

        Parameters
        ----------
        ``F`` : np.ndarray
            Objective matrix of shape ``(n, n_obj)``.
        ``first_front`` : np.ndarray
            Index array of solutions belonging to the first non-dominated
            front.
        """
        if _HAS_RUST:
            new_ideal, nadir = _rs_norm_update(
                np.ascontiguousarray(self.ideal_point, dtype=np.float64),
                np.ascontiguousarray(F, dtype=np.float64),
                np.ascontiguousarray(first_front, dtype=np.int64),
                self.n_dim,
            )
            self.ideal_point = np.asarray(new_ideal)
            self.nadir_point = np.asarray(nadir) if nadir is not None else None
            return

        # --- ideal point (component-wise minimum) ---
        self.ideal_point = np.minimum(self.ideal_point, F.min(axis=0))

        F_front = F[first_front]

        if len(first_front) < self.n_dim:
            # Not enough points for a hyperplane; fall back to worst
            # point of the first front.
            self.nadir_point = F_front.max(axis=0)
            return

        # --- extreme points via Achievement Scalarizing Function ---
        extreme_indices = _find_extreme_points(F_front, self.ideal_point)
        extreme_points = F_front[extreme_indices]

        # --- nadir point via hyperplane fitting ---
        nadir = _compute_nadir(extreme_points, self.ideal_point)
        if nadir is not None:
            self.nadir_point = nadir
        else:
            # Degenerate hyperplane; use worst of the first front.
            self.nadir_point = F_front.max(axis=0)


def _find_extreme_points(F_front: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    """
    For each objective dimension, find the front member that minimises
    the Achievement Scalarizing Function (ASF) weighted toward that
    dimension.

    Returns an array of indices into *F_front* (length ``n_obj``).
    """
    n_obj = F_front.shape[1]
    translated = F_front - ideal

    extreme = np.empty(n_obj, dtype=np.intp)
    for j in range(n_obj):
        # Weight vector: small weight on target objective, large on others
        weights = np.full(n_obj, 1e6)
        weights[j] = 1.0
        asf = np.max(translated * weights, axis=1)
        extreme[j] = np.argmin(asf)

    return extreme


def _compute_nadir(extreme_points: np.ndarray, ideal: np.ndarray) -> np.ndarray | None:
    """
    Fit a hyperplane through the extreme points and derive intercepts.

    Returns the nadir point (``ideal + intercepts``), or *None* if the
    system is singular.
    """
    n_dim = extreme_points.shape[1]
    translated = extreme_points - ideal

    # Solve  translated @ plane = 1  for the plane-normal coefficients.
    try:
        plane = np.linalg.solve(translated[:n_dim], np.ones(n_dim))
    except np.linalg.LinAlgError:
        return None

    # Intercepts along each axis: 1 / plane[j]
    if np.any(np.abs(plane) < 1e-14):
        return None

    intercepts = 1.0 / plane
    if np.any(intercepts <= 0):
        return None

    return ideal + intercepts
