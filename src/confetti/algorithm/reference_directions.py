"""Das-Dennis reference directions for NSGA-III."""

from __future__ import annotations

import numpy as np


def das_dennis(n_dim: int, n_partitions: int) -> np.ndarray:
    """
    Generate Das-Dennis reference directions on an (``n_dim`` - 1)-simplex.

    Produces a set of uniformly spaced weight vectors whose components sum
    to one and are all non-negative.  The number of points equals
    ``C(n_dim + n_partitions - 1, n_partitions)`` (stars-and-bars).

    Parameters
    ----------
    ``n_dim`` : int
        Number of objectives (dimensionality of each reference direction).
        Must be >= 1.
    ``n_partitions`` : int
        Number of equal divisions along each axis.  Must be >= 0.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, n_dim)`` with ``dtype=float64``,
        lexicographically sorted by row.

    Raises
    ------
    ValueError
        If ``n_dim < 1`` or ``n_partitions < 0``.

    References
    ----------
    Das, I. & Dennis, J.E. (1998). Normal-Boundary Intersection: A New
    Method for Generating the Pareto Surface in Nonlinear Multicriteria
    Optimization Problems. *SIAM Journal on Optimization*, 8(3), 631-657.
    """
    if n_dim < 1:
        raise ValueError(f"n_dim must be >= 1, got {n_dim}")
    if n_partitions < 0:
        raise ValueError(f"n_partitions must be >= 0, got {n_partitions}")

    if n_dim == 1:
        return np.array([[1.0]])

    if n_partitions == 0:
        return np.full((1, n_dim), 1.0 / n_dim)

    points: list[list[float]] = []
    stack: list[tuple[list[float], int]] = [([], n_partitions)]

    while stack:
        partial, beta = stack.pop()
        depth = len(partial)

        if depth == n_dim - 1:
            points.append([*partial, beta / n_partitions])
        else:
            for i in range(beta, -1, -1):
                stack.append(([*partial, i / n_partitions], beta - i))

    ref_dirs = np.array(points, dtype=np.float64)

    order = np.lexsort(ref_dirs[:, ::-1].T)
    return ref_dirs[order]
