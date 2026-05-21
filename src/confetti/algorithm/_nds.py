"""Fast non-dominated sorting for multi-objective optimization."""

from __future__ import annotations

import numpy as np

try:
    from confetti._rust_core import fast_non_dominated_sort_py as _rs_nds
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """
    Partition a set of objective vectors into non-dominated fronts.

    Uses a vectorised domination-matrix approach: for every pair
    ``(i, j)`` we check whether *i* dominates *j* (all objectives ≤
    and at least one <), then iteratively peel off non-dominated layers.

    Parameters
    ----------
    ``F`` : np.ndarray
        Objective matrix of shape ``(n, n_obj)``.  All objectives are
        **minimised**.

    Returns
    -------
    list[np.ndarray]
        Fronts ordered from best (0) to worst.  Each element is an
        int array of row indices into *F*.
    """
    if _HAS_RUST:
        return [np.asarray(f) for f in _rs_nds(np.ascontiguousarray(F, dtype=np.float64))]

    n = F.shape[0]
    if n == 0:
        return []

    # dominates[i, j] = True iff solution i dominates solution j
    # i dominates j when: F[i] <= F[j] in ALL objectives AND F[i] < F[j] in AT LEAST one
    diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]   # (n, n, n_obj)
    all_leq = np.all(diff <= 0, axis=2)                 # (n, n)
    any_lt = np.any(diff < 0, axis=2)                   # (n, n)
    dominates = all_leq & any_lt                        # (n, n)

    # Number of solutions that dominate each solution
    domination_count = dominates.sum(axis=0)            # (n,)

    remaining = np.ones(n, dtype=bool)
    fronts: list[np.ndarray] = []

    while remaining.any():
        # Current front: remaining solutions with zero dominators
        front_mask = remaining & (domination_count == 0)
        front = np.where(front_mask)[0]

        if front.size == 0:
            # Safety valve: if numerical issues prevent progress,
            # dump all remaining into the last front.
            fronts.append(np.where(remaining)[0])
            break

        fronts.append(front)
        remaining[front] = False

        # Update domination counts: remove contributions from the
        # solutions we just assigned to a front.
        domination_count -= dominates[front].sum(axis=0)

    return fronts
