"""Two-point crossover operator for CONFETTI's genetic algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np

from confetti.algorithm._operators import Crossover


class TwoPointCrossover(Crossover):
    """
    Binary two-point crossover.

    For each mating, two split points ``i`` and ``j`` (``0 < i <= j < n``)
    are chosen uniformly at random.  Offspring are constructed by swapping
    the middle segment ``[i, j)`` between the two parents:

    * Offspring 0: ``parent_0[:i]  | parent_1[i:j] | parent_0[j:]``
    * Offspring 1: ``parent_1[:i]  | parent_0[i:j] | parent_1[j:]``

    This is a drop-in replacement for
    ``pymoo.operators.crossover.pntx.TwoPointCrossover``.

    Parameters
    ----------
    ``prob`` : float, default=0.9
        Per-mating probability that crossover is applied.  Matings that
        do not undergo crossover copy parents unchanged.  Handled by the
        inherited ``Crossover.do()`` method.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)

    def _do(self, problem: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply two-point crossover to paired parents.

        Parameters
        ----------
        ``problem`` : object
            A problem-like object exposing an ``n_var`` attribute.
        ``X`` : np.ndarray
            Parent array of shape ``(2, n_matings, n_var)``.

        Returns
        -------
        np.ndarray
            Offspring array of shape ``(2, n_matings, n_var)``.
        """
        _, n_matings, n_var = X.shape

        r = np.vstack([np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, :2]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, n_var)])

        M = np.full((n_matings, n_var), False)
        for i in range(n_matings):
            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[i, int(a) : int(b)] = True
                j += 2

        Xp = np.copy(X)
        Xp[0][M] = X[1][M]
        Xp[1][M] = X[0][M]
        return Xp
