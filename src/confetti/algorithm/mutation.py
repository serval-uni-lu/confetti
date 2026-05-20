"""Bitflip mutation operator for CONFETTI's genetic algorithm."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pymoo.core.mutation import Mutation


class _HasNVar(Protocol):
    n_var: int


class BitflipMutation(Mutation):
    """
    Flip individual bits of binary decision vectors.

    Each bit in every selected individual is independently flipped with
    probability ``prob_var``.  This is a drop-in replacement for
    ``pymoo.operators.mutation.bitflip.BitflipMutation``.

    There are **two** probability knobs:

    * ``prob`` — population-level probability that an individual is
      mutated at all.  Handled by the inherited ``Mutation.do()``
      method.
    * ``prob_var`` — per-variable probability that each bit is flipped
      inside ``_do()``.  When *None*, defaults to
      ``min(0.5, 1 / problem.n_var)``.

    Parameters
    ----------
    ``prob`` : float, default=1.0
        Probability that an individual is selected for mutation.
    ``prob_var`` : float or None, default=None
        Per-bit flip probability.  When *None*, the default
        ``min(0.5, 1 / problem.n_var)`` is used.
    """

    def _do(self, problem: _HasNVar, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Flip bits in the population.

        Parameters
        ----------
        ``problem`` : object
            A problem-like object exposing an ``n_var`` attribute.
        ``X`` : np.ndarray
            Binary population of shape ``(n_individuals, n_var)``.

        Returns
        -------
        np.ndarray
            Mutated population with the same shape and dtype as ``X``.
        """
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = np.random.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp
