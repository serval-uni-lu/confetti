"""Binary random sampling operator for CONFETTI's genetic algorithm."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pymoo.core.sampling import Sampling


class _HasNVar(Protocol):
    n_var: int


class BinaryRandomSampling(Sampling):
    """
    Generate an initial population of binary decision vectors.

    Each variable in each individual is drawn independently from a
    Bernoulli(0.5) distribution.  This is a drop-in replacement for
    ``pymoo.operators.sampling.rnd.BinaryRandomSampling``.

    Parameters
    ----------
    ``seed`` : int or None, default=None
        Seed for the internal ``numpy.random.Generator``.  When *None*,
        the global numpy random state is used for backwards compatibility
        with pymoo's seeding mechanism (``minimize(..., seed=...)``).

    Notes
    -----
    This class extends ``pymoo.core.sampling.Sampling`` so it is accepted
    by ``pymoo.algorithms.moo.nsga3.NSGA3`` without modification.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._seed = seed

    def _do(self, problem: _HasNVar, n_samples: int, **kwargs) -> np.ndarray:
        """
        Return a random binary population.

        Parameters
        ----------
        ``problem`` : object
            A problem-like object exposing an ``n_var`` attribute that
            gives the number of decision variables.
        ``n_samples`` : int
            Number of individuals (rows) to generate.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_samples, problem.n_var)`` where
            each element is ``True`` with probability 0.5.
        """
        n_var = problem.n_var
        if self._seed is not None:
            rng = np.random.default_rng(self._seed)
            val = rng.random((n_samples, n_var))
        else:
            val = np.random.random((n_samples, n_var))
        return (val < 0.5).astype(np.bool_)
