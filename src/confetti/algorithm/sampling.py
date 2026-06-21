"""Binary random sampling operator for CONFETTI's genetic algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np

from confetti.algorithm._operators import Sampling


class BinaryRandomSampling(Sampling):
    """
    Generate an initial population of binary decision vectors.

    Each variable in each individual is drawn independently from a
    Bernoulli(0.5) distribution.

    Parameters
    ----------
    ``seed`` : int or None, default=None
        Seed for the internal ``numpy.random.Generator``.

    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    def _do(self, problem: Any, n_samples: int, **kwargs: Any) -> np.ndarray:
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
