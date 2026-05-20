"""Abstract base classes for genetic algorithm operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Sampling(ABC):
    """
    Base class for population initialization operators.

    Subclasses must implement ``_do`` which returns a raw ndarray of
    shape ``(n_samples, problem.n_var)``.
    """

    def do(self, problem: Any, n_samples: int) -> np.ndarray:
        """
        Generate an initial population.

        Parameters
        ----------
        ``problem`` : Problem
            The optimization problem (must expose ``n_var``).
        ``n_samples`` : int
            Number of individuals to generate.

        Returns
        -------
        np.ndarray
            Population matrix of shape ``(n_samples, problem.n_var)``.
        """
        return self._do(problem, n_samples)

    @abstractmethod
    def _do(self, problem: Any, n_samples: int, **kwargs: Any) -> np.ndarray: ...


class Mutation(ABC):
    """
    Base class for mutation operators with per-individual probability gating.

    The ``do`` method decides *which* individuals are mutated (controlled
    by ``prob``).  The subclass ``_do`` applies the actual mutation to the
    selected rows.

    Parameters
    ----------
    ``prob`` : float
        Probability that any given individual is passed to ``_do``.
    ``prob_var`` : float or None
        Per-variable mutation probability used by ``get_prob_var()``.
        When *None*, defaults to ``min(0.5, 1 / problem.n_var)``.
    """

    def __init__(self, prob: float = 1.0, prob_var: float | None = None) -> None:
        self.prob = prob
        self.prob_var = prob_var

    def do(self, problem: Any, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Mutate a population with per-individual probability gating.

        Parameters
        ----------
        ``problem`` : Problem
            The optimization problem.
        ``X`` : np.ndarray
            Decision-variable matrix of shape ``(n, n_var)``.
        ``rng`` : numpy.random.Generator
            Random number generator for probability sampling.

        Returns
        -------
        np.ndarray
            Mutated population, same shape as *X*.
        """
        Xp = X.copy()
        mask = rng.random(X.shape[0]) < self.prob
        if mask.any():
            Xp[mask] = self._do(problem, X[mask])
        return Xp

    def get_prob_var(self, problem: Any, size: tuple[int, ...] | None = None) -> float:
        """
        Return the per-variable mutation probability.

        When ``prob_var`` was not set at construction, defaults to
        ``min(0.5, 1 / problem.n_var)``.

        Parameters
        ----------
        ``problem`` : object
            Must expose an ``n_var`` attribute.
        ``size`` : tuple or None
            Ignored (kept for API compatibility with pymoo).

        Returns
        -------
        float
            Per-variable mutation probability.
        """
        if self.prob_var is not None:
            return self.prob_var
        return min(0.5, 1.0 / problem.n_var)

    @abstractmethod
    def _do(self, problem: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray: ...


class Crossover(ABC):
    """
    Base class for crossover operators with per-mating probability gating.

    Parameters
    ----------
    ``prob`` : float
        Probability that a given mating actually performs crossover.
    ``n_parents`` : int
        Number of parents per mating.
    ``n_offsprings`` : int
        Number of offspring produced per mating.
    """

    def __init__(self, prob: float = 0.9, n_parents: int = 2, n_offsprings: int = 2) -> None:
        self.prob = prob
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings

    def do(
        self, problem: Any, X: np.ndarray, parents: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Perform crossover on selected parent pairs.

        Parameters
        ----------
        ``problem`` : Problem
            The optimization problem.
        ``X`` : np.ndarray
            Decision-variable matrix of shape ``(n, n_var)``.
        ``parents`` : np.ndarray
            Index array of shape ``(n_matings, 2)`` selecting parent rows.
        ``rng`` : numpy.random.Generator
            Random number generator for probability sampling.

        Returns
        -------
        np.ndarray
            Offspring matrix of shape ``(n_offsprings * n_matings, n_var)``.
        """
        n_matings = parents.shape[0]
        n_var = X.shape[1]

        # Build (2, n_matings, n_var) parent tensor
        P = np.stack([X[parents[:, 0]], X[parents[:, 1]]], axis=0)
        Q = P.copy()

        cross = rng.random(n_matings) < self.prob
        if cross.any():
            Q[:, cross] = self._do(problem, P[:, cross])

        return Q.reshape(-1, n_var)

    @abstractmethod
    def _do(self, problem: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray: ...
