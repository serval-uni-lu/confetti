"""NSGA-III multi-objective evolutionary algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from confetti.algorithm._nds import fast_non_dominated_sort
from confetti.algorithm._niching import associate_to_niches, niching_selection
from confetti.algorithm._normalization import HyperplaneNormalization
from confetti.algorithm._operators import Crossover, Mutation, Sampling


@dataclass
class Result:
    """
    Outcome of a multi-objective optimisation run.

    Attributes
    ----------
    ``X`` : np.ndarray or None
        Decision variables of the feasible non-dominated front, shape
        ``(n_solutions, n_var)``.  *None* when no feasible solution was
        found.
    ``F`` : np.ndarray or None
        Objective values, shape ``(n_solutions, n_obj)``.
    ``G`` : np.ndarray or None
        Inequality-constraint values, shape ``(n_solutions, n_ieq_constr)``.
    ``algorithm`` : NSGA3
        The algorithm instance (carries ``n_gen`` and other state).
    """

    X: np.ndarray | None
    F: np.ndarray | None
    G: np.ndarray | None
    algorithm: NSGA3


class NSGA3:
    """
    NSGA-III: Non-dominated Sorting Genetic Algorithm III.

    Parameters
    ----------
    ``pop_size`` : int
        Population size (number of individuals per generation).
    ``ref_dirs`` : np.ndarray
        Reference directions, shape ``(n_ref, n_obj)``.
    ``sampling`` : Sampling
        Population initialization operator.
    ``crossover`` : Crossover
        Crossover operator.
    ``mutation`` : Mutation
        Mutation operator.
    """

    def __init__(
        self,
        pop_size: int,
        ref_dirs: np.ndarray,
        sampling: Sampling,
        crossover: Crossover,
        mutation: Mutation,
    ) -> None:
        self.pop_size = pop_size
        self.ref_dirs = ref_dirs
        self.sampling = sampling
        self.crossover = crossover
        self.mutation = mutation
        self.n_gen: int = 0

    def run(self, problem: Any, n_gen: int, seed: int | None = None) -> Result:
        """
        Execute the NSGA-III optimisation loop.

        Parameters
        ----------
        ``problem`` : Problem
            The multi-objective problem to solve.
        ``n_gen`` : int
            Number of generations to run.
        ``seed`` : int or None
            Random seed for reproducibility.

        Returns
        -------
        Result
            The optimisation result with feasible non-dominated solutions.
        """
        rng = np.random.default_rng(seed)
        n_obj = problem.n_obj

        # --- Initial population ---
        X = self.sampling.do(problem, self.pop_size)
        F, G = problem.evaluate(X)
        norm = HyperplaneNormalization(n_obj)

        # Apply survival on initial population
        X, F, G = self._survive(X, F, G, self.pop_size, norm, rng)

        self.n_gen = 1

        # --- Evolutionary loop ---
        for _ in range(n_gen):
            self.n_gen += 1

            # Mating: tournament selection → crossover → mutation
            n_matings = self.pop_size
            parents = _tournament_selection(F, G, n_matings, rng)
            off_X = self.crossover.do(problem, X, parents, rng)
            off_X = self.mutation.do(problem, off_X, rng)
            off_F, off_G = problem.evaluate(off_X)

            # Merge parents + offspring
            X_all = np.vstack([X, off_X])
            F_all = np.vstack([F, off_F])
            G_all = np.vstack([G, off_G])

            # Environmental selection
            X, F, G = self._survive(X_all, F_all, G_all, self.pop_size, norm, rng)

        # --- Extract result: feasible first-front solutions ---
        return self._extract_result(X, F, G)

    def _survive(
        self,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray,
        n_survive: int,
        norm: HyperplaneNormalization,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select *n_survive* individuals via NSGA-III environmental selection.
        """
        n = X.shape[0]
        if n <= n_survive:
            return X, F, G

        # Constraint violation: CV = sum(max(0, g)) per individual
        cv = np.maximum(0, G).sum(axis=1)

        # Non-dominated sorting (using CV-adjusted objectives)
        # Infeasible solutions are penalised by appending CV as an extra axis
        # for dominance comparison
        F_sort = _cv_adjusted_objectives(F, cv)
        fronts = fast_non_dominated_sort(F_sort)

        # Determine which fronts to keep in full and which is the "last" front
        survivors = np.empty(0, dtype=np.intp)
        last_front_idx = 0
        for i, front in enumerate(fronts):
            if len(survivors) + len(front) <= n_survive:
                survivors = np.concatenate([survivors, front])
                last_front_idx = i + 1
            else:
                break

        n_remaining = n_survive - len(survivors)
        if n_remaining > 0 and last_front_idx < len(fronts):
            last_front = fronts[last_front_idx]

            # Normalization + niche association (use raw F, not CV-adjusted)
            norm.update(F, fronts[0])
            nadir = norm.nadir_point if norm.nadir_point is not None else F.max(axis=0)
            niche_of_all, dist_all = associate_to_niches(F, self.ref_dirs, norm.ideal_point, nadir)

            # Niche counts from already-admitted fronts
            niche_count = np.zeros(len(self.ref_dirs), dtype=np.intp)
            if len(survivors) > 0:
                for niche_idx in niche_of_all[survivors]:
                    niche_count[niche_idx] += 1

            # Select from the last front
            selected = niching_selection(
                n_remaining,
                niche_count,
                niche_of_all[last_front],
                dist_all[last_front],
                rng,
            )
            survivors = np.concatenate([survivors, last_front[selected]])

        return X[survivors], F[survivors], G[survivors]

    def _extract_result(self, X: np.ndarray, F: np.ndarray, G: np.ndarray) -> Result:
        """Build a Result from the final population, filtering to feasible non-dominated solutions."""
        cv = np.maximum(0, G).sum(axis=1)
        feasible = cv <= 0

        if not feasible.any():
            return Result(X=None, F=None, G=None, algorithm=self)

        X_f, F_f, G_f = X[feasible], F[feasible], G[feasible]

        # Return only the first non-dominated front among feasible solutions
        fronts = fast_non_dominated_sort(F_f)
        if len(fronts) > 0:
            first_front = fronts[0]
            return Result(X=X_f[first_front], F=F_f[first_front], G=G_f[first_front], algorithm=self)

        return Result(X=X_f, F=F_f, G=G_f, algorithm=self)


def _tournament_selection(F: np.ndarray, G: np.ndarray, n_matings: int, rng: np.random.Generator) -> np.ndarray:
    """
    Binary tournament selection favouring feasible solutions.

    For each mating slot, two random individuals are drawn and the one
    with lower constraint violation wins.  Ties are broken randomly.

    Returns an index array of shape ``(n_matings, 2)``.
    """
    pop_size = F.shape[0]
    cv = np.maximum(0, G).sum(axis=1)

    parents = np.empty((n_matings, 2), dtype=np.intp)
    for col in range(2):
        a = rng.integers(pop_size, size=n_matings)
        b = rng.integers(pop_size, size=n_matings)

        # Prefer lower constraint violation
        a_wins = cv[a] < cv[b]
        b_wins = cv[b] < cv[a]

        # Break ties randomly
        coin = rng.random(n_matings) < 0.5
        winner = np.where(a_wins, a, np.where(b_wins, b, np.where(coin, a, b)))
        parents[:, col] = winner

    return parents


def _cv_adjusted_objectives(F: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """
    Produce objective values where infeasible solutions are penalised.

    Feasible solutions keep their original F.  Infeasible solutions are
    assigned objectives worse than any feasible solution, ordered by CV.
    """
    feasible = cv <= 0
    if feasible.all():
        return F

    F_adj = F.copy()
    if feasible.any():
        worst = F[feasible].max(axis=0)
    else:
        worst = F.max(axis=0)

    # Push infeasible solutions beyond the worst feasible, scaled by CV
    infeasible = ~feasible
    F_adj[infeasible] = worst + cv[infeasible, np.newaxis]

    return F_adj


def minimize(
    problem: Any,
    algorithm: NSGA3,
    termination: Any,
    seed: int | None = None,
    verbose: bool = False,
) -> Result:
    """
    Run an NSGA-III optimisation.

    Parameters
    ----------
    ``problem`` : Problem
        The multi-objective problem.
    ``algorithm`` : NSGA3
        A configured NSGA-III instance.
    ``termination`` : int or object
        Number of generations, or an object with an ``n_max_gen`` attribute.
    ``seed`` : int or None
        Random seed.
    ``verbose`` : bool
        Unused (kept for API compatibility).

    Returns
    -------
    Result
        The optimisation result.
    """
    if isinstance(termination, int):
        n_gen = termination
    else:
        n_gen = getattr(termination, "n_max_gen", termination)

    return algorithm.run(problem, n_gen, seed=seed)
