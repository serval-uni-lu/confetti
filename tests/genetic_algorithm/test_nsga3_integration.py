"""
End-to-end NSGA-III integration tests on a ``CounterfactualProblem``.

These assert the behavioral contract pymoo's NSGA-III currently fulfills on
this problem — binary output, feasible population, non-dominated front,
termination at the requested generation, infeasible-returns-None behavior.

Assertions are implementation-agnostic. A Rust re-implementation using a
different PRNG should still pass every test in this file (bit-exact equality
of result.X is **intentionally not** asserted).
"""

import numpy as np
import pytest

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.pntx import TwoPointCrossover
from confetti.algorithm.mutation import BitflipMutation
from confetti.algorithm.sampling import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nsga3(n_obj, pop_size=40, n_partitions=3):
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    return NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=1.0),
        mutation=BitflipMutation(prob=0.9),
    )


def _non_dominated(F: np.ndarray) -> np.ndarray:
    """Return a boolean mask marking non-dominated rows of F (minimization)."""
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i ?
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                mask[i] = False
                break
    return mask


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_minimize_returns_binary_X(problem_factory):
    """
    Using the (confidence, proximity) config so the confidence objective is
    correctly negated. Theta is low enough that some solutions are feasible.
    """
    problem = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=True,
        theta=0.35,
        start_timestep=2,
        subsequence_length=3,
    )
    algorithm = _make_nsga3(n_obj=2)
    result = minimize(problem, algorithm, get_termination("n_gen", 10), seed=1, verbose=False)

    assert result.X is not None
    X = np.atleast_2d(result.X)
    assert X.ndim == 2
    assert X.shape[1] == problem.n_var
    uniq = set(np.unique(X.astype(np.int64)).tolist())
    assert uniq.issubset({0, 1})


def test_minimize_returns_feasible_population(problem_factory):
    problem = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=True,
        theta=0.35,
        start_timestep=2,
        subsequence_length=3,
    )
    algorithm = _make_nsga3(n_obj=2)
    result = minimize(problem, algorithm, get_termination("n_gen", 10), seed=1, verbose=False)

    # result.G is what pymoo stored from out["G"]. Every returned individual
    # must satisfy G ≤ 0 (theta - confidence ≤ 0 ⇒ confidence ≥ theta).
    G = np.asarray(result.G)
    assert (G <= 1e-6).all(), f"Some returned individuals are infeasible: max G = {G.max()}"


def test_minimize_returns_non_dominated_front(problem_factory):
    problem = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=True,
        theta=0.35,
        start_timestep=2,
        subsequence_length=3,
    )
    algorithm = _make_nsga3(n_obj=2)
    result = minimize(problem, algorithm, get_termination("n_gen", 15), seed=1, verbose=False)

    F = np.atleast_2d(result.F)
    nd = _non_dominated(F)
    assert nd.all(), (
        f"Expected every returned solution to be non-dominated; "
        f"{(~nd).sum()} of {len(F)} are dominated."
    )


def test_minimize_infeasible_returns_none_X(problem_factory, mock_classifier_weak):
    """With a uniform classifier and theta=0.99, no individual can ever satisfy G ≤ 0."""
    problem = problem_factory(
        classifier=mock_classifier_weak,
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=True,
        theta=0.99,
        start_timestep=2,
        subsequence_length=3,
    )
    algorithm = _make_nsga3(n_obj=2)
    result = minimize(problem, algorithm, get_termination("n_gen", 8), seed=1, verbose=False)

    assert result.X is None


def test_minimize_respects_n_gen_termination(problem_factory):
    problem = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=True,
        theta=0.35,
    )
    algorithm = _make_nsga3(n_obj=2)
    n_gen = 7
    result = minimize(problem, algorithm, get_termination("n_gen", n_gen), seed=1, verbose=False)
    assert result.algorithm.n_gen == n_gen + 1  # pymoo counts the initial generation; this is locked behavior


def test_minimize_property_stable_across_runs(problem_factory):
    """
    The same pymoo seed must yield *equivalent* shape/feasibility/size across
    two independent runs. We deliberately do **not** assert bit-exact equality
    of result.X — that would make the suite fail on any reimplementation.
    """
    def _run():
        problem = problem_factory(
            optimize_confidence=True,
            optimize_sparsity=False,
            optimize_proximity=True,
            theta=0.35,
            start_timestep=2,
            subsequence_length=3,
        )
        algorithm = _make_nsga3(n_obj=2)
        return minimize(problem, algorithm, get_termination("n_gen", 10), seed=1, verbose=False)

    r1 = _run()
    r2 = _run()

    assert (r1.X is None) == (r2.X is None)
    if r1.X is not None:
        X1 = np.atleast_2d(r1.X)
        X2 = np.atleast_2d(r2.X)
        assert X1.shape == X2.shape
        F1 = np.atleast_2d(r1.F)
        F2 = np.atleast_2d(r2.F)
        assert F1.shape == F2.shape


def test_minimize_with_confidence_and_sparsity_preserves_quirk(problem_factory):
    """
    Smoke test for the CONFETTI default config (confidence + sparsity + proximity).
    Because of the locked sign-asymmetry quirk, confidence enters F as ``+f1``
    rather than ``-f1``. The GA therefore pushes f1 *down*, but the G ≥ 0
    constraint forces f1 ≥ theta. The expectation is that feasible solutions
    cluster close to the theta boundary.
    """
    theta = 0.5
    problem = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=True,
        optimize_proximity=True,
        theta=theta,
        start_timestep=2,
        subsequence_length=3,
    )
    algorithm = _make_nsga3(n_obj=3, n_partitions=3)
    result = minimize(problem, algorithm, get_termination("n_gen", 15), seed=1, verbose=False)

    if result.X is None:
        pytest.skip("No feasible solution found with the current fixtures at theta=0.5.")

    # Column 0 of F is +f1 (the quirk). Feasibility requires f1 ≥ theta.
    F = np.atleast_2d(result.F)
    assert (F[:, 0] >= theta - 1e-6).all()


def test_minimize_n_gen_one(problem_factory):
    """``n_gen=1`` is the minimum number of generations; must not crash."""
    problem = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=True,
        theta=0.35,
        start_timestep=2,
        subsequence_length=3,
    )
    algorithm = _make_nsga3(n_obj=2)
    result = minimize(problem, algorithm, get_termination("n_gen", 1), seed=1, verbose=False)
    assert result.algorithm.n_gen == 2  # pymoo counts initial generation
    if result.X is not None:
        X = np.atleast_2d(result.X)
        assert X.shape[1] == problem.n_var


def test_minimize_small_pop_size(problem_factory):
    """A very small population (``pop_size=10``) must still produce valid results."""
    problem = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=True,
        theta=0.35,
        start_timestep=2,
        subsequence_length=3,
    )
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=2)
    algorithm = NSGA3(
        pop_size=10,
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=1.0),
        mutation=BitflipMutation(prob=0.9),
    )
    result = minimize(problem, algorithm, get_termination("n_gen", 5), seed=1, verbose=False)
    if result.X is not None:
        X = np.atleast_2d(result.X)
        assert X.shape[1] == problem.n_var
        uniq = set(np.unique(X.astype(np.int64)).tolist())
        assert uniq.issubset({0, 1})
