"""
Tests for the TwoPointCrossover operator.

These tests lock in the structural invariants of two-point crossover on binary
strings. They do not require a CounterfactualProblem — the operator's contract
is decoupled from the domain problem.

Invariants captured:
  * Output shape and binary-ness.
  * With ``prob=0.0``, the population is unchanged (identity).
  * With ``prob=1.0``, every offspring is a valid two-point mix: there exist
    split points 0 < i ≤ j < n such that the offspring matches one parent on
    ``[0:i) ∪ [j:n)`` and the other parent on ``[i:j)``.
  * Alleles are preserved: every bit at index k comes from one of the two parents
    at that same index k.
"""

import numpy as np
import pytest

from pymoo.core.population import Population
from confetti.algorithm.crossover import TwoPointCrossover


def _two_parent_arr(n_var: int, seed: int = 0, n_matings: int = 5):
    """Build an X array of shape (2, n_matings, n_var) with two distinct parents per mating."""
    rng = np.random.default_rng(seed)
    parents_a = rng.integers(0, 2, size=(n_matings, n_var), dtype=np.int64).astype(bool)
    parents_b = ~parents_a  # guarantee distinct bits at every index
    X = np.stack([parents_a, parents_b], axis=0)  # (2, n_matings, n_var)
    return X


def _find_two_point_split(parent_a, parent_b, offspring):
    """
    Return (i, j, swapped) such that offspring matches parent_a on [0:i) and [j:n)
    and parent_b on [i:j) — or None if no such split exists.

    ``swapped=True`` means the middle segment came from parent_a instead.
    """
    n = len(offspring)
    for swapped in (False, True):
        source_outer = parent_a if not swapped else parent_b
        source_inner = parent_b if not swapped else parent_a
        for i in range(0, n + 1):
            if not np.array_equal(offspring[:i], source_outer[:i]):
                continue
            for j in range(i, n + 1):
                if (
                    np.array_equal(offspring[i:j], source_inner[i:j])
                    and np.array_equal(offspring[j:], source_outer[j:])
                ):
                    return (i, j, swapped)
    return None


def test_crossover_output_shape_and_binary(dummy_problem_factory):
    prob = dummy_problem_factory(n_var=12)
    X = _two_parent_arr(n_var=12, seed=1, n_matings=4)
    np.random.seed(1)
    off = TwoPointCrossover(prob=1.0)._do(prob, X)
    assert off.shape == (2, 4, 12)  # (n_offsprings, n_matings, n_var)
    uniq = set(np.unique(off.astype(np.int64)).tolist())
    assert uniq.issubset({0, 1})


def test_crossover_prob_zero_is_identity(dummy_problem_factory):
    """
    ``prob=0.0`` at the ``.do()`` level means no mating is actually crossed
    over — offspring are copied directly from the parent pool. Order may be
    permuted across offspring slots, but the *set* of offspring X-vectors
    equals the *set* of parent X-vectors within each mating.
    """
    n_var = 10
    n_matings = 6
    prob = dummy_problem_factory(n_var=n_var)

    X = _two_parent_arr(n_var=n_var, seed=2, n_matings=n_matings)
    # Build a Population per mating for .do()
    # pymoo expects a flat list of parents in pairs: pop[0]/pop[1] = mating 0, etc.
    flat = np.concatenate([X[0], X[1]], axis=0)  # just materialize parents
    # The .do() API with parents argument: we hand it (n_matings, n_parents) indices.
    parent_indices = np.stack(
        [np.arange(n_matings), np.arange(n_matings) + n_matings], axis=1
    )  # shape (n_matings, 2)

    pop = Population.new("X", flat)
    np.random.seed(0)
    off_pop = TwoPointCrossover(prob=0.0).do(prob, pop, parents=parent_indices)
    off_X = off_pop.get("X")
    # n_offsprings=2 per mating → 2 * n_matings rows
    assert off_X.shape == (2 * n_matings, n_var)

    # For each mating, the two offspring rows must equal the two parents
    # (as a set).
    for m in range(n_matings):
        mating_offspring = off_X[m::n_matings]  # offsprings for mating m are at indices m and m+n_matings
        parents_set = {tuple(X[0, m].tolist()), tuple(X[1, m].tolist())}
        offspring_set = {tuple(row.tolist()) for row in mating_offspring}
        assert offspring_set == parents_set


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_crossover_prob_one_two_point_structure(dummy_problem_factory, seed):
    """For every offspring, verify a valid 2-point split exists."""
    n_var = 16
    n_matings = 8
    prob = dummy_problem_factory(n_var=n_var)
    X = _two_parent_arr(n_var=n_var, seed=seed, n_matings=n_matings)

    np.random.seed(seed)
    off = TwoPointCrossover(prob=1.0)._do(prob, X)  # (2, n_matings, n_var)
    assert off.shape == (2, n_matings, n_var)

    for m in range(n_matings):
        pa = X[0, m].astype(np.int64)
        pb = X[1, m].astype(np.int64)
        for o_idx in range(2):
            o = off[o_idx, m].astype(np.int64)
            split = _find_two_point_split(pa, pb, o)
            assert split is not None, (
                f"Offspring {o_idx} of mating {m} is not a valid 2-point mix of its parents.\n"
                f"Parent A: {pa.tolist()}\nParent B: {pb.tolist()}\nOffspring: {o.tolist()}"
            )


def test_crossover_preserves_alleles(dummy_problem_factory):
    """
    For every offspring, bit at index k must be equal to bit at index k of at
    least one parent. This is a weaker — but important — invariant that must
    hold regardless of crossover variant.
    """
    n_var = 20
    n_matings = 5
    prob = dummy_problem_factory(n_var=n_var)
    rng = np.random.default_rng(99)
    # Use random (non-antipodal) parents to make the invariant non-trivial
    parents_a = rng.integers(0, 2, size=(n_matings, n_var), dtype=np.int64).astype(bool)
    parents_b = rng.integers(0, 2, size=(n_matings, n_var), dtype=np.int64).astype(bool)
    X = np.stack([parents_a, parents_b], axis=0)

    np.random.seed(3)
    off = TwoPointCrossover(prob=1.0)._do(prob, X)

    for m in range(n_matings):
        pa = X[0, m]
        pb = X[1, m]
        for o_idx in range(2):
            o = off[o_idx, m]
            match_a = (o == pa)
            match_b = (o == pb)
            assert np.all(match_a | match_b)


def test_crossover_odd_population_sanity(dummy_problem_factory):
    """Odd numbers of matings must still produce a valid output."""
    n_var = 8
    n_matings = 3
    prob = dummy_problem_factory(n_var=n_var)
    X = _two_parent_arr(n_var=n_var, seed=5, n_matings=n_matings)

    np.random.seed(11)
    off = TwoPointCrossover(prob=1.0)._do(prob, X)
    assert off.shape == (2, 3, 8)
    uniq = set(np.unique(off.astype(np.int64)).tolist())
    assert uniq.issubset({0, 1})


def test_crossover_n_var_two_minimal_nontrivial(dummy_problem_factory):
    """``n_var=2`` is the smallest string where a non-trivial 2-point split can occur."""
    n_var = 2
    n_matings = 5
    prob = dummy_problem_factory(n_var=n_var)
    X = _two_parent_arr(n_var=n_var, seed=10, n_matings=n_matings)

    np.random.seed(10)
    off = TwoPointCrossover(prob=1.0)._do(prob, X)
    assert off.shape == (2, n_matings, n_var)

    for m in range(n_matings):
        pa = X[0, m].astype(np.int64)
        pb = X[1, m].astype(np.int64)
        for o_idx in range(2):
            o = off[o_idx, m].astype(np.int64)
            split = _find_two_point_split(pa, pb, o)
            assert split is not None


def test_crossover_n_var_one_degenerate(dummy_problem_factory):
    """
    With ``n_var=1`` there is no room for a non-trivial 2-point split.
    Every offspring must equal one of its two parents.
    """
    n_var = 1
    n_matings = 4
    prob = dummy_problem_factory(n_var=n_var)
    X = _two_parent_arr(n_var=n_var, seed=12, n_matings=n_matings)

    np.random.seed(12)
    off = TwoPointCrossover(prob=1.0)._do(prob, X)
    assert off.shape == (2, n_matings, n_var)

    for m in range(n_matings):
        pa = X[0, m]
        pb = X[1, m]
        for o_idx in range(2):
            o = off[o_idx, m]
            assert np.array_equal(o, pa) or np.array_equal(o, pb)


def test_crossover_identical_parents(dummy_problem_factory):
    """When both parents are identical, every offspring must be identical to the parent."""
    n_var = 12
    n_matings = 5
    prob = dummy_problem_factory(n_var=n_var)
    rng = np.random.default_rng(20)
    parents = rng.integers(0, 2, size=(n_matings, n_var), dtype=np.int64).astype(bool)
    X = np.stack([parents, parents], axis=0)  # identical parents

    np.random.seed(20)
    off = TwoPointCrossover(prob=1.0)._do(prob, X)
    for m in range(n_matings):
        for o_idx in range(2):
            np.testing.assert_array_equal(off[o_idx, m], parents[m])
