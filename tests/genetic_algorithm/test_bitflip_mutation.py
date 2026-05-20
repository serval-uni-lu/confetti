"""
Tests for BitflipMutation.

There are **two** probability knobs a re-implementation must expose and keep
distinct — the tests pin both:

  * ``prob``     — applied at the population level by ``Mutation.do()``: the
                   probability that an individual is mutated at all.
  * ``prob_var`` — applied per-variable by ``BitflipMutation._do()``: the
                   probability that each individual bit is flipped.
                   ``prob_var=None`` in pymoo defaults to
                   ``min(0.5, 1 / problem.n_var)``.

Both matter because ``src/confetti/explainer/explainer.py`` constructs the
operator as ``BitflipMutation(prob=mutation_probability)`` without setting
``prob_var`` — i.e. it relies on the default per-bit rate.
"""

import numpy as np
import pytest

from pymoo.core.population import Population
from confetti.algorithm.mutation import BitflipMutation


def test_mutation_output_shape_and_binary(dummy_problem_factory):
    prob = dummy_problem_factory(n_var=16)
    X = np.zeros((10, 16), dtype=bool)
    np.random.seed(0)
    M = BitflipMutation(prob=1.0, prob_var=0.5)._do(prob, X)
    assert M.shape == X.shape
    uniq = set(np.unique(M.astype(np.int64)).tolist())
    assert uniq.issubset({0, 1})


def test_mutation_prob_zero_is_identity(dummy_problem_factory):
    """
    ``prob=0.0`` at the ``.do()`` level means no individual is mutated — the
    population is returned unchanged regardless of ``prob_var``.
    """
    n_var = 12
    prob = dummy_problem_factory(n_var=n_var)

    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(8, n_var), dtype=np.int64).astype(bool)
    pop = Population.new("X", X)
    np.random.seed(0)
    out = BitflipMutation(prob=0.0, prob_var=1.0).do(prob, pop)
    np.testing.assert_array_equal(out.get("X"), X)


def test_mutation_prob_var_one_flips_all_bits(dummy_problem_factory):
    """
    With ``prob_var=1.0``, every bit in every individual is flipped in
    ``_do``. (``Mutation.do()``'s per-individual ``prob`` can still gate the
    application, so we test ``_do`` directly to isolate the per-bit semantics.)
    """
    n_var = 20
    prob = dummy_problem_factory(n_var=n_var)

    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, size=(6, n_var), dtype=np.int64).astype(bool)
    np.random.seed(0)
    M = BitflipMutation(prob=1.0, prob_var=1.0)._do(prob, X)
    np.testing.assert_array_equal(M, ~X)


def test_mutation_prob_var_zero_is_identity(dummy_problem_factory):
    """With ``prob_var=0.0``, no bits flip in ``_do``."""
    n_var = 20
    prob = dummy_problem_factory(n_var=n_var)

    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, size=(6, n_var), dtype=np.int64).astype(bool)
    np.random.seed(0)
    M = BitflipMutation(prob=1.0, prob_var=0.0)._do(prob, X)
    np.testing.assert_array_equal(M, X)


def test_mutation_prob_var_half_expected_flip_rate(dummy_problem_factory):
    """Over a large population, the empirical per-bit flip rate is close to 0.5."""
    n_var = 100
    prob = dummy_problem_factory(n_var=n_var)

    rng = np.random.default_rng(4)
    X = rng.integers(0, 2, size=(200, n_var), dtype=np.int64).astype(bool)
    np.random.seed(5)
    M = BitflipMutation(prob=1.0, prob_var=0.5)._do(prob, X)

    flips = (M != X).astype(np.float64)
    rate = flips.mean()
    assert 0.45 <= rate <= 0.55


def test_mutation_default_prob_var_uses_one_over_n_var(dummy_problem_factory):
    """
    Spec of the default ``prob_var`` when the user passes only ``prob``:
    ``min(0.5, 1/n_var)``. CONFETTI relies on this default.
    """
    n_var = 20  # 1/20 = 0.05, clearly < 0.5
    prob = dummy_problem_factory(n_var=n_var)

    rng = np.random.default_rng(6)
    X = rng.integers(0, 2, size=(500, n_var), dtype=np.int64).astype(bool)
    np.random.seed(7)
    M = BitflipMutation(prob=1.0)._do(prob, X)  # no prob_var set

    rate = (M != X).astype(np.float64).mean()
    expected = 1.0 / n_var
    assert abs(rate - expected) < 0.02, (
        f"Default per-bit flip rate should be ~{expected:.3f} (=min(0.5, 1/n_var)), got {rate:.3f}"
    )


def test_mutation_determinism_with_seed(dummy_problem_factory):
    """Same numpy-global seed + same implementation → identical output."""
    n_var = 30
    prob = dummy_problem_factory(n_var=n_var)

    rng = np.random.default_rng(8)
    X = rng.integers(0, 2, size=(10, n_var), dtype=np.int64).astype(bool)

    np.random.seed(123)
    a = BitflipMutation(prob=1.0, prob_var=0.3)._do(prob, X)
    np.random.seed(123)
    b = BitflipMutation(prob=1.0, prob_var=0.3)._do(prob, X)
    np.testing.assert_array_equal(a, b)


def test_mutation_n_var_one_default_prob_var(dummy_problem_factory):
    """
    When ``n_var=1``, ``min(0.5, 1/1) = 0.5`` — the boundary where the
    min clamp activates. The empirical flip rate should be approximately 0.5.
    """
    prob = dummy_problem_factory(n_var=1)
    rng = np.random.default_rng(10)
    X = rng.integers(0, 2, size=(500, 1), dtype=np.int64).astype(bool)
    np.random.seed(11)
    M = BitflipMutation(prob=1.0)._do(prob, X)
    rate = (M != X).astype(np.float64).mean()
    assert 0.40 <= rate <= 0.60, f"Expected flip rate ~0.5, got {rate:.3f}"


def test_mutation_homogeneous_all_zeros(dummy_problem_factory):
    """``prob_var=1.0`` on an all-zeros population flips every bit to one."""
    prob = dummy_problem_factory(n_var=16)
    X = np.zeros((10, 16), dtype=bool)
    np.random.seed(0)
    M = BitflipMutation(prob=1.0, prob_var=1.0)._do(prob, X)
    assert np.all(M)


def test_mutation_homogeneous_all_ones(dummy_problem_factory):
    """``prob_var=1.0`` on an all-ones population flips every bit to zero."""
    prob = dummy_problem_factory(n_var=16)
    X = np.ones((10, 16), dtype=bool)
    np.random.seed(0)
    M = BitflipMutation(prob=1.0, prob_var=1.0)._do(prob, X)
    assert not np.any(M)
