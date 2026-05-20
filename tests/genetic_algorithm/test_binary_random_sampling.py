"""
Tests for the BinaryRandomSampling operator.

Locks the contract a re-implementation must satisfy: shape, binary values,
uniform Bernoulli(0.5) distribution. We deliberately do NOT assert bit-exact
equality against pymoo's RNG — a re-implementation will use a different PRNG,
so only statistical / structural properties are specified.
"""

import numpy as np
import pytest

from confetti.algorithm.sampling import BinaryRandomSampling


def _sample(problem, n_samples, seed=1):
    np.random.seed(seed)
    return BinaryRandomSampling()._do(problem, n_samples)


@pytest.mark.parametrize("n_samples, n_var", [(10, 8), (100, 32), (1, 64)])
def test_sampling_shape(dummy_problem_factory, n_samples, n_var):
    prob = dummy_problem_factory(n_var)
    X = _sample(prob, n_samples)
    assert X.shape == (n_samples, n_var)


def test_sampling_values_binary(dummy_problem_factory):
    prob = dummy_problem_factory(n_var=20)
    X = _sample(prob, n_samples=50)
    # Cast to int for a clean {0, 1} comparison that accepts either bool or int dtypes.
    uniq = set(np.unique(X.astype(np.int64)).tolist())
    assert uniq.issubset({0, 1})


def test_sampling_dtype_is_bool(dummy_problem_factory):
    """
    Current spec: pymoo returns a boolean ndarray. Locked because downstream
    code (``_counterfactual_problem._evaluate``) uses the mask with
    ``np.where``, which works for both bool and int — but dtype stability
    matters for anyone porting operator-level behavior.
    """
    prob = dummy_problem_factory(n_var=8)
    X = _sample(prob, n_samples=4)
    assert X.dtype == np.bool_


def test_sampling_seed_reproducibility_within_implementation(dummy_problem_factory):
    """
    Same RNG seed + same implementation → identical output. This is a property
    of the *pymoo* implementation, not a requirement on a re-implementation
    using a different PRNG. The reimplementation test should assert its own
    seed-reproducibility analogously rather than matching these exact bits.
    """
    prob = dummy_problem_factory(n_var=16)
    a = _sample(prob, n_samples=32, seed=42)
    b = _sample(prob, n_samples=32, seed=42)
    np.testing.assert_array_equal(a, b)

    c = _sample(prob, n_samples=32, seed=43)
    assert not np.array_equal(a, c)


def test_sampling_mean_near_half(dummy_problem_factory):
    """Over a large sample, the Bernoulli(0.5) mean should be within ±0.05."""
    prob = dummy_problem_factory(n_var=64)
    X = _sample(prob, n_samples=500, seed=7)
    mean = float(X.astype(np.float64).mean())
    assert 0.45 <= mean <= 0.55


def test_sampling_n_samples_zero(dummy_problem_factory):
    """Requesting zero samples must return a valid empty array."""
    prob = dummy_problem_factory(n_var=8)
    X = _sample(prob, n_samples=0)
    assert X.shape == (0, 8)
    assert X.dtype == np.bool_


def test_sampling_n_var_one(dummy_problem_factory):
    """``n_var=1`` is the minimum meaningful variable count — each sample is a single bit."""
    prob = dummy_problem_factory(n_var=1)
    X = _sample(prob, n_samples=50, seed=9)
    assert X.shape == (50, 1)
    assert X.dtype == np.bool_
    uniq = set(np.unique(X.astype(np.int64)).tolist())
    assert uniq == {0, 1}
