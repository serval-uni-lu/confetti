"""
Tests for ``CounterfactualProblem._evaluate``.

This is the deterministic heart of the GA — given a batch of binary masks it
must produce exact, reproducible objective and constraint values. The Rust
re-implementation must match this bit-for-bit (within float tolerance) for the
NSGA-III search to converge to the same Pareto front.

Two current spec quirks are explicitly locked here so the reimplementation
author sees them:

  1. **G shape quirk** — ``out["G"] = [self.theta - f1]`` produces an array of
     shape ``(1, n_samples)`` rather than the ``(n_samples, 1)`` that
     ``n_ieq_constr=1`` implies. This test suite locks ``(1, n_samples)``.

  2. **Confidence-sign asymmetry** — when both ``optimize_confidence`` and
     ``optimize_sparsity`` are True, the confidence column of F is ``+f1``
     (raw probability) rather than ``-f1``. When only confidence is active,
     it is ``-f1``. The reimplementation must preserve (or consciously fix)
     this asymmetry.
"""

import importlib

import numpy as np
import pytest

from confetti.errors import CONFETTIConfigurationError
from confetti.explainer._counterfactual_problem import CounterfactualProblem


# ---------------------------------------------------------------------------
# Problem attribute tests
# ---------------------------------------------------------------------------

def test_n_var_equals_L_times_F(problem_factory):
    p = problem_factory(subsequence_length=3)  # original_instance has 2 channels
    assert p.n_var == 3 * 2


_OBJ_COMBOS = [
    (True, True, True),    # n_obj = 3
    (True, True, False),   # n_obj = 2
    (True, False, True),   # n_obj = 2
    (False, True, True),   # n_obj = 2
    (True, False, False),  # n_obj = 1
    (False, True, False),  # n_obj = 1
    (False, False, True),  # n_obj = 1
]


@pytest.mark.parametrize("conf, spars, prox", _OBJ_COMBOS)
def test_n_obj_matches_flag_count(problem_factory, conf, spars, prox):
    p = problem_factory(
        optimize_confidence=conf,
        optimize_sparsity=spars,
        optimize_proximity=prox,
    )
    assert p.n_obj == int(conf) + int(spars) + int(prox)


def test_n_ieq_constr_is_one(problem_factory):
    p = problem_factory()
    assert p.n_ieq_constr == 1


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("conf, spars, prox", _OBJ_COMBOS)
def test_evaluate_F_shape(problem_factory, conf, spars, prox):
    p = problem_factory(
        optimize_confidence=conf, optimize_sparsity=spars, optimize_proximity=prox,
    )
    n_samples = 5
    x = np.zeros((n_samples, p.n_var), dtype=bool)
    out = {}
    p._evaluate(x, out)
    assert out["F"].shape == (n_samples, p.n_obj)


def test_evaluate_G_shape_quirk(problem_factory):
    """
    LOCKED QUIRK: the current implementation sets ``out["G"] = [self.theta - f1]``
    which, after ``np.asarray``, has shape ``(1, n_samples)`` — NOT ``(n_samples, 1)``
    as ``n_ieq_constr=1`` would imply. Pymoo tolerates this via broadcasting.
    Re-implementations must preserve this shape (or explicitly decide to fix it).
    """
    p = problem_factory()
    n_samples = 7
    x = np.zeros((n_samples, p.n_var), dtype=bool)
    out = {}
    p._evaluate(x, out)

    G = np.asarray(out["G"])
    assert G.shape == (1, n_samples)


# ---------------------------------------------------------------------------
# Counterfactual-construction tests (mask semantics)
# ---------------------------------------------------------------------------

def _build_counterfactual(original, nun, mask_reshaped, start, L):
    """Reference impl of the CF construction (used to cross-check _evaluate)."""
    N = mask_reshaped.shape[0]
    cf = np.tile(original, (N, 1, 1))
    patch = nun[start:start + L]
    cf[:, start:start + L, :] = np.where(mask_reshaped, patch[None, :, :], cf[:, start:start + L, :])
    return cf


def test_all_zeros_mask_yields_original(problem_factory, original_instance):
    p = problem_factory()
    n_samples = 4
    x = np.zeros((n_samples, p.n_var), dtype=bool)
    out = {}
    p._evaluate(x, out)

    # Sparsity should be exactly 0 for all samples.
    # Column layout: (conf, spars, prox) with conf and spars both on → F cols: [f1, f2, f3]
    f2 = out["F"][:, 1]
    np.testing.assert_allclose(f2, np.zeros(n_samples))

    # Proximity to the original should be 0 because the CF equals the original.
    f3 = out["F"][:, 2]
    np.testing.assert_allclose(f3, np.zeros(n_samples), atol=1e-12)


def test_all_ones_mask_replaces_subsequence(problem_factory, original_instance, nun_instance):
    p = problem_factory(start_timestep=2, subsequence_length=3)
    n_samples = 2
    x = np.ones((n_samples, p.n_var), dtype=bool)
    out = {}
    p._evaluate(x, out)

    # Sparsity = sum(mask) / (L*F) = (L*F)/(L*F) = 1.0
    np.testing.assert_allclose(out["F"][:, 1], np.ones(n_samples))

    # Rebuild the expected CF and compare proximity (euclidean) manually
    mask_reshaped = x.reshape(n_samples, 3, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, 2, 3)
    expected_prox = np.sqrt(((cf - original_instance[None]) ** 2).sum(axis=(1, 2)))
    np.testing.assert_allclose(out["F"][:, 2], expected_prox, atol=1e-12)


def test_sparsity_matches_manual_formula(problem_factory):
    p = problem_factory(subsequence_length=3)  # n_var = 3 * 2 = 6
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, size=(5, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    expected = x.sum(axis=1) / p.n_var
    np.testing.assert_allclose(out["F"][:, 1], expected)


def test_partial_mask_only_affects_subsequence_window(
    problem_factory, original_instance, nun_instance
):
    """
    Outside the ``[start, start+L)`` subsequence, the counterfactual must equal
    the original instance, regardless of the mask. We can't observe the full CF
    directly through the public API, but ``_evaluate`` with sparsity-only off
    and proximity-only on lets us reason about it via the euclidean distance.
    Instead, replicate the construction manually and check invariants.
    """
    start, L = 3, 4
    p = problem_factory(start_timestep=start, subsequence_length=L)
    n_samples = 3
    x = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert x.shape == (n_samples, p.n_var)
    mask_reshaped = x.reshape(n_samples, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)

    # Outside the window
    np.testing.assert_array_equal(
        cf[:, :start, :], np.tile(original_instance[:start], (n_samples, 1, 1))
    )
    np.testing.assert_array_equal(
        cf[:, start + L:, :], np.tile(original_instance[start + L:], (n_samples, 1, 1))
    )


# ---------------------------------------------------------------------------
# Proximity tests
# ---------------------------------------------------------------------------

def test_euclidean_proximity_matches_numpy(problem_factory, original_instance, nun_instance):
    p = problem_factory(start_timestep=1, subsequence_length=4)
    rng = np.random.default_rng(7)
    x = rng.integers(0, 2, size=(6, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(6, 4, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, 1, 4)
    expected = np.sqrt(((cf - original_instance[None]) ** 2).sum(axis=(1, 2)))
    np.testing.assert_allclose(out["F"][:, 2], expected, atol=1e-12)


def test_proximity_zero_for_zero_mask(problem_factory):
    p = problem_factory()
    x = np.zeros((3, p.n_var), dtype=bool)
    out = {}
    p._evaluate(x, out)
    np.testing.assert_allclose(out["F"][:, 2], np.zeros(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Constraint + confidence-sign tests (the locked quirks)
# ---------------------------------------------------------------------------

def test_constraint_G_equals_theta_minus_confidence(
    problem_factory, original_instance, nun_instance, reference_labels, mock_classifier_biased
):
    """G[i] must equal ``theta - classifier.predict(cf_i)[target_class]``."""
    theta = 0.6
    start, L = 2, 3
    nun_idx = 0
    target_class = reference_labels[nun_idx]

    p = problem_factory(theta=theta, start_timestep=start, subsequence_length=L, nun_index=nun_idx)
    rng = np.random.default_rng(11)
    x = rng.integers(0, 2, size=(4, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    # Compute expected confidence manually
    mask_reshaped = x.reshape(4, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    conf = mock_classifier_biased.predict(cf)[:, target_class]
    expected_G = theta - conf

    G = np.asarray(out["G"]).ravel()
    np.testing.assert_allclose(G, expected_G, atol=1e-6)


def test_confidence_sign_when_only_confidence_active(
    problem_factory, original_instance, nun_instance, reference_labels, mock_classifier_biased
):
    """When only confidence is on, F column 0 == -confidence (minimization-friendly)."""
    start, L = 1, 3
    p = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=False,
        optimize_proximity=False,
        start_timestep=start,
        subsequence_length=L,
    )
    rng = np.random.default_rng(17)
    x = rng.integers(0, 2, size=(5, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(5, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    target_class = reference_labels[0]
    conf = mock_classifier_biased.predict(cf)[:, target_class]

    np.testing.assert_allclose(out["F"][:, 0], -conf, atol=1e-6)


def test_confidence_sign_when_confidence_and_sparsity_active(
    problem_factory, original_instance, nun_instance, reference_labels, mock_classifier_biased
):
    """
    LOCKED QUIRK: when confidence AND sparsity are both on, F column 0 is the
    raw (positive) confidence — NOT negated. This pins the asymmetry at
    ``_counterfactual_problem.py:140-147``. The reimplementation must match
    or explicitly decide to fix it.
    """
    start, L = 1, 3
    p = problem_factory(
        optimize_confidence=True,
        optimize_sparsity=True,
        optimize_proximity=False,
        start_timestep=start,
        subsequence_length=L,
    )
    rng = np.random.default_rng(19)
    x = rng.integers(0, 2, size=(5, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(5, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    target_class = reference_labels[0]
    conf = mock_classifier_biased.predict(cf)[:, target_class]

    # QUIRK: +conf, not -conf
    np.testing.assert_allclose(out["F"][:, 0], conf, atol=1e-6)


# ---------------------------------------------------------------------------
# Dtype-agnostic masking
# ---------------------------------------------------------------------------

def test_dtype_agnostic_mask(problem_factory):
    """``_evaluate`` must yield identical F / G for bool, int8, int64 masks."""
    p = problem_factory()
    rng = np.random.default_rng(23)
    base = rng.integers(0, 2, size=(4, p.n_var), dtype=np.int64)

    results = []
    for dtype in (bool, np.int8, np.int64):
        x = base.astype(dtype)
        out = {}
        p._evaluate(x, out)
        results.append((out["F"].copy(), np.asarray(out["G"]).copy()))

    F0, G0 = results[0]
    for F, G in results[1:]:
        np.testing.assert_allclose(F, F0, atol=1e-12)
        np.testing.assert_allclose(G, G0, atol=1e-12)


# ---------------------------------------------------------------------------
# tslearn-gated proximity tests
# ---------------------------------------------------------------------------

def test_dtw_proximity_matches_tslearn(problem_factory, original_instance, nun_instance):
    tslearn_metrics = pytest.importorskip("tslearn.metrics")

    start, L = 2, 3
    p = problem_factory(
        proximity_distance="dtw",
        start_timestep=start,
        subsequence_length=L,
        optimize_proximity=True,
    )
    rng = np.random.default_rng(29)
    x = rng.integers(0, 2, size=(3, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(3, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    expected = tslearn_metrics.cdist_dtw(cf, original_instance[None, :, :]).ravel()
    # proximity column is last
    np.testing.assert_allclose(out["F"][:, -1], expected, atol=1e-6)


def test_dtw_with_sakoe_chiba_window(problem_factory, original_instance, nun_instance):
    tslearn_metrics = pytest.importorskip("tslearn.metrics")

    start, L = 1, 4
    p = problem_factory(
        proximity_distance="dtw",
        dtw_window=2,
        start_timestep=start,
        subsequence_length=L,
        optimize_proximity=True,
    )
    rng = np.random.default_rng(31)
    x = rng.integers(0, 2, size=(2, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(2, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    expected = tslearn_metrics.cdist_dtw(
        cf, original_instance[None, :, :], global_constraint="sakoe_chiba", sakoe_chiba_radius=2,
    ).ravel()
    np.testing.assert_allclose(out["F"][:, -1], expected, atol=1e-6)


def test_unsupported_metric_raises_configuration_error(problem_factory):
    pytest.importorskip("tslearn.metrics")
    p = problem_factory(proximity_distance="not-a-real-metric", optimize_proximity=True)
    x = np.zeros((1, p.n_var), dtype=bool)
    out = {}
    with pytest.raises(CONFETTIConfigurationError):
        p._evaluate(x, out)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

def test_single_sample_evaluation(problem_factory):
    """``n_samples=1`` must not trigger off-by-one errors in reshaping or broadcasting."""
    p = problem_factory()
    x = np.zeros((1, p.n_var), dtype=bool)
    out = {}
    p._evaluate(x, out)
    assert out["F"].shape == (1, p.n_obj)
    G = np.asarray(out["G"])
    assert G.ravel().shape[0] == 1


def test_subsequence_length_one(problem_factory):
    """``subsequence_length=1`` is the minimal patch size (``n_var = 1 × channels``)."""
    p = problem_factory(subsequence_length=1, start_timestep=0)
    assert p.n_var == 2  # 1 timestep × 2 channels
    rng = np.random.default_rng(33)
    x = rng.integers(0, 2, size=(4, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)
    assert out["F"].shape == (4, p.n_obj)
    expected_sparsity = x.sum(axis=1) / p.n_var
    np.testing.assert_allclose(out["F"][:, 1], expected_sparsity)


def test_ctw_proximity_matches_tslearn(problem_factory, original_instance, nun_instance):
    tslearn_metrics = pytest.importorskip("tslearn.metrics")

    start, L = 2, 3
    p = problem_factory(
        proximity_distance="ctw",
        start_timestep=start,
        subsequence_length=L,
        optimize_proximity=True,
    )
    rng = np.random.default_rng(35)
    x = rng.integers(0, 2, size=(3, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(3, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    expected = tslearn_metrics.cdist_ctw(cf, original_instance[None, :, :]).ravel()
    np.testing.assert_allclose(out["F"][:, -1], expected, atol=1e-6)


def test_softdtw_proximity_matches_tslearn(problem_factory, original_instance, nun_instance):
    tslearn_metrics = pytest.importorskip("tslearn.metrics")

    start, L = 2, 3
    p = problem_factory(
        proximity_distance="softdtw",
        start_timestep=start,
        subsequence_length=L,
        optimize_proximity=True,
    )
    rng = np.random.default_rng(37)
    x = rng.integers(0, 2, size=(3, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(3, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    expected = tslearn_metrics.cdist_soft_dtw(cf, original_instance[None, :, :]).ravel()
    np.testing.assert_allclose(out["F"][:, -1], expected, atol=1e-4)


def test_gak_proximity_matches_tslearn(problem_factory, original_instance, nun_instance):
    tslearn_metrics = pytest.importorskip("tslearn.metrics")

    start, L = 2, 3
    p = problem_factory(
        proximity_distance="gak",
        start_timestep=start,
        subsequence_length=L,
        optimize_proximity=True,
    )
    rng = np.random.default_rng(39)
    x = rng.integers(0, 2, size=(3, p.n_var), dtype=np.int64).astype(bool)
    out = {}
    p._evaluate(x, out)

    mask_reshaped = x.reshape(3, L, 2)
    cf = _build_counterfactual(original_instance, nun_instance, mask_reshaped, start, L)
    expected = tslearn_metrics.cdist_gak(cf, original_instance[None, :, :]).ravel()
    np.testing.assert_allclose(out["F"][:, -1], expected, atol=1e-4)
