"""
Fixtures for the genetic-algorithm test suite.

These tests lock in the current pymoo-based NSGA-III behavior as a specification
for a future hand-rolled (then Rust / PyO3) re-implementation. All fixtures are
numpy-only and deterministic — no Keras/TensorFlow is required.
"""

import numpy as np
import pytest

from confetti.explainer._counterfactual_problem import CounterfactualProblem


# ---------------------------------------------------------------------------
# Deterministic time-series inputs
# ---------------------------------------------------------------------------

@pytest.fixture
def original_instance():
    """Deterministic (T=8, C=2) original time series."""
    return (np.arange(16, dtype=float).reshape(8, 2) / 16.0)


@pytest.fixture
def nun_instance():
    """Deterministic (T=8, C=2) NUN that is visibly different from the original."""
    return np.full((8, 2), 2.0, dtype=float)


@pytest.fixture
def reference_labels():
    """Labels for the reference dataset. Index 0 maps to target class 1."""
    return np.array([1, 0, 1, 0], dtype=int)


# ---------------------------------------------------------------------------
# Mock classifiers
# ---------------------------------------------------------------------------

class _BiasedClassifier:
    """
    Deterministic mock classifier.

    Returns ``(N, 2)`` float32 probabilities where class 1 becomes more likely
    as the input mean grows. Because ``nun_instance`` has larger values than
    ``original_instance``, masks that turn on more bits push the counterfactual
    toward class 1 — which is exactly what the CounterfactualProblem expects
    to happen when it tries to flip the predicted label.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=(1, 2))
        # Logistic-like mapping; range of mean is roughly [0.5, 2.0] for our
        # fixtures, so the output spans roughly [0.33, 0.88].
        p1 = 1.0 / (1.0 + np.exp(-(mean - 1.0) * 2.0))
        p1 = np.clip(p1, 0.01, 0.99)
        return np.vstack([1.0 - p1, p1]).T.astype(np.float32)


class _WeakClassifier:
    """Always predicts an almost-uniform distribution. Used to trigger infeasibility."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        return np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (N, 1))


@pytest.fixture
def mock_classifier_biased():
    return _BiasedClassifier()


@pytest.fixture
def mock_classifier_weak():
    return _WeakClassifier()


# ---------------------------------------------------------------------------
# CounterfactualProblem factory
# ---------------------------------------------------------------------------

@pytest.fixture
def problem_factory(original_instance, nun_instance, reference_labels, mock_classifier_biased):
    """
    Return a factory that builds a ``CounterfactualProblem`` with sensible
    defaults but allows overriding individual knobs.
    """

    def _make(
        *,
        classifier=None,
        optimize_confidence=True,
        optimize_sparsity=True,
        optimize_proximity=True,
        proximity_distance="euclidean",
        dtw_window=None,
        theta=0.5,
        start_timestep=2,
        subsequence_length=3,
        nun_index=0,
    ):
        return CounterfactualProblem(
            original_instance=original_instance,
            nun_instance=nun_instance,
            nun_index=nun_index,
            start_timestep=start_timestep,
            subsequence_length=subsequence_length,
            classifier=classifier if classifier is not None else mock_classifier_biased,
            reference_labels=reference_labels,
            optimize_confidence=optimize_confidence,
            optimize_sparsity=optimize_sparsity,
            optimize_proximity=optimize_proximity,
            proximity_distance=proximity_distance,
            dtw_window=dtw_window,
            theta=theta,
        )

    return _make


# ---------------------------------------------------------------------------
# Raw binary populations for operator tests
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_population():
    """Deterministic binary population of shape (20, 16)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(20, 16), dtype=np.int64).astype(bool)


class _DummyBinaryProblem:
    """
    Minimal stand-in for a pymoo Problem, used only when operators need to
    introspect ``n_var`` or a similar attribute. We do not want a real
    CounterfactualProblem here — the GA operators should be decoupled from it.
    """

    def __init__(self, n_var: int):
        self.n_var = n_var
        self.xl = 0
        self.xu = 1


@pytest.fixture
def dummy_problem_factory():
    return _DummyBinaryProblem
