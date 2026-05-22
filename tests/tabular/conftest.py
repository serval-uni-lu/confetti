from __future__ import annotations

import numpy as np
import pytest


class MockBinaryClassifier:
    """Deterministic mock classifier for testing.

    Returns class-0 probability proportional to the sum of features.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = np.clip(np.mean(X, axis=1) / 100, 0.0, 1.0)
        return np.column_stack([1 - score, score])


class MockPredictOnlyClassifier:
    """Mock that only exposes ``predict`` (keras-style), not ``predict_proba``."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = np.clip(np.mean(X, axis=1) / 100, 0.0, 1.0)
        return np.column_stack([1 - score, score])


@pytest.fixture
def binary_classifier():
    return MockBinaryClassifier()


@pytest.fixture
def predict_only_classifier():
    return MockPredictOnlyClassifier()


class MockScaledClassifier:
    """Classifier that expects input pre-scaled by 2x.

    Returns class-1 probability via sigmoid on the mean of (X / 2) — if the
    input is already doubled by the preprocessor the division cancels out and
    the decision boundary sits at mean == 50.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = 1 / (1 + np.exp(-(np.mean(X / 2, axis=1) - 50) / 10))
        return np.column_stack([1 - score, score])


class MockScaledPredictOnlyClassifier:
    """Predict-only variant of ``MockScaledClassifier``."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = 1 / (1 + np.exp(-(np.mean(X / 2, axis=1) - 50) / 10))
        return np.column_stack([1 - score, score])


@pytest.fixture
def mock_preprocessor():
    """Preprocessor that doubles every value."""
    return lambda X: X * 2


@pytest.fixture
def scaled_classifier():
    return MockScaledClassifier()


@pytest.fixture
def scaled_predict_only_classifier():
    return MockScaledPredictOnlyClassifier()
