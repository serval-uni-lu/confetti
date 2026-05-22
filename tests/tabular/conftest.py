from __future__ import annotations

import numpy as np
import pandas as pd
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


class MockNoInterfaceModel:
    """Model with no predict interface at all."""
    pass


@pytest.fixture
def binary_classifier():
    return MockBinaryClassifier()


@pytest.fixture
def predict_only_classifier():
    return MockPredictOnlyClassifier()


@pytest.fixture
def numeric_df():
    return pd.DataFrame({
        "age": [25, 30, 50, 60, 70, 80],
        "income": [20000, 35000, 60000, 75000, 80000, 90000],
    })


@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "age": [25, 30, 50, 60],
        "city": ["NYC", "LA", "NYC", "LA"],
        "income": [20000, 35000, 60000, 75000],
    })


@pytest.fixture
def numeric_reference_np():
    return np.array([
        [25, 20000],
        [30, 35000],
        [50, 60000],
        [60, 75000],
        [70, 80000],
        [80, 90000],
    ], dtype=np.float64)
