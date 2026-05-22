from __future__ import annotations

import numpy as np
import pytest

from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError
from confetti.tabular._tabular_problem import TabularCounterfactualProblem


class MockClassifier:
    """Classifier that returns score proportional to feature mean."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = np.clip(np.mean(X, axis=1) / 100, 0.0, 1.0)
        return np.column_stack([1 - score, score])


@pytest.fixture
def original():
    return np.array([10.0, 20.0, 30.0, 40.0])


@pytest.fixture
def nun():
    return np.array([80.0, 90.0, 70.0, 60.0])


@pytest.fixture
def reference_labels():
    return np.array([0, 1, 0, 1])


@pytest.fixture
def classifier():
    return MockClassifier()


@pytest.fixture
def problem(original, nun, classifier, reference_labels):
    return TabularCounterfactualProblem(
        original_instance=original,
        nun_instance=nun,
        nun_index=1,
        classifier=classifier,
        reference_labels=reference_labels,
    )


class TestMaskApplication:

    def test_all_zeros_returns_original(self, problem, original):
        x = np.zeros((1, 4))
        out = {}
        problem._evaluate(x, out)
        cf = np.where(x, problem.nun_instance, problem.original_instance)
        np.testing.assert_array_equal(cf[0], original)

    def test_all_ones_returns_nun(self, problem, nun):
        x = np.ones((1, 4))
        out = {}
        problem._evaluate(x, out)
        cf = np.where(x, problem.nun_instance, problem.original_instance)
        np.testing.assert_array_equal(cf[0], nun)

    def test_partial_mask(self, problem, original, nun):
        x = np.array([[1, 0, 1, 0]])
        cf = np.where(x, problem.nun_instance, problem.original_instance)
        expected = np.array([nun[0], original[1], nun[2], original[3]])
        np.testing.assert_array_equal(cf[0], expected)


class TestObjectiveShapes:

    def test_two_objectives_shape(self, problem):
        x = np.random.randint(0, 2, (10, 4)).astype(float)
        out = {}
        problem._evaluate(x, out)
        assert out["F"].shape == (10, 2)
        g = np.array(out["G"])
        assert g.shape == (1, 10)

    def test_three_objectives_shape(self, original, nun, classifier, reference_labels):
        prob = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=reference_labels,
            optimize_proximity=True,
        )
        x = np.random.randint(0, 2, (5, 4)).astype(float)
        out = {}
        prob._evaluate(x, out)
        assert out["F"].shape == (5, 3)


class TestSparsity:

    def test_all_zeros_sparsity_is_zero(self, problem):
        x = np.zeros((1, 4))
        out = {}
        problem._evaluate(x, out)
        assert out["F"][0, 1] == 0.0

    def test_all_ones_sparsity_is_one(self, problem):
        x = np.ones((1, 4))
        out = {}
        problem._evaluate(x, out)
        assert out["F"][0, 1] == 1.0

    def test_half_mask_sparsity(self, problem):
        x = np.array([[1, 1, 0, 0]], dtype=float)
        out = {}
        problem._evaluate(x, out)
        assert out["F"][0, 1] == pytest.approx(0.5)


class TestConstraint:

    def test_constraint_shape(self, problem):
        x = np.zeros((3, 4))
        out = {}
        problem._evaluate(x, out)
        g = np.array(out["G"])
        assert g.shape[1] == 3


class TestProximity:

    def test_euclidean_proximity(self, original, nun, classifier, reference_labels):
        prob = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=reference_labels,
            optimize_proximity=True,
            proximity_distance="euclidean",
        )
        x = np.ones((1, 4))
        out = {}
        prob._evaluate(x, out)
        expected_dist = np.sqrt(np.sum((nun - original) ** 2))
        assert out["F"][0, 2] == pytest.approx(expected_dist)

    def test_manhattan_proximity(self, original, nun, classifier, reference_labels):
        prob = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=reference_labels,
            optimize_proximity=True,
            proximity_distance="manhattan",
        )
        x = np.ones((1, 4))
        out = {}
        prob._evaluate(x, out)
        expected_dist = np.sum(np.abs(nun - original))
        assert out["F"][0, 2] == pytest.approx(expected_dist)

    def test_zero_mask_proximity_is_zero(self, original, nun, classifier, reference_labels):
        prob = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=reference_labels,
            optimize_proximity=True,
        )
        x = np.zeros((1, 4))
        out = {}
        prob._evaluate(x, out)
        assert out["F"][0, 2] == pytest.approx(0.0)


class TestConfidenceSignQuirk:

    def test_confidence_positive_when_both_objectives(self, problem):
        x = np.ones((1, 4))
        out = {}
        problem._evaluate(x, out)
        assert out["F"][0, 0] > 0

    def test_confidence_negated_when_only_confidence(self, original, nun, classifier, reference_labels):
        prob = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=reference_labels,
            optimize_confidence=True,
            optimize_sparsity=False,
            optimize_proximity=True,
        )
        x = np.ones((1, 4))
        out = {}
        prob._evaluate(x, out)
        assert out["F"][0, 0] < 0


class TestValidation:

    def test_2d_original_raises(self, nun, classifier, reference_labels):
        with pytest.raises(CONFETTIDataTypeError, match="1-D"):
            TabularCounterfactualProblem(
                original_instance=np.array([[1, 2]]),
                nun_instance=nun,
                nun_index=1,
                classifier=classifier,
                reference_labels=reference_labels,
            )

    def test_2d_nun_raises(self, original, classifier, reference_labels):
        with pytest.raises(CONFETTIDataTypeError, match="1-D"):
            TabularCounterfactualProblem(
                original_instance=original,
                nun_instance=np.array([[1, 2]]),
                nun_index=1,
                classifier=classifier,
                reference_labels=reference_labels,
            )

    def test_shape_mismatch_raises(self, classifier, reference_labels):
        with pytest.raises(CONFETTIConfigurationError, match="Shape mismatch"):
            TabularCounterfactualProblem(
                original_instance=np.array([1.0, 2.0]),
                nun_instance=np.array([1.0, 2.0, 3.0]),
                nun_index=1,
                classifier=classifier,
                reference_labels=reference_labels,
            )

    def test_unsupported_metric_raises(self, original, nun, classifier, reference_labels):
        with pytest.raises(CONFETTIConfigurationError, match="Unsupported"):
            TabularCounterfactualProblem(
                original_instance=original,
                nun_instance=nun,
                nun_index=1,
                classifier=classifier,
                reference_labels=reference_labels,
                optimize_proximity=True,
                proximity_distance="dtw",
            )

    def test_fewer_than_two_objectives_raises(self, original, nun, classifier, reference_labels):
        with pytest.raises(CONFETTIConfigurationError, match="two objectives"):
            TabularCounterfactualProblem(
                original_instance=original,
                nun_instance=nun,
                nun_index=1,
                classifier=classifier,
                reference_labels=reference_labels,
                optimize_confidence=True,
                optimize_sparsity=False,
                optimize_proximity=False,
            )

    def test_theta_out_of_range_raises(self, original, nun, classifier, reference_labels):
        with pytest.raises(CONFETTIConfigurationError, match="theta"):
            TabularCounterfactualProblem(
                original_instance=original,
                nun_instance=nun,
                nun_index=1,
                classifier=classifier,
                reference_labels=reference_labels,
                theta=1.5,
            )

    def test_theta_zero_raises(self, original, nun, classifier, reference_labels):
        with pytest.raises(CONFETTIConfigurationError, match="theta"):
            TabularCounterfactualProblem(
                original_instance=original,
                nun_instance=nun,
                nun_index=1,
                classifier=classifier,
                reference_labels=reference_labels,
                theta=0.0,
            )
