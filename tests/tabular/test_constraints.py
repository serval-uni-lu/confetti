"""Tests for constraint AST, evaluator, and GA integration."""

from __future__ import annotations

import numpy as np
import pytest

from confetti.constraints import (
    And,
    Constant,
    ConstraintEvaluator,
    Count,
    Equal,
    Feature,
    Less,
    LessEqual,
    Log,
    ManySum,
    MathOperation,
    Or,
    SafeDivision,
    repair_equality_constraints,
)
from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError
from confetti.tabular._tabular_problem import TabularCounterfactualProblem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockClassifier:
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = np.clip(np.mean(X, axis=1) / 100, 0.0, 1.0)
        return np.column_stack([1 - score, score])


@pytest.fixture
def classifier():
    return MockClassifier()


# ---------------------------------------------------------------------------
# T-1: AST construction
# ---------------------------------------------------------------------------


class TestASTConstruction:
    def test_le_builds_less_equal(self):
        result = Feature(0) <= Feature(1)
        assert isinstance(result, LessEqual)

    def test_lt_builds_less(self):
        result = Feature(0) < Feature(1)
        assert isinstance(result, Less)

    def test_add_builds_many_sum(self):
        result = Feature("a") + Constant(1)
        assert isinstance(result, ManySum)
        assert len(result.operands) == 2

    def test_equal_direct_construction(self):
        result = Equal(Feature(0), Constant(3))
        assert isinstance(result, Equal)
        assert isinstance(result.left, Feature)
        assert isinstance(result.right, Constant)

    def test_or_operator(self):
        c1 = Feature(0) <= Feature(1)
        c2 = Feature(1) <= Feature(2)
        result = c1 | c2
        assert isinstance(result, Or)
        assert len(result.operands) == 2

    def test_and_operator(self):
        c1 = Feature(0) <= Feature(1)
        c2 = Feature(1) <= Feature(2)
        result = c1 & c2
        assert isinstance(result, And)
        assert len(result.operands) == 2

    def test_chained_add_accumulates(self):
        a, b, c = Feature(0), Feature(1), Feature(2)
        result = a + b + c
        assert isinstance(result, ManySum)
        assert len(result.operands) == 3

    def test_and_requires_two_operands(self):
        c = Feature(0) <= Feature(1)
        with pytest.raises(CONFETTIConfigurationError, match="at least 2"):
            And([c])

    def test_or_requires_two_operands(self):
        c = Feature(0) <= Feature(1)
        with pytest.raises(CONFETTIConfigurationError, match="at least 2"):
            Or([c])

    def test_safe_division_construction(self):
        sd = SafeDivision(Feature(0), Feature(1), Constant(0))
        assert isinstance(sd, SafeDivision)

    def test_log_construction(self):
        log = Log(Feature(0), safe_value=Constant(0))
        assert isinstance(log, Log)

    def test_complex_nested_expression(self):
        expr = (Feature(0) * Constant(2)) + Feature(1)
        constraint = expr <= Feature(2) ** Constant(0.5)
        assert isinstance(constraint, LessEqual)
        data = np.array([[1.0, 3.0, 25.0]])
        v = constraint.violation(data, None)
        assert v.shape == (1,)

    def test_sub_builds_math_op(self):
        result = Feature(0) - Constant(1)
        assert isinstance(result, MathOperation)
        assert result.operator == "-"

    def test_mul_builds_math_op(self):
        result = Feature(0) * Constant(2)
        assert isinstance(result, MathOperation)
        assert result.operator == "*"

    def test_truediv_builds_math_op(self):
        result = Feature(0) / Constant(2)
        assert isinstance(result, MathOperation)
        assert result.operator == "/"

    def test_pow_builds_math_op(self):
        result = Feature(0) ** Constant(2)
        assert isinstance(result, MathOperation)
        assert result.operator == "**"

    def test_mod_builds_math_op(self):
        result = Feature(0) % Constant(2)
        assert isinstance(result, MathOperation)
        assert result.operator == "%"

    def test_radd_with_numeric(self):
        result = 1 + Feature(0)
        assert isinstance(result, ManySum)

    def test_rsub_with_numeric(self):
        result = 10 - Feature(0)
        assert isinstance(result, MathOperation)
        assert result.operator == "-"

    def test_rmul_with_numeric(self):
        result = 2 * Feature(0)
        assert isinstance(result, MathOperation)
        assert result.operator == "*"

    def test_many_sum_requires_two_operands(self):
        with pytest.raises(CONFETTIConfigurationError, match="at least 2"):
            ManySum([Constant(1)])

    def test_count_requires_two_operands(self):
        c = Feature(0) <= Feature(1)
        with pytest.raises(CONFETTIConfigurationError, match="at least 2"):
            Count([c])


# ---------------------------------------------------------------------------
# T-2: Constraint evaluation
# ---------------------------------------------------------------------------


class TestConstraintEvaluation:
    def test_less_equal_satisfied(self):
        data = np.array([[3.0, 5.0]])
        c = Feature(0) <= Feature(1)
        assert c.violation(data, None)[0] == pytest.approx(0.0)

    def test_less_equal_violated(self):
        data = np.array([[7.0, 5.0]])
        c = Feature(0) <= Feature(1)
        assert c.violation(data, None)[0] == pytest.approx(2.0)

    def test_equal_with_tolerance_satisfied(self):
        data = np.array([[3.01, 3.0]])
        c = Equal(Feature(0), Feature(1), Constant(0.05))
        assert c.violation(data, None)[0] == pytest.approx(0.0)

    def test_equal_with_tolerance_violated(self):
        data = np.array([[3.1, 3.0]])
        c = Equal(Feature(0), Feature(1), Constant(0.05))
        assert c.violation(data, None)[0] == pytest.approx(0.05)

    def test_equal_without_tolerance(self):
        data = np.array([[3.5, 3.0]])
        c = Equal(Feature(0), Feature(1))
        assert c.violation(data, None)[0] == pytest.approx(0.5)

    def test_and_sums_violations(self):
        data = np.array([[7.0, 5.0, 2.0]])
        c1 = Feature(0) <= Feature(1)  # violation = 2
        c2 = Feature(2) <= Feature(1)  # violation = 0
        c = c1 & c2
        assert c.violation(data, None)[0] == pytest.approx(2.0)

    def test_or_takes_min(self):
        data = np.array([[7.0, 5.0, 10.0]])
        c1 = Feature(0) <= Feature(1)  # violation = 2
        c2 = Feature(0) <= Feature(2)  # violation = 0
        c = c1 | c2
        assert c.violation(data, None)[0] == pytest.approx(0.0)

    def test_batch_shape(self):
        data = np.random.default_rng(42).random((10, 3))
        c = Feature(0) <= Feature(1)
        result = c.violation(data, None)
        assert result.shape == (10,)

    def test_feature_by_name(self):
        data = np.array([[10.0, 20.0]])
        c = Feature("age") <= Feature("income")
        result = c.violation(data, ["age", "income"])
        assert result[0] == pytest.approx(0.0)

    def test_feature_name_not_found(self):
        with pytest.raises(CONFETTIConfigurationError, match="not found"):
            ConstraintEvaluator(
                Feature("unknown") <= Constant(1),
                feature_names=["age"],
            )

    def test_feature_index_out_of_range(self):
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        c = Feature(99) <= Constant(1)
        with pytest.raises(CONFETTIConfigurationError, match="out of range"):
            c.violation(data, None)

    def test_count_violated(self):
        data = np.array([[7.0, 5.0, 1.0]])
        c1 = Feature(0) <= Feature(1)  # violated (7 > 5)
        c2 = Feature(2) <= Feature(1)  # satisfied (1 <= 5)
        c3 = Feature(0) <= Feature(2)  # violated (7 > 1)
        count = Count([c1, c2, c3])
        assert count._eval(data, None)[0] == pytest.approx(2.0)

    def test_count_inverse(self):
        data = np.array([[7.0, 5.0, 1.0]])
        c1 = Feature(0) <= Feature(1)  # violated
        c2 = Feature(2) <= Feature(1)  # satisfied
        c3 = Feature(0) <= Feature(2)  # violated
        count = Count([c1, c2, c3], inverse=True)
        assert count._eval(data, None)[0] == pytest.approx(1.0)

    def test_math_operations_all(self):
        data = np.array([[10.0, 3.0]])
        assert (Feature(0) + Feature(1))._eval(data, None)[0] == pytest.approx(13.0)
        assert (Feature(0) - Feature(1))._eval(data, None)[0] == pytest.approx(7.0)
        assert (Feature(0) * Feature(1))._eval(data, None)[0] == pytest.approx(30.0)
        assert (Feature(0) / Feature(1))._eval(data, None)[0] == pytest.approx(10.0 / 3.0)
        assert (Feature(0) ** Constant(2))._eval(data, None)[0] == pytest.approx(100.0)
        assert (Feature(0) % Feature(1))._eval(data, None)[0] == pytest.approx(1.0)

    def test_safe_division_zero_divisor(self):
        data = np.array([[10.0, 0.0]])
        sd = SafeDivision(Feature(0), Feature(1), Constant(0))
        assert sd._eval(data, None)[0] == pytest.approx(0.0)

    def test_safe_division_nonzero_divisor(self):
        data = np.array([[10.0, 2.0]])
        sd = SafeDivision(Feature(0), Feature(1), Constant(-1))
        assert sd._eval(data, None)[0] == pytest.approx(5.0)

    def test_log_positive(self):
        data = np.array([[np.e]])
        log = Log(Feature(0))
        assert log._eval(data, None)[0] == pytest.approx(1.0)

    def test_log_nonpositive_with_safe_value(self):
        data = np.array([[-1.0]])
        log = Log(Feature(0), safe_value=Constant(0))
        assert log._eval(data, None)[0] == pytest.approx(0.0)

    def test_less_strict(self):
        data = np.array([[5.0, 5.0]])
        c = Feature(0) < Feature(1)
        assert c.violation(data, None)[0] > 0

    def test_less_satisfied(self):
        data = np.array([[3.0, 5.0]])
        c = Feature(0) < Feature(1)
        assert c.violation(data, None)[0] == pytest.approx(0.0, abs=1e-7)

    def test_evaluator_wraps_constraint(self):
        data = np.array([[3.0, 5.0]])
        ev = ConstraintEvaluator(Feature(0) <= Feature(1))
        result = ev.evaluate(data)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0)

    def test_constant_eval(self):
        data = np.array([[1.0, 2.0, 3.0]])
        c = Constant(42)
        assert c._eval(data, None)[0] == pytest.approx(42.0)
        assert c._eval(data, None).shape == (1,)


# ---------------------------------------------------------------------------
# T-3: GA integration
# ---------------------------------------------------------------------------


class TestGAIntegration:
    def test_relation_constraint_adds_g_column(self, classifier):
        original = np.array([10.0, 20.0, 30.0, 40.0])
        nun = np.array([80.0, 90.0, 70.0, 60.0])
        ref_labels = np.array([0, 1, 0, 1])

        constraint = Feature(0) <= Feature(1)

        problem = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=ref_labels,
            relation_constraints=[constraint],
        )

        assert problem.n_ieq_constr == 2

        x = np.zeros((10, 4))
        F, G = problem.evaluate(x)
        assert G.shape == (10, 2)

    def test_no_constraint_preserves_original_g(self, classifier):
        original = np.array([10.0, 20.0, 30.0, 40.0])
        nun = np.array([80.0, 90.0, 70.0, 60.0])
        ref_labels = np.array([0, 1, 0, 1])

        problem = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=ref_labels,
        )

        assert problem.n_ieq_constr == 1

        x = np.zeros((10, 4))
        F, G = problem.evaluate(x)
        assert G.shape == (10, 1)

    def test_violated_constraint_produces_positive_g(self, classifier):
        original = np.array([10.0, 20.0, 30.0, 40.0])
        nun = np.array([80.0, 5.0, 70.0, 60.0])
        ref_labels = np.array([0, 1, 0, 1])

        # Feature(0) <= Feature(1) will be violated when NUN[0]=80 > NUN[1]=5
        constraint = Feature(0) <= Feature(1)

        problem = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=ref_labels,
            relation_constraints=[constraint],
        )

        x = np.ones((1, 4))  # swap all features to NUN values
        _, G = problem.evaluate(x)
        assert G[0, 1] > 0  # relation constraint violation is positive


# ---------------------------------------------------------------------------
# T-4: Equality repair
# ---------------------------------------------------------------------------


class TestEqualityRepair:
    def test_repair_sets_feature_from_expression(self):
        counterfactuals = np.array([[3.0, 4.0, 0.0]])
        constraints = [Equal(Feature(2), Feature(0) + Feature(1))]
        repair_equality_constraints(counterfactuals, constraints)
        assert counterfactuals[0, 2] == pytest.approx(7.0)

    def test_repair_skips_non_feature_left(self):
        counterfactuals = np.array([[3.0, 4.0, 0.0]])
        constraints = [Equal(Feature(0) + Feature(1), Constant(10))]
        original = counterfactuals.copy()
        repair_equality_constraints(counterfactuals, constraints)
        np.testing.assert_array_equal(counterfactuals, original)

    def test_repair_chained_equalities(self):
        counterfactuals = np.array([[3.0, 0.0, 0.0]])
        constraints = [
            Equal(Feature(1), Feature(0) * Constant(2)),  # x[1] = 6
            Equal(Feature(2), Feature(0) + Feature(1)),    # x[2] = 3 + 6 = 9
        ]
        repair_equality_constraints(counterfactuals, constraints)
        assert counterfactuals[0, 1] == pytest.approx(6.0)
        assert counterfactuals[0, 2] == pytest.approx(9.0)

    def test_repair_in_evaluate(self, classifier):
        original = np.array([10.0, 20.0, 30.0])
        nun = np.array([50.0, 60.0, 0.0])
        ref_labels = np.array([0, 1, 0, 1])

        # Feature(2) should equal Feature(0) + Feature(1)
        constraint = Equal(Feature(2), Feature(0) + Feature(1))

        problem = TabularCounterfactualProblem(
            original_instance=original,
            nun_instance=nun,
            nun_index=1,
            classifier=classifier,
            reference_labels=ref_labels,
            relation_constraints=[constraint],
        )

        x = np.ones((1, 3))  # swap all → NUN values [50, 60, 0]
        # After repair: feature 2 = 50 + 60 = 110
        # Constraint violation should be 0 after repair
        _, G = problem.evaluate(x)
        assert G[0, 1] == pytest.approx(0.0)

    def test_repair_with_feature_names(self):
        counterfactuals = np.array([[3.0, 4.0, 0.0]])
        constraints = [Equal(Feature("c"), Feature("a") + Feature("b"))]
        repair_equality_constraints(counterfactuals, constraints, feature_names=["a", "b", "c"])
        assert counterfactuals[0, 2] == pytest.approx(7.0)

    def test_repair_nested_in_and(self):
        counterfactuals = np.array([[3.0, 4.0, 0.0]])
        constraints = [And([
            Equal(Feature(2), Feature(0) + Feature(1)),
            Feature(0) <= Feature(1),
        ])]
        repair_equality_constraints(counterfactuals, constraints)
        assert counterfactuals[0, 2] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_relation_constraint_type_in_problem(self, classifier):
        original = np.array([10.0, 20.0])
        nun = np.array([80.0, 90.0])
        ref_labels = np.array([0, 1])

        with pytest.raises(Exception):
            TabularCounterfactualProblem(
                original_instance=original,
                nun_instance=nun,
                nun_index=1,
                classifier=classifier,
                reference_labels=ref_labels,
                relation_constraints=["not a constraint"],
            )

    def test_feature_name_validation_at_evaluator_init(self):
        with pytest.raises(CONFETTIConfigurationError, match="not found"):
            ConstraintEvaluator(
                Feature("missing") <= Constant(5),
                feature_names=["a", "b"],
            )

    def test_feature_name_requires_names(self):
        with pytest.raises(CONFETTIConfigurationError, match="no feature_names"):
            ConstraintEvaluator(
                Feature("x") <= Constant(5),
                feature_names=None,
            )
