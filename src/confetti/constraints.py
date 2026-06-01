"""Constraint expression AST and evaluator for inter-feature relational constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Value(ABC):
    """Base class for all expression nodes in the constraint AST."""

    @abstractmethod
    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        """Evaluate this node against a batch of samples.

        Parameters
        ----------
        ``data`` : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        ``names`` : list[str] or None
            Feature names for resolving string-based ``Feature`` references.

        Returns
        -------
        np.ndarray
            Evaluated values of shape ``(n_samples,)``.
        """

    def __add__(self, other: Value | int | float) -> ManySum:
        other = _coerce(other)
        if isinstance(self, ManySum):
            return ManySum(operands=[*self.operands, other])
        return ManySum(operands=[self, other])

    def __radd__(self, other: Value | int | float) -> ManySum:
        other = _coerce(other)
        if isinstance(self, ManySum):
            return ManySum(operands=[other, *self.operands])
        return ManySum(operands=[other, self])

    def __sub__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="-", left=self, right=_coerce(other))

    def __rsub__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="-", left=_coerce(other), right=self)

    def __mul__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="*", left=self, right=_coerce(other))

    def __rmul__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="*", left=_coerce(other), right=self)

    def __truediv__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="/", left=self, right=_coerce(other))

    def __rtruediv__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="/", left=_coerce(other), right=self)

    def __pow__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="**", left=self, right=_coerce(other))

    def __rpow__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="**", left=_coerce(other), right=self)

    def __mod__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="%", left=self, right=_coerce(other))

    def __rmod__(self, other: Value | int | float) -> MathOperation:
        return MathOperation(operator="%", left=_coerce(other), right=self)

    def __le__(self, other: Value | int | float) -> LessEqual:
        return LessEqual(left=self, right=_coerce(other))

    def __lt__(self, other: Value | int | float) -> Less:
        return Less(left=self, right=_coerce(other))


class Constant(Value):
    """A literal numeric constant.

    Parameters
    ----------
    ``value`` : int or float
        The constant value.
    """

    def __init__(self, value: int | float) -> None:
        self.value = value

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        return np.full(data.shape[0], self.value, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Constant({self.value!r})"


class Feature(Value):
    """A reference to a feature column by name or index.

    Parameters
    ----------
    ``feature_id`` : str or int
        Column name or zero-based index.
    """

    def __init__(self, feature_id: str | int) -> None:
        self.feature_id = feature_id

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        idx = _resolve_feature(self.feature_id, names, data.shape[1])
        return data[:, idx].astype(np.float64)

    def __repr__(self) -> str:
        return f"Feature({self.feature_id!r})"


_VALID_OPS = frozenset({"+", "-", "*", "/", "**", "%"})


class MathOperation(Value):
    """Binary arithmetic operation between two value nodes.

    Parameters
    ----------
    ``operator`` : str
        One of ``"+"``, ``"-"``, ``"*"``, ``"/"``, ``"**"``, ``"%"``.
    ``left`` : Value
        Left operand.
    ``right`` : Value
        Right operand.
    """

    def __init__(self, operator: str, left: Value, right: Value) -> None:
        if operator not in _VALID_OPS:
            raise ValueError(f"Unsupported operator '{operator}'. Must be one of {sorted(_VALID_OPS)}.")
        self.operator = operator
        self.left = left
        self.right = right

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        lv = self.left._eval(data, names)
        rv = self.right._eval(data, names)
        op = self.operator
        if op == "+":
            return lv + rv
        if op == "-":
            return lv - rv
        if op == "*":
            return lv * rv
        if op == "/":
            return lv / rv
        if op == "**":
            return lv**rv
        return lv % rv

    def __repr__(self) -> str:
        return f"MathOperation({self.operator!r}, {self.left!r}, {self.right!r})"


class SafeDivision(Value):
    """Division with a fallback value when the divisor is zero.

    Parameters
    ----------
    ``dividend`` : Value
        Numerator.
    ``divisor`` : Value
        Denominator.
    ``fill_value`` : Value
        Value used when divisor is zero.
    """

    def __init__(self, dividend: Value, divisor: Value, fill_value: Value) -> None:
        self.dividend = dividend
        self.divisor = divisor
        self.fill_value = fill_value

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        num = self.dividend._eval(data, names)
        den = self.divisor._eval(data, names)
        fill = self.fill_value._eval(data, names)
        mask = den != 0
        result = np.where(mask, np.where(mask, num / np.where(mask, den, 1.0), 0.0), fill)
        return result

    def __repr__(self) -> str:
        return f"SafeDivision({self.dividend!r}, {self.divisor!r}, {self.fill_value!r})"


class ManySum(Value):
    """N-ary sum of value nodes.

    Parameters
    ----------
    ``operands`` : list[Value]
        Two or more values to sum.
    """

    def __init__(self, operands: list[Value]) -> None:
        if len(operands) < 2:
            raise ValueError("ManySum requires at least 2 operands.")
        self.operands = operands

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        result = self.operands[0]._eval(data, names)
        for op in self.operands[1:]:
            result = result + op._eval(data, names)
        return result

    def __repr__(self) -> str:
        return f"ManySum({self.operands!r})"


class Log(Value):
    """Natural logarithm with an optional safe fallback for non-positive inputs.

    Parameters
    ----------
    ``operand`` : Value
        Value to take the log of.
    ``safe_value`` : Value or None, default=None
        Fallback when operand is non-positive. If ``None``, non-positive
        inputs produce ``-inf``.
    """

    def __init__(self, operand: Value, safe_value: Value | None = None) -> None:
        self.operand = operand
        self.safe_value = safe_value

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        vals = self.operand._eval(data, names)
        if self.safe_value is None:
            return np.log(vals)
        safe = self.safe_value._eval(data, names)
        return np.where(vals > 0, np.log(np.where(vals > 0, vals, 1.0)), safe)

    def __repr__(self) -> str:
        return f"Log({self.operand!r}, safe_value={self.safe_value!r})"


class Count(Value):
    """Count the number of (un)satisfied constraints per sample.

    Parameters
    ----------
    ``operands`` : list[RelationConstraint]
        Constraints to count.
    ``inverse`` : bool, default=False
        If ``False``, count violated constraints. If ``True``, count
        satisfied constraints.
    """

    def __init__(self, operands: list[RelationConstraint], inverse: bool = False) -> None:
        if len(operands) < 2:
            raise ValueError("Count requires at least 2 operands.")
        self.operands = operands
        self.inverse = inverse

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        violations = np.column_stack([c.violation(data, names) for c in self.operands])
        if self.inverse:
            return np.sum(violations == 0, axis=1).astype(np.float64)
        return np.sum(violations > 0, axis=1).astype(np.float64)

    def __repr__(self) -> str:
        return f"Count({self.operands!r}, inverse={self.inverse!r})"


# ---------------------------------------------------------------------------
# Constraint hierarchy
# ---------------------------------------------------------------------------


class RelationConstraint(ABC):
    """Base class for all constraint predicates."""

    @abstractmethod
    def violation(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        """Compute the soft violation magnitude per sample.

        Parameters
        ----------
        ``data`` : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        ``names`` : list[str] or None
            Feature names for resolving string-based ``Feature`` references.

        Returns
        -------
        np.ndarray
            Violation per sample, shape ``(n_samples,)``. Zero means
            the constraint is satisfied.
        """

    def __or__(self, other: RelationConstraint) -> Or:
        if isinstance(self, Or):
            return Or(operands=[*self.operands, other])
        return Or(operands=[self, other])

    def __and__(self, other: RelationConstraint) -> And:
        if isinstance(self, And):
            return And(operands=[*self.operands, other])
        return And(operands=[self, other])


class LessEqual(RelationConstraint):
    """Constraint: ``left <= right``.

    Parameters
    ----------
    ``left`` : Value
        Left-hand side expression.
    ``right`` : Value
        Right-hand side expression.
    """

    def __init__(self, left: Value, right: Value) -> None:
        self.left = left
        self.right = right

    def violation(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        lv = self.left._eval(data, names)
        rv = self.right._eval(data, names)
        return np.maximum(0.0, lv - rv)

    def __repr__(self) -> str:
        return f"LessEqual({self.left!r}, {self.right!r})"


class Less(RelationConstraint):
    """Constraint: ``left < right``.

    Parameters
    ----------
    ``left`` : Value
        Left-hand side expression.
    ``right`` : Value
        Right-hand side expression.
    """

    _EPS = 1e-8

    def __init__(self, left: Value, right: Value) -> None:
        self.left = left
        self.right = right

    def violation(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        lv = self.left._eval(data, names)
        rv = self.right._eval(data, names)
        return np.maximum(0.0, lv - rv + self._EPS)

    def __repr__(self) -> str:
        return f"Less({self.left!r}, {self.right!r})"


class Equal(RelationConstraint):
    """Constraint: ``left == right`` (within optional tolerance).

    Parameters
    ----------
    ``left`` : Value
        Left-hand side expression.
    ``right`` : Value
        Right-hand side expression.
    ``tolerance`` : Value or None, default=None
        If provided, the constraint is satisfied when
        ``|left - right| <= tolerance``.
    """

    def __init__(self, left: Value, right: Value, tolerance: Value | None = None) -> None:
        self.left = left
        self.right = right
        self.tolerance = tolerance

    def violation(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        lv = self.left._eval(data, names)
        rv = self.right._eval(data, names)
        diff = np.abs(lv - rv)
        if self.tolerance is None:
            return diff
        tol = self.tolerance._eval(data, names)
        return np.maximum(0.0, diff - tol)

    def __repr__(self) -> str:
        return f"Equal({self.left!r}, {self.right!r}, tolerance={self.tolerance!r})"


class And(RelationConstraint):
    """Conjunction: all child constraints must hold.

    Violation is the sum of child violations.

    Parameters
    ----------
    ``operands`` : list[RelationConstraint]
        Two or more constraints.
    """

    def __init__(self, operands: list[RelationConstraint]) -> None:
        if len(operands) < 2:
            raise ValueError("And requires at least 2 operands.")
        self.operands = operands

    def violation(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        return sum(c.violation(data, names) for c in self.operands)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"And({self.operands!r})"


class Or(RelationConstraint):
    """Disjunction: at least one child constraint must hold.

    Violation is the minimum of child violations.

    Parameters
    ----------
    ``operands`` : list[RelationConstraint]
        Two or more constraints.
    """

    def __init__(self, operands: list[RelationConstraint]) -> None:
        if len(operands) < 2:
            raise ValueError("Or requires at least 2 operands.")
        self.operands = operands

    def violation(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        violations = np.column_stack([c.violation(data, names) for c in self.operands])
        return np.min(violations, axis=1)

    def __repr__(self) -> str:
        return f"Or({self.operands!r})"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ConstraintEvaluator:
    """Evaluate a relation constraint against batches of feature data.

    Parameters
    ----------
    ``constraint`` : RelationConstraint
        The constraint (or ``And``/``Or`` composite) to evaluate.
    ``feature_names`` : list[str] or None, default=None
        Feature names for resolving string-based ``Feature`` references.
    """

    def __init__(
        self,
        constraint: RelationConstraint,
        feature_names: list[str] | None = None,
    ) -> None:
        self.constraint = constraint
        self.feature_names = feature_names
        _validate_feature_refs(constraint, feature_names)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Compute violation magnitudes for a batch of samples.

        Parameters
        ----------
        ``x`` : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Violation per sample, shape ``(n_samples,)``. Zero means
            the constraint is satisfied.
        """
        return self.constraint.violation(x, self.feature_names)


# ---------------------------------------------------------------------------
# Equality repair
# ---------------------------------------------------------------------------


def repair_equality_constraints(
    counterfactuals: np.ndarray,
    constraints: list[RelationConstraint],
    feature_names: list[str] | None = None,
) -> None:
    """Repair ``Equal(Feature(i), <expr>)`` constraints in-place.

    For each ``Equal`` constraint whose left operand is a single
    ``Feature``, set the corresponding column to the value computed
    from the right-hand expression.

    Parameters
    ----------
    ``counterfactuals`` : np.ndarray
        Counterfactual matrix of shape ``(n_samples, n_features)``.
        Modified in-place.
    ``constraints`` : list[RelationConstraint]
        Constraints to scan for repairable ``Equal`` nodes.
    ``feature_names`` : list[str] or None, default=None
        Feature names for resolving string-based ``Feature`` references.
    """
    for c in constraints:
        if isinstance(c, Equal) and isinstance(c.left, Feature):
            idx = _resolve_feature(c.left.feature_id, feature_names, counterfactuals.shape[1])
            counterfactuals[:, idx] = c.right._eval(counterfactuals, feature_names)
        elif isinstance(c, And):
            repair_equality_constraints(counterfactuals, c.operands, feature_names)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce(val: Value | int | float) -> Value:
    """Wrap numeric literals in ``Constant``."""
    if isinstance(val, Value):
        return val
    if isinstance(val, (int, float)):
        return Constant(val)
    raise TypeError(f"Cannot coerce {type(val).__name__} to Value.")


def _resolve_feature(feature_id: str | int, names: list[str] | None, n_features: int) -> int:
    """Resolve a feature reference to a column index."""
    if isinstance(feature_id, int):
        if feature_id < 0 or feature_id >= n_features:
            raise IndexError(f"Feature index {feature_id} is out of range for {n_features} features.")
        return feature_id
    if names is None:
        raise ValueError(f"Feature name '{feature_id}' requires feature_names to be provided.")
    if feature_id not in names:
        raise ValueError(f"Feature name '{feature_id}' not found in feature_names: {names}.")
    return names.index(feature_id)


def _collect_features(node: Value | RelationConstraint) -> list[Feature]:
    """Recursively collect all Feature nodes from an AST."""
    features: list[Feature] = []
    if isinstance(node, Feature):
        features.append(node)
    elif isinstance(node, (MathOperation,)):
        features.extend(_collect_features(node.left))
        features.extend(_collect_features(node.right))
    elif isinstance(node, SafeDivision):
        features.extend(_collect_features(node.dividend))
        features.extend(_collect_features(node.divisor))
        features.extend(_collect_features(node.fill_value))
    elif isinstance(node, ManySum):
        for op in node.operands:
            features.extend(_collect_features(op))
    elif isinstance(node, Log):
        features.extend(_collect_features(node.operand))
        if node.safe_value is not None:
            features.extend(_collect_features(node.safe_value))
    elif isinstance(node, Count):
        for op in node.operands:
            features.extend(_collect_features(op))
    elif isinstance(node, (LessEqual, Less, Equal)):
        features.extend(_collect_features(node.left))
        features.extend(_collect_features(node.right))
        if isinstance(node, Equal) and node.tolerance is not None:
            features.extend(_collect_features(node.tolerance))
    elif isinstance(node, (And, Or)):
        for op in node.operands:
            features.extend(_collect_features(op))
    return features


def _validate_feature_refs(constraint: RelationConstraint, names: list[str] | None) -> None:
    """Validate all Feature references in a constraint tree."""
    for feat in _collect_features(constraint):
        if isinstance(feat.feature_id, str):
            if names is None:
                raise ValueError(
                    f"Feature('{feat.feature_id}') requires feature_names to be provided."
                )
            if feat.feature_id not in names:
                raise ValueError(
                    f"Feature name '{feat.feature_id}' not found in feature_names: {names}."
                )
