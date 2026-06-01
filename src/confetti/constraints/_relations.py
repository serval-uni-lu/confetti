"""Relation constraint nodes for the constraint AST."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from confetti.constraints._values import Value
from confetti.errors import CONFETTIConfigurationError


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
            raise CONFETTIConfigurationError(
                message=f"And requires at least 2 operands, got {len(operands)}.",
                config={"n_operands": len(operands), "operands": [repr(o) for o in operands]},
                param="operands",
                hint="Combine at least 2 RelationConstraint instances (e.g. And([c1, c2])).",
                source="And.__init__",
            )
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
            raise CONFETTIConfigurationError(
                message=f"Or requires at least 2 operands, got {len(operands)}.",
                config={"n_operands": len(operands), "operands": [repr(o) for o in operands]},
                param="operands",
                hint="Combine at least 2 RelationConstraint instances (e.g. Or([c1, c2])).",
                source="Or.__init__",
            )
        self.operands = operands

    def violation(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        violations = np.column_stack([c.violation(data, names) for c in self.operands])
        return np.min(violations, axis=1)

    def __repr__(self) -> str:
        return f"Or({self.operands!r})"


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
            raise CONFETTIConfigurationError(
                message=f"Count requires at least 2 operands, got {len(operands)}.",
                config={"n_operands": len(operands), "inverse": inverse, "operands": [repr(o) for o in operands]},
                param="operands",
                hint="Provide at least 2 RelationConstraint instances to Count (e.g. Count([c1, c2])).",
                source="Count.__init__",
            )
        self.operands = operands
        self.inverse = inverse

    def _eval(self, data: np.ndarray, names: list[str] | None) -> np.ndarray:
        violations = np.column_stack([c.violation(data, names) for c in self.operands])
        if self.inverse:
            return np.sum(violations == 0, axis=1).astype(np.float64)
        return np.sum(violations > 0, axis=1).astype(np.float64)

    def __repr__(self) -> str:
        return f"Count({self.operands!r}, inverse={self.inverse!r})"
