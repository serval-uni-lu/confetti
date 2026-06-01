"""Value expression nodes for the constraint AST."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from confetti.errors import CONFETTIConfigurationError, CONFETTIDataTypeError

if TYPE_CHECKING:
    from confetti.constraints._relations import Less, LessEqual


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
        from confetti.constraints._relations import LessEqual as _LessEqual

        return _LessEqual(left=self, right=_coerce(other))

    def __lt__(self, other: Value | int | float) -> Less:
        from confetti.constraints._relations import Less as _Less

        return _Less(left=self, right=_coerce(other))


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
        idx = resolve_feature(self.feature_id, names, data.shape[1])
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
            raise CONFETTIConfigurationError(
                message=f"Unsupported operator '{operator}'. Must be one of {sorted(_VALID_OPS)}.",
                config={"operator": operator, "left": repr(left), "right": repr(right)},
                param="operator",
                hint=f"Use one of the supported arithmetic operators: {sorted(_VALID_OPS)}.",
                source="MathOperation.__init__",
            )
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
            raise CONFETTIConfigurationError(
                message=f"ManySum requires at least 2 operands, got {len(operands)}.",
                config={"n_operands": len(operands), "operands": [repr(o) for o in operands]},
                param="operands",
                hint="Provide at least 2 Value operands to ManySum (e.g. ManySum([Feature(0), Constant(1)])).",
                source="ManySum.__init__",
            )
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


def _coerce(val: Value | int | float) -> Value:
    """Wrap numeric literals in ``Constant``."""
    if isinstance(val, Value):
        return val
    if isinstance(val, (int, float)):
        return Constant(val)
    raise CONFETTIDataTypeError(
        message=f"Cannot coerce {type(val).__name__} to Value.",
        config={"value": repr(val), "type": type(val).__name__},
        param="val",
        hint="Operands in constraint expressions must be Value nodes, int, or float.",
        source="_coerce",
    )


def resolve_feature(feature_id: str | int, names: list[str] | None, n_features: int) -> int:
    """Resolve a feature reference to a column index."""
    if isinstance(feature_id, int):
        if feature_id < 0 or feature_id >= n_features:
            raise CONFETTIConfigurationError(
                message=f"Feature index {feature_id} is out of range for {n_features} features.",
                config={"feature_id": feature_id, "n_features": n_features},
                param="feature_id",
                hint=f"Use a feature index between 0 and {n_features - 1} (inclusive).",
                source="resolve_feature",
            )
        return feature_id
    if names is None:
        raise CONFETTIConfigurationError(
            message=f"Feature('{feature_id}') uses a string name but no feature_names were provided.",
            config={"feature_id": feature_id, "feature_names": None},
            param="feature_names",
            hint="Pass feature_names to the ConstraintEvaluator or use integer indices in Feature().",
            source="resolve_feature",
        )
    if feature_id not in names:
        raise CONFETTIConfigurationError(
            message=f"Feature name '{feature_id}' not found in feature_names: {names}.",
            config={"feature_id": feature_id, "feature_names": names},
            param="feature_id",
            hint=f"Available feature names are: {names}.",
            source="resolve_feature",
        )
    return names.index(feature_id)
