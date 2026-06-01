"""Constraint evaluator with Rust-accelerated bytecode backend."""

from __future__ import annotations

import struct

import numpy as np

from confetti.constraints._relations import (
    And,
    Count,
    Equal,
    Less,
    LessEqual,
    Or,
    RelationConstraint,
)
from confetti.constraints._validation import validate_feature_refs
from confetti.constraints._values import (
    Constant,
    Feature,
    Log,
    ManySum,
    MathOperation,
    SafeDivision,
    Value,
)

try:
    from confetti._rust_core import evaluate_constraints as _rs_evaluate_constraints

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# Opcodes must match src/constraints/evaluator.rs exactly.
_OP_CONST = 1
_OP_FEATURE = 2
_OP_ADD = 3
_OP_SUB = 4
_OP_MUL = 5
_OP_DIV = 6
_OP_POW = 7
_OP_MOD = 8
_OP_SAFE_DIV = 9
_OP_LOG = 10
_OP_LOG_SAFE = 11
_OP_LESS_EQUAL = 20
_OP_LESS = 21
_OP_EQUAL = 22
_OP_EQUAL_TOL = 23
_OP_AND = 24
_OP_OR = 25
_OP_COUNT = 26
_OP_COUNT_INV = 27

_MATH_OP_CODES = {
    "+": _OP_ADD,
    "-": _OP_SUB,
    "*": _OP_MUL,
    "/": _OP_DIV,
    "**": _OP_POW,
    "%": _OP_MOD,
}


class ConstraintEvaluator:
    """Evaluate a relation constraint against batches of feature data.

    Compiles the constraint AST to bytecode at construction time.
    When the Rust extension is available, evaluation runs in Rust
    with Rayon parallelism across samples; otherwise falls back to
    the pure-Python AST walk.

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
        validate_feature_refs(constraint, feature_names)

        self._bytecode: np.ndarray | None = None
        self._constants: np.ndarray | None = None
        self._feature_indices: np.ndarray | None = None

        if _HAS_RUST:
            compiler = _BytecodeCompiler(feature_names)
            compiler.compile(constraint)
            self._bytecode = np.array(compiler.bytecode, dtype=np.uint8)
            self._constants = np.array(compiler.constants, dtype=np.float64)
            self._feature_indices = np.array(compiler.feature_indices, dtype=np.uint32)

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
        if self._bytecode is not None:
            data = np.ascontiguousarray(x, dtype=np.float64)
            return np.asarray(
                _rs_evaluate_constraints(data, self._bytecode, self._constants, self._feature_indices)
            )
        return self.constraint.violation(x, self.feature_names)


class _BytecodeCompiler:
    """Compile a constraint AST into a flat instruction stream for the Rust evaluator."""

    def __init__(self, feature_names: list[str] | None) -> None:
        self.bytecode: list[int] = []
        self.constants: list[float] = []
        self.feature_indices: list[int] = []
        self._names = feature_names

    def _emit_op(self, opcode: int) -> None:
        self.bytecode.append(opcode)

    def _emit_op_u32(self, opcode: int, arg: int) -> None:
        self.bytecode.append(opcode)
        self.bytecode.extend(struct.pack("<I", arg))

    def _add_constant(self, value: float) -> int:
        idx = len(self.constants)
        self.constants.append(value)
        return idx

    def _add_feature_index(self, col_idx: int) -> int:
        idx = len(self.feature_indices)
        self.feature_indices.append(col_idx)
        return idx

    def compile(self, node: Value | RelationConstraint) -> None:
        """Recursively compile an AST node to bytecode."""
        if isinstance(node, Constant):
            self._emit_op_u32(_OP_CONST, self._add_constant(float(node.value)))

        elif isinstance(node, Feature):
            if isinstance(node.feature_id, str):
                assert self._names is not None
                col = self._names.index(node.feature_id)
            else:
                col = node.feature_id
            self._emit_op_u32(_OP_FEATURE, self._add_feature_index(col))

        elif isinstance(node, MathOperation):
            self.compile(node.left)
            self.compile(node.right)
            self._emit_op(_MATH_OP_CODES[node.operator])

        elif isinstance(node, SafeDivision):
            self.compile(node.dividend)
            self.compile(node.divisor)
            self.compile(node.fill_value)
            self._emit_op(_OP_SAFE_DIV)

        elif isinstance(node, ManySum):
            self.compile(node.operands[0])
            for operand in node.operands[1:]:
                self.compile(operand)
                self._emit_op(_OP_ADD)

        elif isinstance(node, Log):
            if node.safe_value is None:
                self.compile(node.operand)
                self._emit_op(_OP_LOG)
            else:
                self.compile(node.operand)
                self.compile(node.safe_value)
                self._emit_op(_OP_LOG_SAFE)

        elif isinstance(node, Count):
            for operand in node.operands:
                self.compile(operand)
            op = _OP_COUNT_INV if node.inverse else _OP_COUNT
            self._emit_op_u32(op, len(node.operands))

        elif isinstance(node, LessEqual):
            self.compile(node.left)
            self.compile(node.right)
            self._emit_op(_OP_LESS_EQUAL)

        elif isinstance(node, Less):
            self.compile(node.left)
            self.compile(node.right)
            self._emit_op(_OP_LESS)

        elif isinstance(node, Equal):
            self.compile(node.left)
            self.compile(node.right)
            if node.tolerance is not None:
                self.compile(node.tolerance)
                self._emit_op(_OP_EQUAL_TOL)
            else:
                self._emit_op(_OP_EQUAL)

        elif isinstance(node, And):
            for operand in node.operands:
                self.compile(operand)
            self._emit_op_u32(_OP_AND, len(node.operands))

        elif isinstance(node, Or):
            for operand in node.operands:
                self.compile(operand)
            self._emit_op_u32(_OP_OR, len(node.operands))

        else:
            raise TypeError(f"Cannot compile {type(node).__name__} to bytecode.")
