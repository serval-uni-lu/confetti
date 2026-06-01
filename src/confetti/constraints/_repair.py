"""Equality repair for constraint-satisfying counterfactuals."""

from __future__ import annotations

import numpy as np

from confetti.constraints._relations import And, Equal, RelationConstraint
from confetti.constraints._values import Feature, resolve_feature


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
            idx = resolve_feature(c.left.feature_id, feature_names, counterfactuals.shape[1])
            counterfactuals[:, idx] = c.right._eval(counterfactuals, feature_names)
        elif isinstance(c, And):
            repair_equality_constraints(counterfactuals, c.operands, feature_names)
