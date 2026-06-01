"""Validation utilities for constraint AST trees."""

from __future__ import annotations

from confetti.errors import CONFETTIConfigurationError

from confetti.constraints._relations import (
    And,
    Count,
    Equal,
    Less,
    LessEqual,
    Or,
    RelationConstraint,
)
from confetti.constraints._values import (
    Feature,
    Log,
    ManySum,
    MathOperation,
    SafeDivision,
    Value,
)


def collect_features(node: Value | RelationConstraint) -> list[Feature]:
    """Recursively collect all Feature nodes from an AST."""
    features: list[Feature] = []
    if isinstance(node, Feature):
        features.append(node)
    elif isinstance(node, (MathOperation,)):
        features.extend(collect_features(node.left))
        features.extend(collect_features(node.right))
    elif isinstance(node, SafeDivision):
        features.extend(collect_features(node.dividend))
        features.extend(collect_features(node.divisor))
        features.extend(collect_features(node.fill_value))
    elif isinstance(node, ManySum):
        for op in node.operands:
            features.extend(collect_features(op))
    elif isinstance(node, Log):
        features.extend(collect_features(node.operand))
        if node.safe_value is not None:
            features.extend(collect_features(node.safe_value))
    elif isinstance(node, Count):
        for op in node.operands:
            features.extend(collect_features(op))
    elif isinstance(node, (LessEqual, Less, Equal)):
        features.extend(collect_features(node.left))
        features.extend(collect_features(node.right))
        if isinstance(node, Equal) and node.tolerance is not None:
            features.extend(collect_features(node.tolerance))
    elif isinstance(node, (And, Or)):
        for op in node.operands:
            features.extend(collect_features(op))
    return features


def validate_feature_refs(constraint: RelationConstraint, names: list[str] | None) -> None:
    """Validate all Feature references in a constraint tree."""
    for feat in collect_features(constraint):
        if isinstance(feat.feature_id, str):
            if names is None:
                raise CONFETTIConfigurationError(
                    message=f"Feature('{feat.feature_id}') uses a string name but no feature_names were provided.",
                    config={"feature_id": feat.feature_id, "feature_names": None},
                    param="feature_names",
                    hint="Pass feature_names to the ConstraintEvaluator or use integer indices in Feature().",
                    source="validate_feature_refs",
                )
            if feat.feature_id not in names:
                raise CONFETTIConfigurationError(
                    message=f"Feature name '{feat.feature_id}' not found in feature_names: {names}.",
                    config={"feature_id": feat.feature_id, "feature_names": names},
                    param="feature_id",
                    hint=f"Available feature names are: {names}.",
                    source="validate_feature_refs",
                )
